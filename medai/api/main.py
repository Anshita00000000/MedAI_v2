"""MedAI FastAPI application."""

import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedAI Clinical Intelligence Platform",
    description="Converts doctor-patient conversations into structured medical records.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Lazy-load heavy pipeline components on first request
# ------------------------------------------------------------------ #

_components: Dict[str, Any] = {}


def _get_component(name: str):
    if name not in _components:
        if name == "transcriber":
            from medai.pipeline.asr.transcribe import ASRTranscriber
            _components[name] = ASRTranscriber()
        elif name == "diariser":
            from medai.pipeline.asr.diarise import SpeakerDiariser
            _components[name] = SpeakerDiariser()
        elif name == "cleaner":
            from medai.pipeline.clean.text_cleaner import TranscriptCleaner
            _components[name] = TranscriptCleaner()
        elif name == "deidentifier":
            from medai.pipeline.clean.deidentify import DeIdentifier
            _components[name] = DeIdentifier()
        elif name == "role_classifier":
            from medai.pipeline.identify.role_classifier import RoleClassifier
            _components[name] = RoleClassifier()
        elif name == "ner":
            from medai.pipeline.identify.ner import ClinicalNER
            _components[name] = ClinicalNER()
        elif name == "negation":
            from medai.pipeline.identify.negation import NegationDetector
            _components[name] = NegationDetector()
        elif name == "soap_generator":
            from medai.pipeline.intelligence.soap_generator import SOAPGenerator
            _components[name] = SOAPGenerator()
        elif name == "ddx_engine":
            from medai.pipeline.intelligence.ddx_engine import DDxEngine
            _components[name] = DDxEngine()
        elif name == "risk_stratifier":
            from medai.pipeline.intelligence.risk_stratifier import RiskStratifier
            _components[name] = RiskStratifier()
        elif name == "treatment":
            from medai.pipeline.intelligence.treatment import TreatmentRecommender
            _components[name] = TreatmentRecommender()
        elif name == "pdf_generator":
            from medai.reports.pdf_generator import PDFGenerator
            _components[name] = PDFGenerator()
        elif name == "fhir_builder":
            from medai.reports.fhir_builder import FHIRBuilder
            _components[name] = FHIRBuilder()
    return _components.get(name)


# ------------------------------------------------------------------ #
# Request / Response models
# ------------------------------------------------------------------ #

class TranscriptRequest(BaseModel):
    text: str
    input_format: str = "auto"


class GenerateSOAPRequest(BaseModel):
    cleaned_transcript: dict
    mode: str = "sectional"


class AnalyseRequest(BaseModel):
    cleaned_transcript: dict


class ReportRequest(BaseModel):
    analysis: dict
    report_type: str = "doctor"  # "doctor", "patient", "fhir"


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    loaded = {name: True for name in _components}
    return {"status": "healthy", "models_loaded": loaded}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Upload audio file → TranscriptionResult + DiarisationResult."""
    transcriber = _get_component("transcriber")
    diariser = _get_component("diariser")

    if not transcriber or not transcriber.available:
        raise HTTPException(503, "ASR backend not available.")

    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(audio.filename)[1] or ".wav", delete=False
    ) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        transcription = transcriber.transcribe(tmp_path)
        result: dict = {
            "transcription": {
                "text": transcription.text,
                "segments": transcription.segments,
                "language": transcription.language,
                "model_used": transcription.model_used,
                "duration_seconds": transcription.duration_seconds,
            }
        }
        if diariser and diariser.available:
            try:
                diarisation = diariser.diarise(tmp_path)
                result["diarisation"] = {
                    "segments": diarisation.segments,
                    "num_speakers": diarisation.num_speakers,
                    "model_used": diarisation.model_used,
                }
            except Exception as exc:
                logger.warning("Diarisation failed: %s", exc)
                result["diarisation"] = None
        else:
            result["diarisation"] = None
    finally:
        os.unlink(tmp_path)

    return result


@app.post("/process-transcript")
def process_transcript(req: TranscriptRequest):
    """Process raw/prefixed text transcript → CleanedTranscript with roles."""
    cleaner = _get_component("cleaner")
    role_classifier = _get_component("role_classifier")

    cleaned = cleaner.clean(req.text, req.input_format)

    if role_classifier and role_classifier.available:
        try:
            turns_with_roles = role_classifier.classify_transcript(cleaned.turns)
            cleaned = cleaned.__class__(
                turns=turns_with_roles,
                raw_text=cleaned.raw_text,
                cleaned_text=cleaned.cleaned_text,
                input_format_detected=cleaned.input_format_detected,
                num_turns=cleaned.num_turns,
            )
        except Exception as exc:
            logger.warning("Role classification failed: %s", exc)

    return {
        "turns": [
            {
                "speaker_id": t.speaker_id,
                "role": t.role,
                "text": t.text,
                "start": t.start,
                "end": t.end,
            }
            for t in cleaned.turns
        ],
        "cleaned_text": cleaned.cleaned_text,
        "input_format_detected": cleaned.input_format_detected,
        "num_turns": cleaned.num_turns,
    }


@app.post("/generate-soap")
def generate_soap(req: GenerateSOAPRequest):
    """Generate SOAP note from a cleaned transcript."""
    from medai.pipeline.clean.text_cleaner import CleanedTranscript

    soap_generator = _get_component("soap_generator")
    if not soap_generator or not soap_generator.available:
        raise HTTPException(503, "SOAP generator not available.")

    ct = _dict_to_cleaned_transcript(req.cleaned_transcript)
    soap = soap_generator.generate(ct, mode=req.mode)

    return _soap_to_dict(soap)


@app.post("/analyse")
def analyse(req: AnalyseRequest):
    """Full analysis pipeline: SOAP + DDx + Risk + Treatment."""
    ct = _dict_to_cleaned_transcript(req.cleaned_transcript)

    soap_generator = _get_component("soap_generator")
    ner = _get_component("ner")
    negation = _get_component("negation")
    ddx_engine = _get_component("ddx_engine")
    risk_stratifier = _get_component("risk_stratifier")
    treatment_recommender = _get_component("treatment")

    if not soap_generator or not soap_generator.available:
        raise HTTPException(503, "SOAP generator not available.")

    soap = soap_generator.generate(ct)

    if not ner or not ner.available:
        raise HTTPException(503, "NER not available.")

    ner_result = ner.extract(ct.cleaned_text)
    if negation and negation.available:
        try:
            ner_result.entities = negation.detect(ct.cleaned_text, ner_result.entities)
        except Exception as exc:
            logger.warning("Negation detection failed: %s", exc)

    if not ddx_engine or not ddx_engine.available:
        raise HTTPException(503, "DDx engine not available.")
    ddx = ddx_engine.generate_ddx(ner_result, soap.assessment)

    if not risk_stratifier or not risk_stratifier.available:
        raise HTTPException(503, "Risk stratifier not available.")
    risk = risk_stratifier.stratify(ner_result, ddx, soap.assessment)

    if not treatment_recommender or not treatment_recommender.available:
        raise HTTPException(503, "Treatment recommender not available.")
    treatment = treatment_recommender.recommend(soap, ddx, risk)

    return {
        "soap": _soap_to_dict(soap),
        "ddx": _ddx_to_dict(ddx),
        "risk": _risk_to_dict(risk),
        "treatment": _treatment_to_dict(treatment),
    }


@app.post("/generate-report")
def generate_report(req: ReportRequest):
    """Generate PDF or FHIR bundle from analysis result."""
    analysis = req.analysis
    soap = _dict_to_soap(analysis.get("soap", {}))
    ddx = _dict_to_ddx(analysis.get("ddx", {}))
    risk = _dict_to_risk(analysis.get("risk", {}))
    treatment = _dict_to_treatment(analysis.get("treatment", {}))

    if req.report_type == "fhir":
        fhir_builder = _get_component("fhir_builder")
        bundle = fhir_builder.build_bundle(soap, ddx, risk, treatment)
        return bundle

    pdf_generator = _get_component("pdf_generator")
    if not pdf_generator or not pdf_generator.available:
        raise HTTPException(503, "PDF generator not available.")

    from fastapi.responses import Response
    if req.report_type == "patient":
        pdf_bytes = pdf_generator.generate_patient_summary(soap, treatment)
    else:
        pdf_bytes = pdf_generator.generate_doctor_report(soap, ddx, risk, treatment)

    return Response(content=pdf_bytes, media_type="application/pdf")


# ------------------------------------------------------------------ #
# Serialisation helpers
# ------------------------------------------------------------------ #

def _dict_to_cleaned_transcript(d: dict):
    from medai.pipeline.clean.text_cleaner import CleanedTranscript
    from medai.pipeline.asr.diarise import SpeakerTurn
    turns = [
        SpeakerTurn(
            speaker_id=t.get("speaker_id", ""),
            role=t.get("role", ""),
            text=t.get("text", ""),
            start=t.get("start", 0.0),
            end=t.get("end", 0.0),
        )
        for t in d.get("turns", [])
    ]
    return CleanedTranscript(
        turns=turns,
        raw_text=d.get("raw_text", d.get("cleaned_text", "")),
        cleaned_text=d.get("cleaned_text", ""),
        input_format_detected=d.get("input_format_detected", ""),
        num_turns=len(turns),
    )


def _soap_to_dict(soap) -> dict:
    return {
        "subjective": soap.subjective,
        "objective": soap.objective,
        "assessment": soap.assessment,
        "plan": soap.plan,
        "model_used": soap.model_used,
        "generation_time_ms": soap.generation_time_ms,
        "icd_codes": soap.icd_codes,
        "hcc_codes": soap.hcc_codes,
    }


def _ddx_to_dict(ddx) -> dict:
    return {
        "differentials": [
            {
                "rank": d.rank,
                "disease_name": d.disease_name,
                "umls_code": d.umls_code,
                "icd10_code": d.icd10_code,
                "confidence_score": d.confidence_score,
                "kb_score": d.kb_score,
                "reasoning": d.reasoning,
                "supporting_symptoms": d.supporting_symptoms,
            }
            for d in ddx.differentials
        ],
        "method_used": ddx.method_used,
        "model_used": ddx.model_used,
    }


def _risk_to_dict(risk) -> dict:
    return {
        "urgency": risk.urgency,
        "red_flags": risk.red_flags,
        "recommended_action": risk.recommended_action,
        "reasoning": risk.reasoning,
        "model_used": risk.model_used,
        "rag_used": risk.rag_used,
    }


def _treatment_to_dict(treatment) -> dict:
    return {
        "recommendations": [
            {
                "diagnosis": r.diagnosis,
                "intervention": r.intervention,
                "priority": r.priority,
                "rationale": r.rationale,
            }
            for r in treatment.recommendations
        ],
        "medications": [
            {
                "name": m.name,
                "dose": m.dose,
                "frequency": m.frequency,
                "duration": m.duration,
                "notes": m.notes,
            }
            for m in treatment.medications
        ],
        "follow_up": treatment.follow_up,
        "referrals": treatment.referrals,
        "patient_education": treatment.patient_education,
        "model_used": treatment.model_used,
    }


def _dict_to_soap(d: dict):
    from medai.pipeline.intelligence.soap_generator import SOAPNote
    return SOAPNote(
        subjective=d.get("subjective", ""),
        objective=d.get("objective", ""),
        assessment=d.get("assessment", ""),
        plan=d.get("plan", ""),
        model_used=d.get("model_used", ""),
        generation_time_ms=d.get("generation_time_ms", 0.0),
        icd_codes=d.get("icd_codes", []),
        hcc_codes=d.get("hcc_codes", []),
    )


def _dict_to_ddx(d: dict):
    from medai.pipeline.intelligence.ddx_engine import DDxResult, Differential
    diffs = [
        Differential(
            rank=item.get("rank", i + 1),
            disease_name=item.get("disease_name", ""),
            umls_code=item.get("umls_code", ""),
            icd10_code=item.get("icd10_code", ""),
            confidence_score=item.get("confidence_score", 0.0),
            kb_score=item.get("kb_score", 0.0),
            reasoning=item.get("reasoning", ""),
            supporting_symptoms=item.get("supporting_symptoms", []),
        )
        for i, item in enumerate(d.get("differentials", []))
    ]
    return DDxResult(
        differentials=diffs,
        method_used=d.get("method_used", "llm_only"),
        kb_candidates=d.get("kb_candidates", []),
        model_used=d.get("model_used", ""),
    )


def _dict_to_risk(d: dict):
    from medai.pipeline.intelligence.risk_stratifier import RiskResult
    return RiskResult(
        urgency=d.get("urgency", "MEDIUM"),
        red_flags=d.get("red_flags", []),
        recommended_action=d.get("recommended_action", ""),
        reasoning=d.get("reasoning", ""),
        model_used=d.get("model_used", ""),
        rag_used=d.get("rag_used", False),
    )


def _dict_to_treatment(d: dict):
    from medai.pipeline.intelligence.treatment import TreatmentResult, Recommendation, Medication
    recs = [Recommendation(**r) for r in d.get("recommendations", [])]
    meds = [Medication(**m) for m in d.get("medications", [])]
    return TreatmentResult(
        recommendations=recs,
        medications=meds,
        follow_up=d.get("follow_up", ""),
        referrals=d.get("referrals", []),
        patient_education=d.get("patient_education", []),
        model_used=d.get("model_used", ""),
    )
