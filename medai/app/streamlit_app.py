"""MedAI Streamlit demo application."""

import json
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="MedAI — Clinical Intelligence Platform",
    page_icon="🏥",
    layout="wide",
)

# ------------------------------------------------------------------ #
# Custom CSS — Navy/Teal scheme
# ------------------------------------------------------------------ #
st.markdown(
    """
<style>
  .main-header {
    background: #0B1F3A;
    color: #FFFFFF;
    padding: 18px 24px;
    border-radius: 8px;
    margin-bottom: 20px;
  }
  .main-header h1 { margin: 0; font-size: 2em; }
  .main-header p  { margin: 4px 0 0; color: #9BBBD4; font-size: 0.95em; }
  .section-card {
    background: #F0F4F8;
    border-left: 4px solid #0D7377;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
  }
  .urgency-CRITICAL { background:#C0392B; color:white; padding:8px 16px; border-radius:6px; font-weight:bold; }
  .urgency-HIGH     { background:#E67E22; color:white; padding:8px 16px; border-radius:6px; font-weight:bold; }
  .urgency-MEDIUM   { background:#F1C40F; color:#333;  padding:8px 16px; border-radius:6px; font-weight:bold; }
  .urgency-LOW      { background:#27AE60; color:white; padding:8px 16px; border-radius:6px; font-weight:bold; }
  .stTabs [data-baseweb="tab"] { font-size: 1em; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
  <h1>MedAI</h1>
  <p>Clinical Intelligence Platform — Conversation → Structured Records</p>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "cleaned_transcript" not in st.session_state:
    st.session_state.cleaned_transcript = None

# ------------------------------------------------------------------ #
# Tabs
# ------------------------------------------------------------------ #
tab1, tab2, tab3 = st.tabs(["📥 Input", "🩺 Clinical Analysis", "📄 Reports"])

# ================================================================== #
# TAB 1 — INPUT
# ================================================================== #
with tab1:
    st.subheader("Upload or Paste Consultation")
    input_mode = st.radio(
        "Input type",
        ["Audio file", "Prefixed text transcript", "Raw text transcript"],
        horizontal=True,
    )

    raw_input = None
    input_format = "auto"

    if input_mode == "Audio file":
        audio_file = st.file_uploader(
            "Upload audio (WAV, MP3, FLAC, M4A)", type=["wav", "mp3", "flac", "m4a"]
        )
        if audio_file:
            st.audio(audio_file)
            if st.button("Transcribe Audio"):
                with st.spinner("Transcribing…"):
                    try:
                        import tempfile, os
                        from medai.pipeline.asr.transcribe import ASRTranscriber
                        from medai.pipeline.asr.diarise import SpeakerDiariser

                        with tempfile.NamedTemporaryFile(
                            suffix=os.path.splitext(audio_file.name)[1], delete=False
                        ) as tmp:
                            tmp.write(audio_file.read())
                            tmp_path = tmp.name

                        transcriber = ASRTranscriber()
                        result = transcriber.transcribe(tmp_path)
                        os.unlink(tmp_path)

                        raw_input = result.text
                        input_format = "raw_asr"
                        st.success(f"Transcribed ({result.model_used})")
                        st.text_area("Transcript preview", result.text, height=150)
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")

    elif input_mode == "Prefixed text transcript":
        raw_input = st.text_area(
            "Paste transcript (Doctor: ... / Patient: ... format)",
            height=250,
            placeholder="Doctor: What brings you in today?\nPatient: I've been having chest pain...",
        )
        input_format = "prefixed"

    else:
        raw_input = st.text_area(
            "Paste raw transcript text",
            height=250,
            placeholder="Doctor asked about the pain. Patient said it started three days ago...",
        )
        input_format = "raw_asr"

    if raw_input and st.button("▶ Process Transcript", type="primary"):
        with st.spinner("Processing transcript…"):
            try:
                from medai.pipeline.clean.text_cleaner import TranscriptCleaner
                from medai.pipeline.clean.deidentify import DeIdentifier
                from medai.pipeline.identify.role_classifier import RoleClassifier

                cleaner = TranscriptCleaner()
                cleaned = cleaner.clean(raw_input, input_format)

                deidentifier = DeIdentifier()
                try:
                    turns_deid = deidentifier.deidentify_turns(cleaned.turns)
                    from medai.pipeline.clean.text_cleaner import CleanedTranscript
                    cleaned = CleanedTranscript(
                        turns=turns_deid,
                        raw_text=cleaned.raw_text,
                        cleaned_text=" ".join(t.text for t in turns_deid),
                        input_format_detected=cleaned.input_format_detected,
                        num_turns=cleaned.num_turns,
                    )
                except Exception:
                    pass

                role_cls = RoleClassifier()
                if role_cls.available:
                    turns = role_cls.classify_transcript(cleaned.turns)
                    from medai.pipeline.clean.text_cleaner import CleanedTranscript
                    cleaned = CleanedTranscript(
                        turns=turns,
                        raw_text=cleaned.raw_text,
                        cleaned_text=cleaned.cleaned_text,
                        input_format_detected=cleaned.input_format_detected,
                        num_turns=cleaned.num_turns,
                    )

                st.session_state.cleaned_transcript = cleaned
                st.success(f"Processed {cleaned.num_turns} turns.")

                with st.expander("Preview turns"):
                    for t in cleaned.turns:
                        role_label = f" [{t.role}]" if t.role else ""
                        st.write(f"**{t.speaker_id}{role_label}:** {t.text}")

            except Exception as e:
                st.error(f"Processing failed: {e}")

    if st.session_state.cleaned_transcript and st.button(
        "🔬 Run Full Analysis", type="primary"
    ):
        with st.spinner("Running clinical analysis (SOAP + DDx + Risk + Treatment)…"):
            try:
                cleaned = st.session_state.cleaned_transcript

                from medai.pipeline.intelligence.soap_generator import SOAPGenerator
                from medai.pipeline.identify.ner import ClinicalNER
                from medai.pipeline.identify.negation import NegationDetector
                from medai.pipeline.intelligence.ddx_engine import DDxEngine
                from medai.pipeline.intelligence.risk_stratifier import RiskStratifier
                from medai.pipeline.intelligence.treatment import TreatmentRecommender

                soap_gen = SOAPGenerator()
                soap = soap_gen.generate(cleaned)

                ner = ClinicalNER()
                ner_result = ner.extract(cleaned.cleaned_text)

                neg = NegationDetector()
                if neg.available:
                    ner_result.entities = neg.detect(cleaned.cleaned_text, ner_result.entities)

                ddx_engine = DDxEngine()
                ddx = ddx_engine.generate_ddx(ner_result, soap.assessment)

                risk = RiskStratifier().stratify(ner_result, ddx, soap.assessment)
                treatment = TreatmentRecommender().recommend(soap, ddx, risk)

                st.session_state.analysis_result = {
                    "soap": soap,
                    "ddx": ddx,
                    "risk": risk,
                    "treatment": treatment,
                }
                st.success("Analysis complete! Switch to the Clinical Analysis tab.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ================================================================== #
# TAB 2 — CLINICAL ANALYSIS
# ================================================================== #
with tab2:
    if not st.session_state.analysis_result:
        st.info("Process a transcript in the Input tab first.")
    else:
        result = st.session_state.analysis_result
        soap = result["soap"]
        ddx = result["ddx"]
        risk = result["risk"]
        treatment = result["treatment"]

        # SOAP note
        st.subheader("SOAP Note")
        with st.expander("Subjective", expanded=True):
            st.markdown(soap.subjective)
        with st.expander("Objective"):
            st.markdown(soap.objective)
        with st.expander("Assessment"):
            st.markdown(soap.assessment)
        with st.expander("Plan"):
            st.markdown(soap.plan)

        if soap.icd_codes or soap.hcc_codes:
            st.subheader("Clinical Codes")
            cols = st.columns(len(soap.icd_codes) + len(soap.hcc_codes) or 1)
            for i, code in enumerate(soap.icd_codes):
                cols[i].markdown(
                    f"<span style='background:#0B1F3A;color:white;padding:4px 10px;"
                    f"border-radius:12px;font-size:0.85em'>ICD-10: {code}</span>",
                    unsafe_allow_html=True,
                )
            for j, code in enumerate(soap.hcc_codes):
                cols[len(soap.icd_codes) + j].markdown(
                    f"<span style='background:#0D7377;color:white;padding:4px 10px;"
                    f"border-radius:12px;font-size:0.85em'>HCC {code}</span>",
                    unsafe_allow_html=True,
                )

        # DDx
        st.subheader("Differential Diagnoses")
        if ddx.differentials:
            import pandas as pd
            rows = [
                {
                    "Rank": d.rank,
                    "Diagnosis": d.disease_name,
                    "ICD-10": d.icd10_code,
                    "Confidence": round(d.confidence_score, 2),
                    "KB Score": round(d.kb_score, 3),
                    "Reasoning": d.reasoning[:120] + "..." if len(d.reasoning) > 120 else d.reasoning,
                }
                for d in ddx.differentials
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No differentials generated.")

        # Risk
        st.subheader("Risk Assessment")
        urgency = risk.urgency
        st.markdown(
            f"<div class='urgency-{urgency}'>Urgency: {urgency}</div>",
            unsafe_allow_html=True,
        )
        if risk.red_flags:
            st.warning("**Red Flags Detected:**")
            for flag in risk.red_flags:
                st.markdown(f"- ⚠ {flag}")
        st.write(f"**Recommended Action:** {risk.recommended_action}")
        st.write(f"**Reasoning:** {risk.reasoning}")

        # Treatment
        st.subheader("Treatment Recommendations")
        if treatment.recommendations:
            for rec in treatment.recommendations:
                with st.expander(f"{rec.diagnosis} — {rec.priority}"):
                    st.write(f"**Intervention:** {rec.intervention}")
                    st.write(f"**Rationale:** {rec.rationale}")

        if treatment.medications:
            st.markdown("**Medications**")
            import pandas as pd
            med_rows = [
                {
                    "Medication": m.name,
                    "Dose": m.dose,
                    "Frequency": m.frequency,
                    "Duration": m.duration,
                    "Notes": m.notes,
                }
                for m in treatment.medications
            ]
            st.dataframe(pd.DataFrame(med_rows), use_container_width=True)

        if treatment.follow_up:
            st.info(f"**Follow-up:** {treatment.follow_up}")

# ================================================================== #
# TAB 3 — REPORTS
# ================================================================== #
with tab3:
    if not st.session_state.analysis_result:
        st.info("Complete analysis in the Input tab first.")
    else:
        result = st.session_state.analysis_result
        soap = result["soap"]
        ddx = result["ddx"]
        risk = result["risk"]
        treatment = result["treatment"]

        st.subheader("Download Reports")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📋 Doctor Report (PDF)"):
                try:
                    from medai.reports.pdf_generator import PDFGenerator
                    gen = PDFGenerator()
                    pdf_bytes = gen.generate_doctor_report(soap, ddx, risk, treatment)
                    st.download_button(
                        "⬇ Download Doctor Report",
                        data=pdf_bytes,
                        file_name="medai_doctor_report.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        with col2:
            if st.button("🧑 Patient Summary (PDF)"):
                try:
                    from medai.reports.pdf_generator import PDFGenerator
                    gen = PDFGenerator()
                    pdf_bytes = gen.generate_patient_summary(soap, treatment)
                    st.download_button(
                        "⬇ Download Patient Summary",
                        data=pdf_bytes,
                        file_name="medai_patient_summary.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        with col3:
            if st.button("🔗 FHIR R4 Bundle (JSON)"):
                try:
                    from medai.reports.fhir_builder import FHIRBuilder
                    builder = FHIRBuilder()
                    bundle = builder.build_bundle(soap, ddx, risk, treatment)
                    fhir_json = json.dumps(bundle, indent=2)
                    st.download_button(
                        "⬇ Download FHIR Bundle",
                        data=fhir_json,
                        file_name="medai_fhir_bundle.json",
                        mime="application/json",
                    )
                except Exception as e:
                    st.error(f"FHIR generation failed: {e}")

        st.subheader("Raw JSON Analysis")
        with st.expander("View full analysis JSON"):
            import dataclasses

            def _to_json_serialisable(obj):
                if dataclasses.is_dataclass(obj):
                    return dataclasses.asdict(obj)
                return str(obj)

            st.json(json.loads(json.dumps(
                {
                    "soap": dataclasses.asdict(soap),
                    "ddx": dataclasses.asdict(ddx),
                    "risk": dataclasses.asdict(risk),
                    "treatment": dataclasses.asdict(treatment),
                },
                default=_to_json_serialisable,
            )))
