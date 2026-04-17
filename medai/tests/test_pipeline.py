"""
End-to-end pipeline data-flow test using mocked LLM calls.
No real models needed — verifies dataclasses and pipeline wiring.
"""

import json
import unittest
from unittest.mock import MagicMock, patch


# ------------------------------------------------------------------ #
# Sample transcript
# ------------------------------------------------------------------ #
SAMPLE_TRANSCRIPT = """Doctor: What brings you in today?
Patient: I've been having chest pain for three days. It gets worse when I breathe deeply.
Doctor: Any shortness of breath or fever?
Patient: A little breathless, no fever. I also smoke about 10 cigarettes a day.
Doctor: I'll order an ECG. Your blood pressure is 140 over 90 today.
Patient: Is that high?
Doctor: Yes, a little elevated. I want you to start aspirin 75mg daily.
"""


class TestTranscriptCleaner(unittest.TestCase):

    def setUp(self):
        from medai.pipeline.clean.text_cleaner import TranscriptCleaner
        self.cleaner = TranscriptCleaner()

    def test_detect_prefixed_format(self):
        cleaned = self.cleaner.clean(SAMPLE_TRANSCRIPT, "auto")
        self.assertEqual(cleaned.input_format_detected, "prefixed")

    def test_turns_populated(self):
        cleaned = self.cleaner.clean(SAMPLE_TRANSCRIPT, "prefixed")
        self.assertGreater(cleaned.num_turns, 0)

    def test_speaker_ids(self):
        cleaned = self.cleaner.clean(SAMPLE_TRANSCRIPT, "prefixed")
        speaker_ids = {t.speaker_id for t in cleaned.turns}
        self.assertIn("DOCTOR", speaker_ids)
        self.assertIn("PATIENT", speaker_ids)

    def test_filler_word_removal(self):
        result = self.cleaner.clean_raw_asr("um so the patient uh said he was fine")
        self.assertNotIn("um", result)
        self.assertNotIn("uh", result)

    def test_repeated_word_removal(self):
        result = self.cleaner.clean_raw_asr("the the patient said pain pain chest")
        self.assertNotIn("the the", result)

    def test_medical_abbreviation_expansion(self):
        result = self.cleaner.normalise_medical_terms("Patient has SOB and HTN with h/o DM")
        self.assertIn("shortness of breath", result)
        self.assertIn("hypertension", result)
        self.assertIn("history of", result)
        self.assertIn("diabetes mellitus", result)

    def test_encoding_artifact_removal(self):
        result = self.cleaner.clean_raw_asr("some text_x000D_more text")
        self.assertNotIn("_x000D_", result)

    def test_raw_asr_format(self):
        raw = "Um so the patient had chest pain and uh shortness of breath"
        cleaned = self.cleaner.clean(raw, "raw_asr")
        self.assertEqual(cleaned.input_format_detected, "raw_asr")
        self.assertEqual(cleaned.num_turns, 1)


class TestDeIdentifier(unittest.TestCase):

    def setUp(self):
        # Mock spaCy so test runs without model
        with patch("medai.pipeline.clean.deidentify.DeIdentifier._load_spacy_mock", create=True):
            from medai.pipeline.clean.deidentify import DeIdentifier
            self.deid = DeIdentifier()
            self.deid.available = False  # Force regex-only mode

    def test_phone_redaction(self):
        result = self.deid.deidentify("Call me on 07700 900123 please.")
        self.assertIn("[PHONE]", result.clean_text)

    def test_date_redaction(self):
        result = self.deid.deidentify("DOB: 15/03/1985")
        self.assertIn("[DATE]", result.clean_text)

    def test_mrn_redaction(self):
        result = self.deid.deidentify("MRN: A1234567")
        self.assertIn("[MRN]", result.clean_text)

    def test_phi_count(self):
        result = self.deid.deidentify("Call 07700 900123 on 15/03/1985.")
        self.assertGreater(result.phi_count, 0)

    def test_original_preserved(self):
        text = "Call me at 07700 900123"
        result = self.deid.deidentify(text)
        self.assertEqual(result.original_text, text)


class TestSpeakerTurnDataclass(unittest.TestCase):

    def test_instantiation(self):
        from medai.pipeline.asr.diarise import SpeakerTurn
        turn = SpeakerTurn(
            speaker_id="SPEAKER_0",
            role="DOCTOR",
            text="Hello, what brings you in?",
            start=0.0,
            end=3.2,
        )
        self.assertEqual(turn.speaker_id, "SPEAKER_0")
        self.assertEqual(turn.role, "DOCTOR")

    def test_default_role_empty(self):
        from medai.pipeline.asr.diarise import SpeakerTurn
        turn = SpeakerTurn(speaker_id="S0", role="", text="test", start=0.0, end=1.0)
        self.assertEqual(turn.role, "")


class TestTranscriptionResult(unittest.TestCase):

    def test_instantiation(self):
        from medai.pipeline.asr.transcribe import TranscriptionResult
        result = TranscriptionResult(
            text="Patient reports chest pain.",
            segments=[{"start": 0, "end": 2, "text": "Patient reports chest pain."}],
            language="en",
            model_used="whisper_local/base.en",
            duration_seconds=2.3,
        )
        self.assertEqual(result.language, "en")
        self.assertIsNone(result.wer)


class TestCleanedTranscript(unittest.TestCase):

    def test_full_pipeline_wiring(self):
        """Verify data flows through cleaner → turns → cleaned_text."""
        from medai.pipeline.clean.text_cleaner import TranscriptCleaner
        cleaner = TranscriptCleaner()
        cleaned = cleaner.clean(SAMPLE_TRANSCRIPT, "prefixed")

        self.assertIsNotNone(cleaned.cleaned_text)
        self.assertGreater(len(cleaned.cleaned_text), 10)
        self.assertEqual(cleaned.num_turns, len(cleaned.turns))

        for turn in cleaned.turns:
            self.assertIsInstance(turn.text, str)
            self.assertIsInstance(turn.speaker_id, str)


class TestNERDataclasses(unittest.TestCase):

    def test_entity_instantiation(self):
        from medai.pipeline.identify.ner import Entity
        e = Entity(
            text="chest pain",
            label="SYMPTOM",
            start=0,
            end=10,
            umls_code="C0008031",
            confidence=0.95,
        )
        self.assertFalse(e.is_negated)

    def test_ner_result_instantiation(self):
        from medai.pipeline.identify.ner import NERResult, Entity
        entities = [
            Entity("chest pain", "SYMPTOM", 0, 10, confidence=0.9),
            Entity("hypertension", "DIAGNOSIS", 20, 32, confidence=0.8),
        ]
        result = NERResult(
            entities=entities,
            model_used="scispacy/en_core_sci_md",
            extraction_time_ms=42.0,
        )
        self.assertEqual(len(result.entities), 2)


class TestSOAPNoteDataclass(unittest.TestCase):

    def test_instantiation(self):
        from medai.pipeline.intelligence.soap_generator import SOAPNote
        note = SOAPNote(
            subjective="Patient reports chest pain.",
            objective="BP 140/90.",
            assessment="1. Hypertension (ICD-10: I10)",
            plan="Start aspirin 75mg. HCC 85.",
            model_used="gemini",
            generation_time_ms=1200.0,
            icd_codes=["I10"],
            hcc_codes=["85"],
        )
        self.assertEqual(note.icd_codes, ["I10"])
        self.assertEqual(note.hcc_codes, ["85"])


class TestDDxDataclasses(unittest.TestCase):

    def test_differential_instantiation(self):
        from medai.pipeline.intelligence.ddx_engine import Differential
        d = Differential(
            rank=1,
            disease_name="Hypertensive disease",
            umls_code="C0020538",
            icd10_code="I10",
            confidence_score=0.9,
            kb_score=1.5,
            reasoning="Elevated BP consistent with hypertension.",
        )
        self.assertEqual(d.rank, 1)

    def test_ddx_result_instantiation(self):
        from medai.pipeline.intelligence.ddx_engine import DDxResult, Differential
        result = DDxResult(
            differentials=[
                Differential(1, "Hypertension", "C0020538", "I10", 0.9, 1.5, "reason")
            ],
            method_used="kb_plus_llm",
            kb_candidates=[{"disease_name": "Hypertension", "kb_score": 1.5}],
            model_used="gemini",
        )
        self.assertEqual(result.method_used, "kb_plus_llm")


class TestRiskResultDataclass(unittest.TestCase):

    def test_instantiation(self):
        from medai.pipeline.intelligence.risk_stratifier import RiskResult
        r = RiskResult(
            urgency="HIGH",
            red_flags=["Elevated blood pressure"],
            recommended_action="Same-day review.",
            reasoning="Hypertension with symptoms.",
            model_used="gemini",
        )
        self.assertEqual(r.urgency, "HIGH")
        self.assertFalse(r.rag_used)


class TestTreatmentDataclasses(unittest.TestCase):

    def test_medication_instantiation(self):
        from medai.pipeline.intelligence.treatment import Medication
        m = Medication(
            name="Aspirin",
            dose="75mg",
            frequency="once daily",
            duration="long-term",
            notes="Take with food",
        )
        self.assertEqual(m.name, "Aspirin")

    def test_treatment_result_instantiation(self):
        from medai.pipeline.intelligence.treatment import TreatmentResult, Recommendation, Medication
        r = TreatmentResult(
            recommendations=[
                Recommendation("Hypertension", "Start antihypertensive", "immediate", "BP elevated")
            ],
            medications=[
                Medication("Aspirin", "75mg", "once daily", "long-term", "")
            ],
            follow_up="Review in 4 weeks",
            referrals=["Cardiology"],
            patient_education=["Reduce salt intake"],
            model_used="gemini",
        )
        self.assertEqual(len(r.medications), 1)
        self.assertEqual(r.referrals, ["Cardiology"])


class TestPromptBuilding(unittest.TestCase):

    def test_role_classifier_prompt_builds(self):
        from medai.prompts.role_classifier_prompts import build_role_classifier_prompt
        prompt = build_role_classifier_prompt(
            target_turn="I have chest pain.",
            context_before=["Doctor: What brings you in?"],
            context_after=["Doctor: How long has this been going on?"],
        )
        self.assertIn("chest pain", prompt)
        self.assertIn("Label:", prompt)
        self.assertGreater(len(prompt), 200)

    def test_soap_prompts_imported(self):
        from medai.prompts.soap_prompts import (
            SOAP_SUBJECTIVE_PROMPT,
            SOAP_OBJECTIVE_PROMPT,
            SOAP_ASSESSMENT_PROMPT,
            SOAP_PLAN_PROMPT,
            SOAP_COMBINED_PROMPT,
        )
        for prompt in (
            SOAP_SUBJECTIVE_PROMPT,
            SOAP_OBJECTIVE_PROMPT,
            SOAP_ASSESSMENT_PROMPT,
            SOAP_PLAN_PROMPT,
            SOAP_COMBINED_PROMPT,
        ):
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 100)


class TestFHIRBuilder(unittest.TestCase):

    def _make_mock_analysis(self):
        from medai.pipeline.intelligence.soap_generator import SOAPNote
        from medai.pipeline.intelligence.ddx_engine import DDxResult, Differential
        from medai.pipeline.intelligence.risk_stratifier import RiskResult
        from medai.pipeline.intelligence.treatment import TreatmentResult, Medication, Recommendation

        soap = SOAPNote(
            subjective="Patient reports chest pain.",
            objective="BP 140/90.",
            assessment="Hypertension (ICD-10: I10)",
            plan="Aspirin 75mg. HCC 85.",
            model_used="mock",
            generation_time_ms=0.0,
            icd_codes=["I10"],
            hcc_codes=["85"],
        )
        ddx = DDxResult(
            differentials=[
                Differential(1, "Hypertension", "C0020538", "I10", 0.9, 1.5, "BP elevated")
            ],
            method_used="llm_only",
            kb_candidates=[],
            model_used="mock",
        )
        risk = RiskResult(
            urgency="HIGH",
            red_flags=["Elevated BP"],
            recommended_action="Review today.",
            reasoning="Hypertension.",
            model_used="mock",
        )
        treatment = TreatmentResult(
            recommendations=[
                Recommendation("Hypertension", "Antihypertensive", "immediate", "BP high")
            ],
            medications=[Medication("Aspirin", "75mg", "daily", "long-term", "")],
            follow_up="4 weeks",
            referrals=[],
            patient_education=["Low salt diet"],
            model_used="mock",
        )
        return soap, ddx, risk, treatment

    def test_fhir_bundle_structure(self):
        from medai.reports.fhir_builder import FHIRBuilder
        soap, ddx, risk, treatment = self._make_mock_analysis()
        builder = FHIRBuilder()
        bundle = builder.build_bundle(soap, ddx, risk, treatment)

        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(bundle["type"], "document")
        self.assertIn("timestamp", bundle)
        self.assertIn("entry", bundle)
        self.assertGreater(len(bundle["entry"]), 0)

    def test_fhir_composition_present(self):
        from medai.reports.fhir_builder import FHIRBuilder
        soap, ddx, risk, treatment = self._make_mock_analysis()
        builder = FHIRBuilder()
        bundle = builder.build_bundle(soap, ddx, risk, treatment)

        resource_types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        self.assertIn("Composition", resource_types)
        self.assertIn("Condition", resource_types)
        self.assertIn("RiskAssessment", resource_types)


if __name__ == "__main__":
    unittest.main()
