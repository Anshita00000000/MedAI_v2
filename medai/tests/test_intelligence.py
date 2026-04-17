"""
Tests for intelligence layer: DDx, Risk, Treatment, SOAP.
LLM calls are mocked — tests verify logic, parsing, and fallback.
"""

import json
import re
import unittest
from unittest.mock import MagicMock, patch


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_ner_result(symptoms=None, diagnoses=None):
    from medai.pipeline.identify.ner import NERResult, Entity
    entities = []
    for s in (symptoms or []):
        entities.append(Entity(text=s, label="SYMPTOM", start=0, end=len(s), confidence=0.9))
    for d in (diagnoses or []):
        entities.append(Entity(text=d, label="DIAGNOSIS", start=0, end=len(d), confidence=0.8))
    return NERResult(entities=entities, model_used="mock", extraction_time_ms=0.0)


def _make_ddx_result(disease="Hypertension", icd="I10"):
    from medai.pipeline.intelligence.ddx_engine import DDxResult, Differential
    return DDxResult(
        differentials=[
            Differential(
                rank=1,
                disease_name=disease,
                umls_code="C0020538",
                icd10_code=icd,
                confidence_score=0.9,
                kb_score=1.5,
                reasoning="Consistent with symptoms.",
            )
        ],
        method_used="llm_only",
        kb_candidates=[],
        model_used="mock",
    )


def _make_soap():
    from medai.pipeline.intelligence.soap_generator import SOAPNote
    return SOAPNote(
        subjective="Patient reports chest pain.",
        objective="BP 140/90.",
        assessment="1. Hypertension (ICD-10: I10)\n- BP elevated",
        plan="## For Hypertension: HCC 85\n- Aspirin 75mg daily",
        model_used="mock",
        generation_time_ms=100.0,
        icd_codes=["I10"],
        hcc_codes=["85"],
    )


def _make_risk():
    from medai.pipeline.intelligence.risk_stratifier import RiskResult
    return RiskResult(
        urgency="HIGH",
        red_flags=["Elevated blood pressure"],
        recommended_action="Same-day review.",
        reasoning="Hypertension.",
        model_used="mock",
    )


# ------------------------------------------------------------------ #
# SOAP Generator
# ------------------------------------------------------------------ #

class TestSOAPGeneratorCodeExtraction(unittest.TestCase):

    def test_icd10_extraction_regex(self):
        import re
        ICD10_PATTERN = re.compile(r"ICD-10:\s*([A-Z]\d{2}\.?\d*)", re.IGNORECASE)
        text = "1. Hypertension (ICD-10: I10)\n2. Diabetes (ICD-10: E11.9)"
        codes = ICD10_PATTERN.findall(text)
        self.assertEqual(codes, ["I10", "E11.9"])

    def test_hcc_extraction_regex(self):
        import re
        HCC_PATTERN = re.compile(r"HCC\s*(\d+)", re.IGNORECASE)
        text = "## For Hypertension: HCC 85\n## For Diabetes: HCC 19"
        codes = HCC_PATTERN.findall(text)
        self.assertEqual(codes, ["85", "19"])

    def test_section_extraction(self):
        from medai.pipeline.intelligence.soap_generator import SOAPGenerator
        combined = """
# Subjective:
Patient reports chest pain.

# Objective:
BP 140/90.

# Assessment:
1. Hypertension (ICD-10: I10)

# Plan:
Start aspirin 75mg. HCC 85.
"""
        gen = SOAPGenerator.__new__(SOAPGenerator)
        subj = gen._extract_section(combined, "subjective")
        self.assertIn("chest pain", subj.lower())

        obj = gen._extract_section(combined, "objective")
        self.assertIn("140", obj)


# ------------------------------------------------------------------ #
# DDx Engine
# ------------------------------------------------------------------ #

class TestDDxParseLogic(unittest.TestCase):

    def test_parse_valid_json(self):
        from medai.pipeline.intelligence.ddx_engine import DDxEngine
        raw_json = json.dumps({
            "differentials": [
                {
                    "rank": 1,
                    "disease": "Hypertension",
                    "icd10": "I10",
                    "confidence": "high",
                    "supporting_symptoms": ["chest pain", "headache"],
                    "reasoning": "Elevated blood pressure with symptoms.",
                },
                {
                    "rank": 2,
                    "disease": "Angina pectoris",
                    "icd10": "I20.9",
                    "confidence": "medium",
                    "supporting_symptoms": ["chest pain"],
                    "reasoning": "Chest pain on exertion.",
                },
            ]
        })
        kb_candidates = [{"disease_name": "Hypertension", "kb_score": 1.5, "umls_code": "C0020538"}]
        diffs = DDxEngine._parse_differentials(raw_json, kb_candidates)
        self.assertEqual(len(diffs), 2)
        self.assertEqual(diffs[0].disease_name, "Hypertension")
        self.assertEqual(diffs[0].icd10_code, "I10")
        self.assertAlmostEqual(diffs[0].confidence_score, 0.9)
        self.assertAlmostEqual(diffs[0].kb_score, 1.5)

    def test_parse_json_with_preamble(self):
        from medai.pipeline.intelligence.ddx_engine import DDxEngine
        raw = 'Here are the differentials: {"differentials": [{"rank": 1, "disease": "Flu", "icd10": "J11.1", "confidence": "low", "supporting_symptoms": [], "reasoning": "viral"}]}'
        diffs = DDxEngine._parse_differentials(raw, [])
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].disease_name, "Flu")

    def test_parse_invalid_json_returns_empty(self):
        from medai.pipeline.intelligence.ddx_engine import DDxEngine
        diffs = DDxEngine._parse_differentials("not json at all @@@@", [])
        self.assertEqual(diffs, [])

    def test_kb_score_lookup(self):
        from medai.pipeline.intelligence.ddx_engine import DDxEngine, Differential
        raw_json = json.dumps({
            "differentials": [
                {"rank": 1, "disease": "Hypertension", "icd10": "I10",
                 "confidence": "high", "supporting_symptoms": [], "reasoning": ""},
            ]
        })
        kb_candidates = [
            {"disease_name": "Hypertension", "kb_score": 2.75, "umls_code": "C0020538"}
        ]
        diffs = DDxEngine._parse_differentials(raw_json, kb_candidates)
        self.assertEqual(diffs[0].kb_score, 2.75)

    def test_kb_lookup_scoring(self):
        """Verify rank-weight scoring: rank 1 → weight 1.0, rank 2 → 0.5."""
        from medai.pipeline.identify.ner import Entity

        kb = {
            "diseases": [
                {
                    "umls_code": "C0020538",
                    "name": "Hypertension",
                    "occurrence_count": 100,
                    "symptoms": [
                        {"umls_code": "C0018802", "name": "chest pain", "rank": 1},
                        {"umls_code": "C0018803", "name": "headache", "rank": 2},
                    ],
                }
            ]
        }

        from medai.pipeline.intelligence.ddx_engine import DDxEngine
        engine = DDxEngine.__new__(DDxEngine)
        engine._kb = kb

        entities = [
            Entity("chest pain", "SYMPTOM", 0, 10),
            Entity("headache", "SYMPTOM", 11, 19),
        ]
        results = engine._kb_lookup(entities)
        self.assertEqual(len(results), 1)
        expected_score = 1.0 + 0.5  # rank 1 + rank 2
        self.assertAlmostEqual(results[0]["kb_score"], expected_score, places=2)


# ------------------------------------------------------------------ #
# Risk Stratifier
# ------------------------------------------------------------------ #

class TestRiskStratifierParsing(unittest.TestCase):

    def test_parse_valid_json(self):
        from medai.pipeline.intelligence.risk_stratifier import RiskStratifier
        strat = RiskStratifier.__new__(RiskStratifier)

        raw = json.dumps({
            "urgency": "HIGH",
            "red_flags": ["Chest pain with radiation"],
            "recommended_action": "Urgent ECG.",
            "reasoning": "Possible ACS.",
        })
        result = strat._parse_result(raw)
        self.assertEqual(result.urgency, "HIGH")
        self.assertIn("Chest pain with radiation", result.red_flags)

    def test_invalid_urgency_defaults_to_medium(self):
        from medai.pipeline.intelligence.risk_stratifier import RiskStratifier
        strat = RiskStratifier.__new__(RiskStratifier)

        raw = json.dumps({
            "urgency": "UNKNOWN_LEVEL",
            "red_flags": [],
            "recommended_action": "",
            "reasoning": "",
        })
        result = strat._parse_result(raw)
        self.assertEqual(result.urgency, "MEDIUM")

    def test_fallback_on_bad_json(self):
        from medai.pipeline.intelligence.risk_stratifier import RiskStratifier
        strat = RiskStratifier.__new__(RiskStratifier)
        result = strat._parse_result("gibberish response with no JSON")
        self.assertEqual(result.urgency, "MEDIUM")
        self.assertIsInstance(result.red_flags, list)


# ------------------------------------------------------------------ #
# Treatment Recommender
# ------------------------------------------------------------------ #

class TestTreatmentParsing(unittest.TestCase):

    def test_parse_valid_json(self):
        from medai.pipeline.intelligence.treatment import TreatmentRecommender
        rec = TreatmentRecommender.__new__(TreatmentRecommender)

        raw = json.dumps({
            "recommendations": [
                {
                    "diagnosis": "Hypertension",
                    "intervention": "Antihypertensive therapy",
                    "priority": "immediate",
                    "rationale": "BP consistently elevated",
                }
            ],
            "medications": [
                {
                    "name": "Aspirin",
                    "dose": "75mg",
                    "frequency": "once daily",
                    "duration": "long-term",
                    "notes": "Take with food",
                }
            ],
            "follow_up": "Review in 4 weeks",
            "referrals": ["Cardiology"],
            "patient_education": ["Low salt diet", "Exercise regularly"],
        })
        result = rec._parse_result(raw)
        self.assertEqual(len(result.recommendations), 1)
        self.assertEqual(result.recommendations[0].diagnosis, "Hypertension")
        self.assertEqual(result.medications[0].name, "Aspirin")
        self.assertEqual(result.follow_up, "Review in 4 weeks")
        self.assertEqual(result.referrals, ["Cardiology"])

    def test_fallback_on_bad_json(self):
        from medai.pipeline.intelligence.treatment import TreatmentRecommender
        rec = TreatmentRecommender.__new__(TreatmentRecommender)
        result = rec._parse_result("not valid json")
        self.assertIsInstance(result.recommendations, list)
        self.assertIsInstance(result.medications, list)

    def test_recommend_with_rag_raises(self):
        from medai.pipeline.intelligence.treatment import TreatmentRecommender
        rec = TreatmentRecommender.__new__(TreatmentRecommender)
        with self.assertRaises(NotImplementedError):
            rec.recommend_with_rag(_make_soap(), _make_ddx_result())


# ------------------------------------------------------------------ #
# Negation Detector
# ------------------------------------------------------------------ #

class TestNegationDetector(unittest.TestCase):

    def setUp(self):
        from medai.pipeline.identify.negation import NegationDetector
        self.detector = NegationDetector()
        self.detector.available = True

    def test_negated_chest_pain(self):
        from medai.pipeline.identify.ner import Entity
        entity = Entity("chest pain", "SYMPTOM", 3, 13)
        text = "no chest pain present"
        result = self.detector.detect(text, [entity])
        self.assertTrue(result[0].is_negated)

    def test_denies_symptom(self):
        from medai.pipeline.identify.ner import Entity
        entity = Entity("shortness of breath", "SYMPTOM", 8, 27)
        text = "patient denies shortness of breath"
        result = self.detector.detect(text, [entity])
        self.assertTrue(result[0].is_negated)

    def test_not_negated(self):
        from medai.pipeline.identify.ner import Entity
        entity = Entity("chest pain", "SYMPTOM", 8, 18)
        text = "patient has chest pain since yesterday"
        result = self.detector.detect(text, [entity])
        self.assertFalse(result[0].is_negated)

    def test_no_history_of(self):
        from medai.pipeline.identify.ner import Entity
        entity = Entity("diabetes", "DIAGNOSIS", 16, 24)
        text = "no history of diabetes in the family"
        result = self.detector.detect(text, [entity])
        self.assertTrue(result[0].is_negated)


# ------------------------------------------------------------------ #
# Columbia Parser helpers (no network)
# ------------------------------------------------------------------ #

class TestColumbiaParserHelpers(unittest.TestCase):

    def test_parse_simple_umls(self):
        from medai.knowledge.columbia_parser import _parse_umls_entry
        result = _parse_umls_entry("UMLS:C0020538_hypertensive disease")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["umls_code"], "C0020538")
        self.assertEqual(result[0]["name"], "hypertensive disease")

    def test_parse_compound_umls(self):
        from medai.knowledge.columbia_parser import _parse_umls_entry
        result = _parse_umls_entry(
            "UMLS:C0038990_sweat^UMLS:C0700590_sweating increased"
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["umls_code"], "C0038990")
        self.assertEqual(result[1]["umls_code"], "C0700590")

    def test_parse_empty_entry(self):
        from medai.knowledge.columbia_parser import _parse_umls_entry
        result = _parse_umls_entry("")
        self.assertEqual(result, [])


# ------------------------------------------------------------------ #
# RAG Pipeline stub
# ------------------------------------------------------------------ #

class TestRAGPipeline(unittest.TestCase):

    def test_is_not_ready_initially(self):
        from medai.knowledge.rag_pipeline import RAGPipeline
        rag = RAGPipeline.__new__(RAGPipeline)
        rag._ready = False
        rag._collection = None
        self.assertFalse(rag.is_ready())

    def test_query_returns_empty_when_not_ready(self):
        from medai.knowledge.rag_pipeline import RAGPipeline
        rag = RAGPipeline.__new__(RAGPipeline)
        rag._ready = False
        rag._collection = None
        result = rag.query("chest pain red flags")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
