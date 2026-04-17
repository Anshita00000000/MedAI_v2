"""Evidence-based treatment recommendation engine."""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from medai.pipeline.intelligence.soap_generator import SOAPNote
    from medai.pipeline.intelligence.ddx_engine import DDxResult
    from medai.pipeline.intelligence.risk_stratifier import RiskResult
except ImportError:
    pass

TREATMENT_PROMPT_TEMPLATE = """
You are a clinical treatment recommendation system.

Primary diagnosis: {primary_ddx}
Differential diagnoses: {ddx_list}
Risk level: {urgency}
SOAP Plan section: {plan}

Provide evidence-based treatment recommendations:
1. Immediate interventions (if any)
2. Medications with dose and frequency
3. Non-pharmacological interventions
4. Follow-up timeline
5. Specialist referrals if indicated
6. Patient education points

Base recommendations on standard clinical guidelines.
Flag any recommendations that require specialist confirmation.

Return as structured JSON only. No preamble.

Format:
{{
  "recommendations": [
    {{
      "diagnosis": "...",
      "intervention": "...",
      "priority": "immediate|short-term|long-term",
      "rationale": "..."
    }}
  ],
  "medications": [
    {{
      "name": "...",
      "dose": "...",
      "frequency": "...",
      "duration": "...",
      "notes": "..."
    }}
  ],
  "follow_up": "...",
  "referrals": ["..."],
  "patient_education": ["..."]
}}
"""


@dataclass
class Recommendation:
    diagnosis: str
    intervention: str
    priority: str
    rationale: str


@dataclass
class Medication:
    name: str
    dose: str
    frequency: str
    duration: str
    notes: str


@dataclass
class TreatmentResult:
    recommendations: List[Recommendation]
    medications: List[Medication]
    follow_up: str
    referrals: List[str]
    patient_education: List[str]
    model_used: str


class TreatmentRecommender:
    """Generate treatment recommendations using MedGemma or Gemini."""

    def __init__(self):
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.available = False
        self._hf_model = None
        self._hf_tokenizer = None
        self._active_model = None

        self._load_medgemma()
        if not self.available and self.gemini_api_key:
            self._active_model = "gemini"
            self.available = True
        if not self.available:
            logger.warning("TreatmentRecommender: no LLM available.")

    def _load_medgemma(self):
        if not self.hf_token:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_id = "google/medgemma-4b-it"
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=self.hf_token
            )
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=self.hf_token,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self._active_model = "medgemma"
            self.available = True
        except Exception as exc:
            logger.warning("Could not load MedGemma for treatment: %s", exc)

    def recommend(
        self,
        soap_note: "SOAPNote",
        ddx_result: "DDxResult",
        risk_result: "RiskResult",
    ) -> TreatmentResult:
        if not self.available:
            raise RuntimeError(
                "No LLM available. Set GEMINI_API_KEY or load a local model."
            )

        primary_ddx = (
            ddx_result.differentials[0].disease_name
            if ddx_result.differentials
            else "Unknown"
        )
        ddx_list = [d.disease_name for d in ddx_result.differentials[:5]]

        prompt = TREATMENT_PROMPT_TEMPLATE.format(
            primary_ddx=primary_ddx,
            ddx_list=", ".join(ddx_list),
            urgency=risk_result.urgency,
            plan=soap_note.plan,
        )

        raw = self._llm_call(prompt)
        result = self._parse_result(raw)
        result.model_used = self._active_model
        return result

    def recommend_with_rag(
        self,
        soap_note: "SOAPNote",
        ddx_result: "DDxResult",
    ) -> TreatmentResult:
        raise NotImplementedError(
            "RAG pipeline not yet loaded. "
            "Load NICE/WHO guidelines via knowledge/rag_pipeline.py first."
        )

    def _llm_call(self, prompt: str) -> str:
        try:
            if self._active_model == "medgemma":
                return self._call_medgemma(prompt)
            else:
                return self._call_gemini(prompt)
        except Exception as exc:
            logger.warning("Treatment primary model failed (%s), using Gemini.", exc)
            return self._call_gemini(prompt)

    def _call_medgemma(self, prompt: str) -> str:
        import torch
        inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(
            self._hf_model.device
        )
        outputs = self._hf_model.generate(
            **inputs, max_new_tokens=1024, do_sample=False
        )
        return self._hf_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _call_gemini(self, prompt: str) -> str:
        from google import genai
        client = genai.Client(api_key=self.gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()

    def _parse_result(self, raw: str) -> TreatmentResult:
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)

            recommendations = [
                Recommendation(**r) for r in data.get("recommendations", [])
            ]
            medications = [
                Medication(**m) for m in data.get("medications", [])
            ]
            return TreatmentResult(
                recommendations=recommendations,
                medications=medications,
                follow_up=data.get("follow_up", ""),
                referrals=data.get("referrals", []),
                patient_education=data.get("patient_education", []),
                model_used="",
            )
        except Exception as exc:
            logger.warning("Failed to parse treatment JSON: %s", exc)
            return TreatmentResult(
                recommendations=[],
                medications=[],
                follow_up="Follow up with clinician.",
                referrals=[],
                patient_education=[],
                model_used="",
            )
