"""Clinical urgency classification and red flag identification."""

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
    from medai.pipeline.identify.ner import NERResult
    from medai.pipeline.intelligence.ddx_engine import DDxResult
except ImportError:
    pass

RISK_STRATIFICATION_PROMPT = """
{rag_context}
You are a clinical risk assessment system.

Patient assessment:
- Confirmed symptoms: {symptoms}
- Top differential diagnosis: {top_ddx}
- All differentials: {all_ddx}
- SOAP Assessment section: {assessment}

Tasks:
1. Classify urgency as exactly one of: CRITICAL, HIGH, MEDIUM, LOW
   - CRITICAL: requires immediate emergency intervention (call 999/911)
   - HIGH: requires urgent same-day review
   - MEDIUM: requires follow-up within days
   - LOW: routine follow-up appropriate

2. Identify any clinical red flags present in this case.
   {rag_instruction}

3. State the recommended immediate action for the treating physician.

4. Provide brief clinical reasoning.

Return ONLY valid JSON in this exact structure:
{{
  "urgency": "CRITICAL|HIGH|MEDIUM|LOW",
  "red_flags": ["flag 1", "flag 2"],
  "recommended_action": "...",
  "reasoning": "..."
}}
"""


@dataclass
class RiskResult:
    urgency: str
    red_flags: List[str]
    recommended_action: str
    reasoning: str
    model_used: str
    rag_used: bool = False


class RiskStratifier:
    """Classify clinical urgency and identify red flags."""

    URGENCY_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

    def __init__(self, rag_pipeline=None):
        self.rag_pipeline = rag_pipeline
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
            logger.warning("RiskStratifier: no LLM available.")

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
            logger.warning("Could not load MedGemma for risk stratification: %s", exc)

    def stratify(
        self,
        entities: "NERResult",
        ddx_result: "DDxResult",
        soap_assessment: str = "",
    ) -> RiskResult:
        if not self.available:
            raise RuntimeError(
                "No LLM available. Set GEMINI_API_KEY or load a local model."
            )

        confirmed_symptoms = [
            e.text for e in entities.entities
            if not e.is_negated and e.label in ("SYMPTOM", "DIAGNOSIS")
        ]
        top_ddx = (
            ddx_result.differentials[0].disease_name
            if ddx_result.differentials
            else "Unknown"
        )
        all_ddx = [d.disease_name for d in ddx_result.differentials[:5]]

        # RAG context
        rag_used = False
        rag_context = ""
        rag_instruction = "Use standard clinical guidelines to identify red flags."

        if self.rag_pipeline and self.rag_pipeline.is_ready():
            try:
                chunks = self.rag_pipeline.query(
                    f"clinical red flags urgent referral criteria {top_ddx}"
                )
                if chunks:
                    chunk_texts = "\n\n".join(c.content for c in chunks)
                    rag_context = f"RELEVANT CLINICAL GUIDELINES:\n{chunk_texts}\n\n"
                    rag_instruction = (
                        "Base red flag identification on the clinical guidelines provided above."
                    )
                    rag_used = True
            except Exception as exc:
                logger.warning("RAG query failed: %s", exc)

        prompt = RISK_STRATIFICATION_PROMPT.format(
            rag_context=rag_context,
            symptoms=", ".join(confirmed_symptoms) or "None documented",
            top_ddx=top_ddx,
            all_ddx=", ".join(all_ddx) or "None",
            assessment=soap_assessment or "Not provided",
            rag_instruction=rag_instruction,
        )

        raw = self._llm_call(prompt)
        result = self._parse_result(raw)
        result.model_used = self._active_model
        result.rag_used = rag_used
        return result

    def _llm_call(self, prompt: str) -> str:
        try:
            if self._active_model == "medgemma":
                return self._call_medgemma(prompt)
            else:
                return self._call_gemini(prompt)
        except Exception as exc:
            logger.warning("Risk stratifier primary model failed (%s), using Gemini.", exc)
            return self._call_gemini(prompt)

    def _call_medgemma(self, prompt: str) -> str:
        import torch
        inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(
            self._hf_model.device
        )
        outputs = self._hf_model.generate(
            **inputs, max_new_tokens=512, do_sample=False
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

    def _parse_result(self, raw: str) -> RiskResult:
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)

            urgency = data.get("urgency", "MEDIUM").upper()
            if urgency not in self.URGENCY_LEVELS:
                urgency = "MEDIUM"

            return RiskResult(
                urgency=urgency,
                red_flags=data.get("red_flags", []),
                recommended_action=data.get("recommended_action", ""),
                reasoning=data.get("reasoning", ""),
                model_used="",
            )
        except Exception as exc:
            logger.warning("Failed to parse risk result JSON: %s", exc)
            return RiskResult(
                urgency="MEDIUM",
                red_flags=[],
                recommended_action="Review with clinician.",
                reasoning="Unable to parse LLM response.",
                model_used="",
            )
