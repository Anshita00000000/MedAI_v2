"""Differential diagnosis engine: Columbia KB lookup + MedGemma reranking."""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from medai.pipeline.identify.ner import NERResult, Entity
except ImportError:
    pass

KB_PATH = Path(__file__).parent.parent.parent / "data" / "knowledge_base" / "columbia_kb.json"

DDX_PROMPT_TEMPLATE = """
You are a clinical decision support system.

Patient symptoms (confirmed): {symptoms}

Top disease candidates from our knowledge base: {kb_candidates}

Based on these symptoms and candidates:
1. Rerank and provide the top 5 differential diagnoses
2. For each diagnosis provide:
   - ICD-10 code
   - Confidence level (high/medium/low)
   - Key supporting symptoms
   - Brief clinical reasoning (1-2 sentences)
3. Add any important differentials not in the candidate list

Return as structured JSON only. No preamble.

Format:
{{
  "differentials": [
    {{
      "rank": 1,
      "disease": "...",
      "icd10": "...",
      "confidence": "high",
      "supporting_symptoms": [],
      "reasoning": "..."
    }}
  ]
}}
"""


@dataclass
class Differential:
    rank: int
    disease_name: str
    umls_code: str
    icd10_code: str
    confidence_score: float
    kb_score: float
    reasoning: str
    supporting_symptoms: List[str] = field(default_factory=list)


@dataclass
class DDxResult:
    differentials: List[Differential]
    method_used: str  # "kb_plus_llm" or "llm_only"
    kb_candidates: List[dict]
    model_used: str


class DDxEngine:
    """Differential diagnosis via Columbia KB + LLM reranking."""

    def __init__(self):
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.available = False
        self._hf_model = None
        self._hf_tokenizer = None
        self._active_model = None
        self._kb: Optional[dict] = None

        self._load_kb()
        self._load_medgemma()
        if not self.available and self.gemini_api_key:
            self._active_model = "gemini"
            self.available = True
        if not self.available:
            logger.warning("DDxEngine: no LLM available.")

    def _load_kb(self):
        try:
            with open(KB_PATH) as f:
                self._kb = json.load(f)
            logger.info("Columbia KB loaded (%d diseases).", len(self._kb["diseases"]))
        except Exception as exc:
            logger.warning("Columbia KB not found at %s: %s", KB_PATH, exc)

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
            logger.warning("Could not load MedGemma for DDx: %s", exc)

    def generate_ddx(
        self,
        entities: "NERResult",
        soap_assessment: Optional[str] = None,
    ) -> DDxResult:
        if not self.available:
            raise RuntimeError(
                "No LLM available. Set GEMINI_API_KEY or load a local model."
            )

        confirmed = [
            e for e in entities.entities
            if not e.is_negated and e.label in ("SYMPTOM", "DIAGNOSIS")
        ]
        symptom_names = list({e.text for e in confirmed})

        kb_candidates = self._kb_lookup(confirmed) if self._kb else []
        method = "kb_plus_llm" if len(kb_candidates) >= 3 else "llm_only"

        prompt = DDX_PROMPT_TEMPLATE.format(
            symptoms=", ".join(symptom_names) or "Not specified",
            kb_candidates=json.dumps(kb_candidates[:10], indent=2),
        )

        raw_json = self._llm_call(prompt)
        differentials = self._parse_differentials(raw_json, kb_candidates)

        return DDxResult(
            differentials=differentials,
            method_used=method,
            kb_candidates=kb_candidates,
            model_used=self._active_model,
        )

    def _kb_lookup(self, entities: List["Entity"]) -> List[dict]:
        if not self._kb:
            return []
        disease_scores: dict = {}
        for disease in self._kb["diseases"]:
            score = 0.0
            for sym in disease["symptoms"]:
                for ent in entities:
                    if (
                        sym["name"].lower() in ent.text.lower()
                        or ent.text.lower() in sym["name"].lower()
                    ):
                        score += 1.0 / sym["rank"]
            if score > 0:
                disease_scores[disease["umls_code"]] = {
                    "disease_name": disease["name"],
                    "umls_code": disease["umls_code"],
                    "kb_score": round(score, 4),
                    "occurrence_count": disease["occurrence_count"],
                }
        sorted_diseases = sorted(
            disease_scores.values(), key=lambda d: d["kb_score"], reverse=True
        )
        return sorted_diseases[:10]

    def _llm_call(self, prompt: str) -> str:
        try:
            if self._active_model == "medgemma":
                return self._call_medgemma(prompt)
            else:
                return self._call_gemini(prompt)
        except Exception as exc:
            logger.warning("Primary DDx model failed (%s), falling back to Gemini.", exc)
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

    @staticmethod
    def _parse_differentials(
        raw_json: str, kb_candidates: List[dict]
    ) -> List[Differential]:
        kb_score_map = {d["disease_name"].lower(): d["kb_score"] for d in kb_candidates}
        kb_umls_map = {d["disease_name"].lower(): d.get("umls_code", "") for d in kb_candidates}

        try:
            json_match = re.search(r"\{.*\}", raw_json, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw_json)

            differentials = []
            conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
            for item in data.get("differentials", []):
                name = item.get("disease", "Unknown")
                differentials.append(
                    Differential(
                        rank=item.get("rank", len(differentials) + 1),
                        disease_name=name,
                        umls_code=kb_umls_map.get(name.lower(), ""),
                        icd10_code=item.get("icd10", ""),
                        confidence_score=conf_map.get(item.get("confidence", "low"), 0.3),
                        kb_score=kb_score_map.get(name.lower(), 0.0),
                        reasoning=item.get("reasoning", ""),
                        supporting_symptoms=item.get("supporting_symptoms", []),
                    )
                )
            return differentials
        except Exception as exc:
            logger.warning("Failed to parse DDx JSON: %s", exc)
            return []
