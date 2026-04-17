"""SOAP note generation using MedGemma with Gemini fallback."""

import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

from medai.prompts.soap_prompts import (
    SOAP_SUBJECTIVE_PROMPT,
    SOAP_OBJECTIVE_PROMPT,
    SOAP_ASSESSMENT_PROMPT,
    SOAP_PLAN_PROMPT,
    SOAP_COMBINED_PROMPT,
)

try:
    from medai.pipeline.clean.text_cleaner import CleanedTranscript
except ImportError:
    from dataclasses import dataclass as _dc

    @_dc
    class CleanedTranscript:  # type: ignore[no-redef]
        turns: list = field(default_factory=list)
        raw_text: str = ""
        cleaned_text: str = ""
        input_format_detected: str = ""
        num_turns: int = 0


ICD10_PATTERN = re.compile(r"ICD-10:\s*([A-Z]\d{2}\.?\d*)", re.IGNORECASE)
HCC_PATTERN = re.compile(r"HCC\s*(\d+)", re.IGNORECASE)


@dataclass
class SOAPNote:
    subjective: str
    objective: str
    assessment: str
    plan: str
    model_used: str
    generation_time_ms: float
    icd_codes: List[str] = field(default_factory=list)
    hcc_codes: List[str] = field(default_factory=list)


class SOAPGenerator:
    """Generate SOAP notes using MedGemma (primary) or Gemini (fallback)."""

    def __init__(self):
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.primary_llm = os.getenv("PRIMARY_LLM", "medgemma").lower()
        self.available = False
        self._hf_model = None
        self._hf_tokenizer = None
        self._active_model = None

        if self.primary_llm == "medgemma" and self.hf_token:
            self._load_medgemma()
        if not self.available and self.gemini_api_key:
            self._active_model = "gemini"
            self.available = True
            logger.info("SOAPGenerator using Gemini fallback.")
        if not self.available:
            logger.warning(
                "SOAPGenerator: no model available. "
                "Set HUGGINGFACE_TOKEN or GEMINI_API_KEY."
            )

    def _load_medgemma(self):
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
            logger.info("MedGemma loaded.")
        except Exception as exc:
            logger.warning("Could not load MedGemma: %s", exc)

    def generate(
        self,
        transcript: CleanedTranscript,
        mode: str = "sectional",
        reference_soap: Optional["SOAPNote"] = None,
    ) -> SOAPNote:
        if not self.available:
            raise RuntimeError(
                "No LLM available. Set GEMINI_API_KEY or load a local model."
            )

        t0 = time.time()
        transcript_text = transcript.cleaned_text

        if mode == "combined":
            combined = self._llm_call(SOAP_COMBINED_PROMPT, transcript_text)
            subjective = self._extract_section(combined, "subjective")
            objective = self._extract_section(combined, "objective")
            assessment = self._extract_section(combined, "assessment")
            plan = self._extract_section(combined, "plan")
        else:
            subjective = self.generate_section(transcript, "subjective")
            objective = self.generate_section(transcript, "objective")
            assessment = self.generate_section(transcript, "assessment")
            plan = self.generate_section(transcript, "plan")

        icd_codes = ICD10_PATTERN.findall(assessment)
        hcc_codes = HCC_PATTERN.findall(plan)

        note = SOAPNote(
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan,
            model_used=self._active_model,
            generation_time_ms=(time.time() - t0) * 1000,
            icd_codes=icd_codes,
            hcc_codes=hcc_codes,
        )

        if reference_soap:
            scores = self._evaluate(note, reference_soap)
            logger.info("SOAP evaluation scores: %s", scores)

        return note

    def generate_section(
        self,
        transcript: CleanedTranscript,
        section: str,
    ) -> str:
        prompts = {
            "subjective": SOAP_SUBJECTIVE_PROMPT,
            "objective": SOAP_OBJECTIVE_PROMPT,
            "assessment": SOAP_ASSESSMENT_PROMPT,
            "plan": SOAP_PLAN_PROMPT,
        }
        if section not in prompts:
            raise ValueError(f"Unknown section: {section}")

        return self._llm_call(prompts[section], transcript.cleaned_text)

    def _llm_call(self, system_prompt: str, user_content: str) -> str:
        try:
            if self._active_model == "medgemma":
                return self._call_medgemma(system_prompt, user_content)
            else:
                return self._call_gemini(system_prompt, user_content)
        except Exception as exc:
            logger.warning("Primary model failed (%s), falling back to Gemini.", exc)
            return self._call_gemini(system_prompt, user_content)

    def _call_medgemma(self, system_prompt: str, user_content: str) -> str:
        import torch
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nTranscript:\n{user_content}"},
        ]
        inputs = self._hf_tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self._hf_model.device)
        outputs = self._hf_model.generate(
            inputs, max_new_tokens=1024, do_sample=False
        )
        decoded = self._hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt prefix from the output
        return decoded.split("<start_of_turn>model")[-1].strip()

    def _call_gemini(self, system_prompt: str, user_content: str) -> str:
        from google import genai
        client = genai.Client(api_key=self.gemini_api_key)
        prompt = f"{system_prompt}\n\nTranscript:\n{user_content}"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()

    @staticmethod
    def _extract_section(text: str, section: str) -> str:
        pattern = re.compile(
            rf"#\s*{section}:?(.*?)(?=#\s*(?:subjective|objective|assessment|plan)|$)",
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(text)
        if m:
            return f"# {section.capitalize()}:\n{m.group(1).strip()}"
        return text

    def _evaluate(self, generated: "SOAPNote", reference: "SOAPNote") -> dict:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
            results = {}
            for section in ("subjective", "objective", "assessment", "plan"):
                gen_text = getattr(generated, section)
                ref_text = getattr(reference, section)
                scores = scorer.score(ref_text, gen_text)
                results[section] = {
                    "rouge1": scores["rouge1"].fmeasure,
                    "rougeL": scores["rougeL"].fmeasure,
                }
            return results
        except Exception as exc:
            logger.warning("ROUGE evaluation failed: %s", exc)
            return {}
