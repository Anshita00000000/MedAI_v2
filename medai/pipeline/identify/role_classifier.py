"""Speaker role classification using MedPhi-Instruct with Gemini fallback."""

import os
import logging
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from medai.pipeline.asr.diarise import SpeakerTurn
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class SpeakerTurn:  # type: ignore[no-redef]
        speaker_id: str = ""
        role: str = ""
        text: str = ""
        start: float = 0.0
        end: float = 0.0

from medai.prompts.role_classifier_prompts import build_role_classifier_prompt

VALID_ROLES = {"DOCTOR", "PATIENT", "NURSE", "FAMILY"}


class RoleClassifier:
    """Classify speaker roles using MedPhi with Gemini fallback."""

    def __init__(self):
        self.model_preference = os.getenv("ROLE_CLASSIFIER_MODEL", "medphi").lower()
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.available = False
        self._hf_model = None
        self._hf_tokenizer = None
        self._active_model = None

        if self.model_preference == "medphi" and self.hf_token:
            self._load_medphi()
        if not self.available and self.gemini_api_key:
            self._active_model = "gemini"
            self.available = True
            logger.info("RoleClassifier using Gemini fallback.")
        if not self.available:
            logger.warning(
                "RoleClassifier: no model available. "
                "Set HUGGINGFACE_TOKEN or GEMINI_API_KEY."
            )

    def _load_medphi(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_id = "microsoft/Phi-3.5-mini-instruct"
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=self.hf_token
            )
            import torch
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=self.hf_token,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self._active_model = "medphi"
            self.available = True
            logger.info("Loaded MedPhi model for role classification.")
        except Exception as exc:
            logger.warning("Could not load MedPhi: %s", exc)

    def classify_turn(
        self,
        turn: SpeakerTurn,
        context: List[SpeakerTurn],
    ) -> str:
        if not self.available:
            raise RuntimeError(
                "No LLM available. Set GEMINI_API_KEY or load a local model."
            )

        idx = None
        for i, t in enumerate(context):
            if t is turn:
                idx = i
                break

        if idx is None:
            ctx_before, ctx_after = [], []
        else:
            ctx_before = [t.text for t in context[max(0, idx - 2): idx]]
            ctx_after = [t.text for t in context[idx + 1: idx + 3]]

        prompt = build_role_classifier_prompt(turn.text, ctx_before, ctx_after)

        try:
            if self._active_model == "medphi":
                label = self._call_medphi(prompt)
            else:
                label = self._call_gemini(prompt)
        except Exception as exc:
            logger.warning("Primary model failed (%s), falling back to Gemini.", exc)
            label = self._call_gemini(prompt)

        label = label.strip().upper()
        if label not in VALID_ROLES:
            label = "PATIENT"  # safe default
        return label

    def _call_medphi(self, prompt: str) -> str:
        inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(
            self._hf_model.device
        )
        outputs = self._hf_model.generate(
            **inputs, max_new_tokens=10, do_sample=False
        )
        decoded = self._hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the last token(s) after the prompt
        answer = decoded[len(prompt):].strip().split()[0] if decoded[len(prompt):].strip() else decoded.strip().split()[-1]
        return answer

    def _call_gemini(self, prompt: str) -> str:
        from google import genai
        client = genai.Client(api_key=self.gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()

    def classify_transcript(
        self, turns: List[SpeakerTurn]
    ) -> List[SpeakerTurn]:
        result = []
        for i, turn in enumerate(turns):
            role = self.classify_turn(turn, turns)
            result.append(
                SpeakerTurn(
                    speaker_id=turn.speaker_id,
                    role=role,
                    text=turn.text,
                    start=turn.start,
                    end=turn.end,
                )
            )
        return result

    def evaluate(
        self,
        turns: List[SpeakerTurn],
        ground_truth: List[str],
    ) -> dict:
        from collections import defaultdict

        predictions = [self.classify_turn(t, turns) for t in turns]
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(ground_truth) if ground_truth else 0.0

        # Per-class counts for F1
        tp: dict = defaultdict(int)
        fp: dict = defaultdict(int)
        fn: dict = defaultdict(int)
        for pred, gold in zip(predictions, ground_truth):
            if pred == gold:
                tp[gold] += 1
            else:
                fp[pred] += 1
                fn[gold] += 1

        f1_scores = {}
        for cls in VALID_ROLES:
            prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
            rec = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
            f1_scores[cls] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        macro_f1 = sum(f1_scores.values()) / len(f1_scores)

        # Confusion matrix
        confusion: dict = {r: {c: 0 for c in VALID_ROLES} for r in VALID_ROLES}
        for pred, gold in zip(predictions, ground_truth):
            if gold in confusion and pred in confusion:
                confusion[gold][pred] += 1

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class_f1": f1_scores,
            "confusion_matrix": confusion,
            "model_used": self._active_model,
        }
