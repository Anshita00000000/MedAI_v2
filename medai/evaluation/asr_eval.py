"""ASR evaluation: WER, CER, medical term accuracy."""

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ASREvaluator:

    def evaluate_wer(self, hypothesis: str, reference: str) -> float:
        try:
            from jiwer import wer
            return float(wer(reference, hypothesis))
        except ImportError:
            logger.warning("jiwer not installed; WER unavailable.")
            return -1.0

    def evaluate_cer(self, hypothesis: str, reference: str) -> float:
        try:
            from jiwer import cer
            return float(cer(reference, hypothesis))
        except ImportError:
            logger.warning("jiwer not installed; CER unavailable.")
            return -1.0

    def evaluate_med_term_accuracy(
        self,
        hypothesis: str,
        reference: str,
        medical_terms: List[str],
    ) -> float:
        """
        Check what % of medical terms in reference appear correctly in hypothesis.
        """
        hyp_lower = hypothesis.lower()
        ref_lower = reference.lower()
        terms_in_ref = [t for t in medical_terms if t.lower() in ref_lower]
        if not terms_in_ref:
            return 1.0  # no medical terms to check
        correct = sum(1 for t in terms_in_ref if t.lower() in hyp_lower)
        return correct / len(terms_in_ref)

    def compare_models(
        self,
        audio_path: str,
        reference: str,
        models: List[str],
    ) -> pd.DataFrame:
        """
        Run multiple ASR models on the same audio file and return a comparison table.
        """
        from medai.pipeline.asr.transcribe import ASRTranscriber
        import os

        rows = []
        for model_name in models:
            os.environ["ASR_MODEL"] = model_name
            try:
                transcriber = ASRTranscriber()
                result = transcriber.transcribe(audio_path, reference_text=reference)
                wer_score = result.wer if result.wer is not None else self.evaluate_wer(result.text, reference)
                cer_score = self.evaluate_cer(result.text, reference)
                rows.append({
                    "model": model_name,
                    "wer": round(wer_score, 4),
                    "cer": round(cer_score, 4),
                    "duration_s": round(result.duration_seconds, 2),
                    "transcript_length": len(result.text),
                })
            except Exception as exc:
                logger.warning("Model %s failed: %s", model_name, exc)
                rows.append({"model": model_name, "wer": -1, "cer": -1, "error": str(exc)})

        return pd.DataFrame(rows)
