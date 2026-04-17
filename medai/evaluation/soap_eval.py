"""SOAP note evaluation: ROUGE + BERTScore."""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class SOAPEvaluator:

    def evaluate(self, generated, reference) -> dict:
        """
        Compute ROUGE-1, ROUGE-L, and BERTScore for each SOAP section.
        generated and reference are SOAPNote objects.
        """
        results = {}
        for section in ("subjective", "objective", "assessment", "plan"):
            gen_text = getattr(generated, section, "") or ""
            ref_text = getattr(reference, section, "") or ""
            results[section] = self.evaluate_section(gen_text, ref_text, section)
        return results

    def evaluate_section(
        self, generated: str, reference: str, section: str = ""
    ) -> dict:
        metrics = {}

        # ROUGE
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            scores = scorer.score(reference, generated)
            metrics["rouge1_f"] = round(scores["rouge1"].fmeasure, 4)
            metrics["rouge1_p"] = round(scores["rouge1"].precision, 4)
            metrics["rouge1_r"] = round(scores["rouge1"].recall, 4)
            metrics["rouge2_f"] = round(scores["rouge2"].fmeasure, 4)
            metrics["rougeL_f"] = round(scores["rougeL"].fmeasure, 4)
        except ImportError:
            logger.warning("rouge_score not installed.")

        # BERTScore
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(
                [generated], [reference], lang="en", verbose=False
            )
            metrics["bertscore_f1"] = round(float(F1.mean()), 4)
            metrics["bertscore_precision"] = round(float(P.mean()), 4)
            metrics["bertscore_recall"] = round(float(R.mean()), 4)
        except ImportError:
            logger.warning("bert_score not installed.")
        except Exception as exc:
            logger.warning("BERTScore failed: %s", exc)

        if section:
            metrics["section"] = section
        return metrics

    def compare_models(
        self,
        transcript,
        reference,
        models: List[str],
    ) -> pd.DataFrame:
        """
        Run multiple SOAP generation models and compare against reference.
        models: list of model identifiers ("medgemma", "gemini", etc.)
        """
        import os
        from medai.pipeline.intelligence.soap_generator import SOAPGenerator

        rows = []
        for model_name in models:
            os.environ["PRIMARY_LLM"] = model_name
            try:
                generator = SOAPGenerator()
                soap = generator.generate(transcript)
                metrics = self.evaluate(soap, reference)
                # Average across sections
                avg_rouge1 = sum(m.get("rouge1_f", 0) for m in metrics.values()) / len(metrics)
                avg_rougeL = sum(m.get("rougeL_f", 0) for m in metrics.values()) / len(metrics)
                avg_bert = sum(m.get("bertscore_f1", 0) for m in metrics.values()) / len(metrics)
                rows.append({
                    "model": model_name,
                    "avg_rouge1_f": round(avg_rouge1, 4),
                    "avg_rougeL_f": round(avg_rougeL, 4),
                    "avg_bertscore_f1": round(avg_bert, 4),
                    "generation_time_ms": round(soap.generation_time_ms, 1),
                })
            except Exception as exc:
                logger.warning("Model %s failed: %s", model_name, exc)
                rows.append({"model": model_name, "error": str(exc)})
        return pd.DataFrame(rows)
