"""NER evaluation: precision, recall, F1 per entity type."""

import logging
from collections import defaultdict
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


class NEREvaluator:

    def evaluate(self, predictions, ground_truth) -> dict:
        """
        Compute precision, recall, F1 per entity type.
        predictions and ground_truth are NERResult objects.
        Matching is based on (text.lower(), label) tuples.
        """
        pred_by_type: Dict[str, set] = defaultdict(set)
        gold_by_type: Dict[str, set] = defaultdict(set)

        for ent in predictions.entities:
            pred_by_type[ent.label].add(ent.text.lower())

        for ent in ground_truth.entities:
            gold_by_type[ent.label].add(ent.text.lower())

        all_labels = set(pred_by_type) | set(gold_by_type)
        results: dict = {}

        for label in all_labels:
            pred_set = pred_by_type[label]
            gold_set = gold_by_type[label]
            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            results[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn,
            }

        # Macro averages
        if results:
            results["macro"] = {
                "precision": round(sum(r["precision"] for r in results.values()) / len(results), 4),
                "recall": round(sum(r["recall"] for r in results.values()) / len(results), 4),
                "f1": round(sum(r["f1"] for r in results.values()) / len(results), 4),
            }

        return results

    def compare_models(self, text: str, ground_truth) -> pd.DataFrame:
        """
        Run all NER models on the same text and compare against ground truth.
        """
        from medai.pipeline.identify.ner import ClinicalNER
        ner = ClinicalNER()
        rows = []
        for model_name in ("scispacy", "bert", "gliner"):
            try:
                result = ner.extract(text, model=model_name)
                metrics = self.evaluate(result, ground_truth)
                macro = metrics.get("macro", {})
                rows.append({
                    "model": model_name,
                    "macro_precision": macro.get("precision", 0),
                    "macro_recall": macro.get("recall", 0),
                    "macro_f1": macro.get("f1", 0),
                    "num_entities": len(result.entities),
                    "extraction_time_ms": round(result.extraction_time_ms, 1),
                })
            except Exception as exc:
                logger.warning("NER model %s failed: %s", model_name, exc)
                rows.append({"model": model_name, "error": str(exc)})
        return pd.DataFrame(rows)
