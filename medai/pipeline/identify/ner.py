"""Clinical Named Entity Recognition with multi-model support."""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from medai.pipeline.asr.diarise import SpeakerTurn
except ImportError:
    from dataclasses import dataclass as _dc

    @_dc
    class SpeakerTurn:  # type: ignore[no-redef]
        speaker_id: str = ""
        role: str = ""
        text: str = ""
        start: float = 0.0
        end: float = 0.0


ENTITY_LABELS = {
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE",
    "PROCEDURE", "ANATOMY", "LAB_TEST", "LAB_VALUE",
}


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    umls_code: Optional[str] = None
    confidence: float = 1.0
    is_negated: bool = False


@dataclass
class NERResult:
    entities: List[Entity]
    model_used: str
    extraction_time_ms: float


class ClinicalNER:
    """Multi-model clinical NER with scispaCy, ClinicalBERT, and GLiNER."""

    def __init__(self):
        self.available = False
        self._scispacy_nlp = None
        self._linker = None
        self._bert_pipeline = None
        self._gliner_model = None

        self._load_scispacy()
        self._load_bert()
        self._load_gliner()

        if self._scispacy_nlp or self._bert_pipeline or self._gliner_model:
            self.available = True

    def _load_scispacy(self):
        try:
            import spacy
            self._scispacy_nlp = spacy.load("en_core_sci_md")
            try:
                from scispacy.linking import EntityLinker
                self._scispacy_nlp.add_pipe(
                    "scispacy_linker",
                    config={"resolve_abbreviations": True, "linker_name": "umls"},
                    last=True,
                )
                logger.info("scispaCy loaded with UMLS linker.")
            except Exception as exc:
                logger.warning("UMLS linker unavailable: %s", exc)
                logger.info("scispaCy loaded without UMLS linker.")
        except Exception as exc:
            logger.warning("scispaCy en_core_sci_md unavailable: %s", exc)

    def _load_bert(self):
        try:
            from transformers import pipeline as hf_pipeline
            self._bert_pipeline = hf_pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
            )
            logger.info("ClinicalBERT NER loaded.")
        except Exception as exc:
            logger.warning("ClinicalBERT NER unavailable: %s", exc)

    def _load_gliner(self):
        try:
            from gliner import GLiNER
            self._gliner_model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
            logger.info("GLiNER loaded.")
        except Exception as exc:
            logger.warning("GLiNER unavailable: %s", exc)

    def extract(self, text: str, model: str = "scispacy") -> NERResult:
        t0 = time.time()
        if model == "scispacy":
            entities = self._extract_scispacy(text)
            model_name = "scispacy/en_core_sci_md"
        elif model == "bert":
            entities = self._extract_bert(text)
            model_name = "bert/d4data-biomedical-ner-all"
        elif model == "gliner":
            entities = self._extract_gliner(text)
            model_name = "gliner/urchade-gliner_mediumv2.1"
        else:
            raise ValueError(f"Unknown model: {model}")
        elapsed = (time.time() - t0) * 1000
        return NERResult(entities=entities, model_used=model_name, extraction_time_ms=elapsed)

    def extract_all_models(self, text: str) -> Dict[str, NERResult]:
        results = {}
        for model in ("scispacy", "bert", "gliner"):
            try:
                results[model] = self.extract(text, model)
            except Exception as exc:
                logger.warning("Model %s failed: %s", model, exc)
        return results

    def extract_from_turns(
        self,
        turns: List[SpeakerTurn],
        speaker_filter: Optional[str] = None,
    ) -> NERResult:
        filtered = turns
        if speaker_filter:
            filtered = [t for t in turns if t.role.upper() == speaker_filter.upper()]

        combined = " ".join(t.text for t in filtered)
        return self.extract(combined)

    # ------------------------------------------------------------------ #
    # Model-specific extraction helpers
    # ------------------------------------------------------------------ #

    def _extract_scispacy(self, text: str) -> List[Entity]:
        if not self._scispacy_nlp:
            raise RuntimeError("scispaCy not available.")
        doc = self._scispacy_nlp(text)
        entities: List[Entity] = []

        label_map = {
            "DISEASE": "DIAGNOSIS",
            "CHEMICAL": "MEDICATION",
            "GENE_OR_GENE_PRODUCT": "ANATOMY",
            "SIMPLE_CHEMICAL": "MEDICATION",
            "ORGANISM": "DIAGNOSIS",
            "CELL": "ANATOMY",
            "CELLULAR_COMPONENT": "ANATOMY",
            "DEVELOPING_ANATOMICAL_STRUCTURE": "ANATOMY",
            "ORGAN": "ANATOMY",
            "TISSUE": "ANATOMY",
        }

        for ent in doc.ents:
            mapped_label = label_map.get(ent.label_, ent.label_)
            if mapped_label not in ENTITY_LABELS:
                mapped_label = "DIAGNOSIS"

            umls_code = None
            confidence = 1.0
            if hasattr(ent, "_.kb_ents") and ent._.kb_ents:
                top = ent._.kb_ents[0]
                umls_code = top[0]
                confidence = float(top[1])

            entities.append(
                Entity(
                    text=ent.text,
                    label=mapped_label,
                    start=ent.start_char,
                    end=ent.end_char,
                    umls_code=umls_code,
                    confidence=confidence,
                )
            )
        return entities

    def _extract_bert(self, text: str) -> List[Entity]:
        if not self._bert_pipeline:
            raise RuntimeError("ClinicalBERT NER not available.")
        raw = self._bert_pipeline(text)
        entities: List[Entity] = []

        label_map = {
            "DISEASE_DISORDER": "DIAGNOSIS",
            "SIGN_SYMPTOM": "SYMPTOM",
            "MEDICATION": "MEDICATION",
            "DOSAGE": "DOSAGE",
            "PROCEDURE": "PROCEDURE",
            "BODY_PART": "ANATOMY",
            "LAB_VALUE": "LAB_VALUE",
            "LAB_TEST": "LAB_TEST",
        }

        for item in raw:
            mapped = label_map.get(item["entity_group"], "DIAGNOSIS")
            entities.append(
                Entity(
                    text=item["word"],
                    label=mapped,
                    start=item["start"],
                    end=item["end"],
                    confidence=float(item["score"]),
                )
            )
        return entities

    def _extract_gliner(self, text: str) -> List[Entity]:
        if not self._gliner_model:
            raise RuntimeError("GLiNER not available.")
        labels = list(ENTITY_LABELS)
        raw = self._gliner_model.predict_entities(text, labels, threshold=0.5)
        entities: List[Entity] = []
        for item in raw:
            entities.append(
                Entity(
                    text=item["text"],
                    label=item["label"],
                    start=item["start"],
                    end=item["end"],
                    confidence=float(item["score"]),
                )
            )
        return entities
