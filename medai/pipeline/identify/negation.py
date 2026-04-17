"""Negation detection for clinical entities using medSpaCy/negspaCy."""

import logging
from typing import List

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

from medai.pipeline.identify.ner import Entity


class NegationDetector:
    """Detect negated clinical entities using medSpaCy negspaCy component."""

    def __init__(self):
        self.available = False
        self._nlp = None

        try:
            import medspacy
            self._nlp = medspacy.load(enable=["negation_detector"])
            self.available = True
            logger.info("medSpaCy negation detector loaded.")
        except Exception as exc:
            logger.warning(
                "medSpaCy unavailable (%s). Falling back to regex-based negation.", exc
            )
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Using spaCy fallback for negation.")
                self.available = True
            except Exception as exc2:
                logger.warning("spaCy also unavailable: %s", exc2)

    def detect(self, text: str, entities: List[Entity]) -> List[Entity]:
        """Add is_negated field to each entity based on context analysis."""
        if not self.available:
            return entities

        result = []
        for entity in entities:
            is_neg = self._is_negated(text, entity)
            updated = Entity(
                text=entity.text,
                label=entity.label,
                start=entity.start,
                end=entity.end,
                umls_code=entity.umls_code,
                confidence=entity.confidence,
                is_negated=is_neg,
            )
            result.append(updated)
        return result

    def detect_from_turns(
        self,
        turns: List[SpeakerTurn],
        entities: List[Entity],
    ) -> List[Entity]:
        full_text = " ".join(t.text for t in turns)
        return self.detect(full_text, entities)

    def _is_negated(self, text: str, entity: Entity) -> bool:
        """
        Check negation using medSpaCy if available, otherwise regex heuristics.
        """
        context_start = max(0, entity.start - 60)
        context_end = min(len(text), entity.end + 60)
        context = text[context_start:context_end].lower()

        # Regex-based negation heuristics as primary/fallback
        import re
        negation_patterns = [
            r"\b(no|not|denies?|deny|without|absent|rules?\s+out|negative\s+for"
            r"|negative|free\s+of|not\s+present|does?\s+not\s+have"
            r"|does?\s+not\s+appear|does?\s+not\s+show|never|none)\b",
        ]
        entity_text_lower = entity.text.lower()

        for pat in negation_patterns:
            matches = list(re.finditer(pat, context))
            for m in matches:
                # Check if the negation word appears before the entity in context
                neg_pos = m.start()
                ent_pos_in_context = context.find(entity_text_lower)
                if ent_pos_in_context >= 0 and neg_pos < ent_pos_in_context:
                    # Make sure negation word is within 10 words of entity
                    words_between = context[neg_pos:ent_pos_in_context].split()
                    if len(words_between) <= 10:
                        return True

        # Try medSpaCy if available
        if self.available and self._nlp and hasattr(self._nlp, "get_pipe"):
            try:
                doc = self._nlp(text[max(0, entity.start - 100): entity.end + 100])
                for token in doc:
                    if hasattr(token._, "is_negated") and token._.is_negated:
                        return True
            except Exception:
                pass

        return False
