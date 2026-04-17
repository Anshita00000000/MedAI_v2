"""PHI/PII removal from clinical transcripts using spaCy NER + regex."""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

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


# Regex patterns for structured PHI
_DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
    re.IGNORECASE,
)
_PHONE_PATTERN = re.compile(
    r"\b(\+?1?\s?[\(\-]?\d{3}[\)\-\s]?\d{3}[\-\s]?\d{4}"   # US: (555) 123-4567
    r"|(?:\+44|0044|0)7\d{3}\s?\d{6}"                         # UK mobile: 07700 900123
    r"|(?:\+44|0044|0)\d{2,4}\s?\d{3,4}\s?\d{3,4})\b"        # UK landline
)
_MRN_PATTERN = re.compile(
    r"\b(MRN|Medical Record Number|Patient ID|Chart #?)\s*[:#]?\s*([A-Z0-9]{4,12})\b",
    re.IGNORECASE,
)
_INSURANCE_PATTERN = re.compile(
    r"\b(Insurance ID|Policy Number|Member ID)\s*[:#]?\s*([A-Z0-9\-]{6,20})\b",
    re.IGNORECASE,
)
_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd"
    r"|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b",
    re.IGNORECASE,
)
_POSTCODE_PATTERN = re.compile(r"\b[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}\b|\b\d{5}(?:-\d{4})?\b")


@dataclass
class DeIdentifiedResult:
    original_text: str
    clean_text: str
    phi_found: List[dict]
    phi_count: int


@dataclass
class _Replacement:
    start: int
    end: int
    original: str
    replacement: str
    phi_type: str


class DeIdentifier:
    """Removes Protected Health Information from clinical text."""

    def __init__(self):
        self.available = False
        self._nlp = None
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self.available = True
            logger.info("Loaded spaCy en_core_web_sm for de-identification.")
        except Exception as exc:
            logger.warning("spaCy model unavailable: %s. Regex-only de-identification.", exc)

    def deidentify(self, text: str) -> DeIdentifiedResult:
        replacements: List[_Replacement] = []

        replacements.extend(self._regex_phi(text))
        if self.available:
            replacements.extend(self._spacy_phi(text))

        replacements = self._merge_replacements(replacements)

        clean_text, phi_found = self._apply_replacements(text, replacements)

        return DeIdentifiedResult(
            original_text=text,
            clean_text=clean_text,
            phi_found=phi_found,
            phi_count=len(phi_found),
        )

    def deidentify_turns(self, turns: List[SpeakerTurn]) -> List[SpeakerTurn]:
        result = []
        for turn in turns:
            deid = self.deidentify(turn.text)
            result.append(
                SpeakerTurn(
                    speaker_id=turn.speaker_id,
                    role=turn.role,
                    text=deid.clean_text,
                    start=turn.start,
                    end=turn.end,
                )
            )
        return result

    def get_phi_report(self, text: str) -> dict:
        result = self.deidentify(text)
        by_type: dict = {}
        for item in result.phi_found:
            t = item["type"]
            by_type.setdefault(t, []).append(item["original"])
        return {
            "total_phi_found": result.phi_count,
            "by_type": by_type,
        }

    def _regex_phi(self, text: str) -> List[_Replacement]:
        replacements = []

        for m in _DATE_PATTERN.finditer(text):
            replacements.append(_Replacement(m.start(), m.end(), m.group(), "[DATE]", "DATE"))

        for m in _PHONE_PATTERN.finditer(text):
            replacements.append(_Replacement(m.start(), m.end(), m.group(), "[PHONE]", "PHONE"))

        for m in _MRN_PATTERN.finditer(text):
            replacements.append(_Replacement(m.start(), m.end(), m.group(), "[MRN]", "MRN"))

        for m in _INSURANCE_PATTERN.finditer(text):
            replacements.append(
                _Replacement(m.start(), m.end(), m.group(), "[INSURANCE_ID]", "INSURANCE_ID")
            )

        for m in _ADDRESS_PATTERN.finditer(text):
            replacements.append(
                _Replacement(m.start(), m.end(), m.group(), "[ADDRESS]", "ADDRESS")
            )

        for m in _POSTCODE_PATTERN.finditer(text):
            replacements.append(
                _Replacement(m.start(), m.end(), m.group(), "[ADDRESS]", "ADDRESS")
            )

        return replacements

    def _spacy_phi(self, text: str) -> List[_Replacement]:
        doc = self._nlp(text)
        replacements = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Heuristic: decide patient vs provider from context
                surrounding = text[max(0, ent.start_char - 30): ent.end_char + 30].lower()
                if any(w in surrounding for w in ("dr.", "doctor", "dr ", "physician", "nurse")):
                    tag = "[PROVIDER_NAME]"
                    phi_type = "PROVIDER_NAME"
                else:
                    tag = "[PATIENT_NAME]"
                    phi_type = "PATIENT_NAME"
                replacements.append(
                    _Replacement(ent.start_char, ent.end_char, ent.text, tag, phi_type)
                )
            elif ent.label_ == "ORG":
                replacements.append(
                    _Replacement(ent.start_char, ent.end_char, ent.text, "[FACILITY]", "FACILITY")
                )
            elif ent.label_ == "GPE":
                replacements.append(
                    _Replacement(ent.start_char, ent.end_char, ent.text, "[ADDRESS]", "ADDRESS")
                )
        return replacements

    @staticmethod
    def _merge_replacements(replacements: List[_Replacement]) -> List[_Replacement]:
        """Remove overlapping replacements (keep longer span)."""
        replacements.sort(key=lambda r: (r.start, -(r.end - r.start)))
        merged: List[_Replacement] = []
        last_end = -1
        for r in replacements:
            if r.start >= last_end:
                merged.append(r)
                last_end = r.end
        return merged

    @staticmethod
    def _apply_replacements(
        text: str, replacements: List[_Replacement]
    ) -> tuple:
        phi_found = []
        result = []
        prev = 0
        for r in sorted(replacements, key=lambda x: x.start):
            result.append(text[prev: r.start])
            result.append(r.replacement)
            phi_found.append(
                {
                    "type": r.phi_type,
                    "original": r.original,
                    "replacement": r.replacement,
                    "position": (r.start, r.end),
                }
            )
            prev = r.end
        result.append(text[prev:])
        return "".join(result), phi_found
