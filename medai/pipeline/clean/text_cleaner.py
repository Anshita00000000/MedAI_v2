"""Transcript cleaning for audio-derived and pre-existing text transcripts."""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# SpeakerTurn imported lazily to avoid circular imports
# (pipeline.asr.diarise defines SpeakerTurn)
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


MEDICAL_ABBREVIATIONS = {
    r"\bSOB\b": "shortness of breath",
    r"\bHTN\b": "hypertension",
    r"\bDM\b": "diabetes mellitus",
    r"\bDM2\b": "type 2 diabetes mellitus",
    r"\bDM1\b": "type 1 diabetes mellitus",
    r"\bCP\b": "chest pain",
    r"\bh/o\b": "history of",
    r"\bw/o\b": "without",
    r"\bc/o\b": "complains of",
    r"\bHx\b": "history",
    r"\bPMH\b": "past medical history",
    r"\bFH\b": "family history",
    r"\bSH\b": "social history",
    r"\bRx\b": "prescription",
    r"\bNKDA\b": "no known drug allergies",
    r"\bNKA\b": "no known allergies",
    r"\bBP\b": "blood pressure",
    r"\bHR\b": "heart rate",
    r"\bRR\b": "respiratory rate",
    r"\bT\b": "temperature",
    r"\bO2\b": "oxygen",
    r"\bSaO2\b": "oxygen saturation",
    r"\bBMI\b": "body mass index",
    r"\bECG\b": "electrocardiogram",
    r"\bEKG\b": "electrocardiogram",
    r"\bCXR\b": "chest X-ray",
    r"\bCT\b": "computed tomography",
    r"\bMRI\b": "magnetic resonance imaging",
    r"\bIV\b": "intravenous",
    r"\bIM\b": "intramuscular",
    r"\bPO\b": "by mouth",
    r"\bPRN\b": "as needed",
    r"\bBID\b": "twice daily",
    r"\bTID\b": "three times daily",
    r"\bQID\b": "four times daily",
    r"\bQD\b": "once daily",
    r"\bSL\b": "sublingual",
    r"\bstat\b": "immediately",
    r"\bA&E\b": "accident and emergency",
    r"\bED\b": "emergency department",
    r"\bICU\b": "intensive care unit",
    r"\bVS\b": "vital signs",
    r"\bWNL\b": "within normal limits",
}

FILLER_WORDS = re.compile(
    r"\b(um+|uh+|er+|ah+|hmm+|hm+|like|you know|i mean|sort of|kind of)\b",
    re.IGNORECASE,
)
REPEATED_WORDS = re.compile(r"\b(\w+)( \1)+\b", re.IGNORECASE)
ENCODING_ARTIFACT = re.compile(r"_x000D_")
MULTI_SPACE = re.compile(r" {2,}")


@dataclass
class CleanedTranscript:
    turns: List[SpeakerTurn]
    raw_text: str
    cleaned_text: str
    input_format_detected: str
    num_turns: int


class TranscriptCleaner:

    def clean(self, raw_text: str, input_format: str = "auto") -> CleanedTranscript:
        detected = input_format

        if input_format == "auto":
            detected = self._detect_format(raw_text)

        if detected == "raw_asr":
            cleaned = self.clean_raw_asr(raw_text)
            turns = [SpeakerTurn(speaker_id="SPEAKER_0", role="", text=cleaned)]
        elif detected == "prefixed":
            turns = self.clean_prefixed(raw_text)
            cleaned = "\n".join(f"{t.speaker_id}: {t.text}" for t in turns)
        elif detected in ("jsonl", "excel"):
            turns = self._clean_structured(raw_text, detected)
            cleaned = "\n".join(f"{t.speaker_id}: {t.text}" for t in turns)
        else:
            cleaned = self.clean_raw_asr(raw_text)
            turns = [SpeakerTurn(speaker_id="SPEAKER_0", role="", text=cleaned)]
            detected = "raw_asr"

        return CleanedTranscript(
            turns=turns,
            raw_text=raw_text,
            cleaned_text=cleaned,
            input_format_detected=detected,
            num_turns=len(turns),
        )

    @staticmethod
    def _detect_format(text: str) -> str:
        lines = text.strip().splitlines()[:10]
        prefixed = sum(1 for l in lines if re.match(r"^(Doctor|Patient|Nurse|Family)\s*:", l, re.I))
        if prefixed >= 2:
            return "prefixed"
        if text.strip().startswith("{") or text.strip().startswith("["):
            return "jsonl"
        return "raw_asr"

    def clean_raw_asr(self, text: str) -> str:
        text = ENCODING_ARTIFACT.sub(" ", text)
        text = FILLER_WORDS.sub("", text)
        text = REPEATED_WORDS.sub(r"\1", text)
        # Fix incomplete sentences that end mid-word due to cut-off
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = MULTI_SPACE.sub(" ", text)
        text = text.strip()
        return text

    def clean_prefixed(self, text: str) -> List[SpeakerTurn]:
        turns: List[SpeakerTurn] = []
        current_speaker: Optional[str] = None
        current_lines: List[str] = []

        for line in text.splitlines():
            m = re.match(r"^(Doctor|Patient|Nurse|Family)\s*:\s*(.*)", line.strip(), re.I)
            if m:
                if current_speaker and current_lines:
                    turns.append(
                        SpeakerTurn(
                            speaker_id=current_speaker,
                            role="",
                            text=self.clean_raw_asr(" ".join(current_lines)),
                        )
                    )
                current_speaker = m.group(1).upper()
                current_lines = [m.group(2).strip()] if m.group(2).strip() else []
            else:
                stripped = line.strip()
                if stripped and current_speaker:
                    current_lines.append(stripped)

        if current_speaker and current_lines:
            turns.append(
                SpeakerTurn(
                    speaker_id=current_speaker,
                    role="",
                    text=self.clean_raw_asr(" ".join(current_lines)),
                )
            )

        return turns

    def _clean_structured(self, text: str, fmt: str) -> List[SpeakerTurn]:
        import json
        turns: List[SpeakerTurn] = []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    turns.append(
                        SpeakerTurn(
                            speaker_id=str(item.get("speaker", "UNKNOWN")).upper(),
                            role="",
                            text=self.clean_raw_asr(str(item.get("text", ""))),
                        )
                    )
        except Exception as exc:
            logger.warning("Could not parse structured format '%s': %s", fmt, exc)
        return turns

    def normalise_medical_terms(self, text: str) -> str:
        for pattern, replacement in MEDICAL_ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text)
        return text
