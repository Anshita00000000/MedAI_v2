"""Speaker diarisation using pyannote.audio."""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class DiarisationResult:
    segments: List[dict]   # [{start, end, speaker}]
    num_speakers: int
    model_used: str


@dataclass
class SpeakerTurn:
    speaker_id: str    # "SPEAKER_0", "SPEAKER_1", etc.
    role: str = ""     # filled in by role_classifier later
    text: str = ""
    start: float = 0.0
    end: float = 0.0


class SpeakerDiariser:
    """Speaker diarisation with pyannote.audio."""

    def __init__(self):
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
        self.model_name = os.getenv(
            "DIARISATION_MODEL", "pyannote/speaker-diarization-3.1"
        )
        self.available = False
        self._pipeline = None

        if not self.hf_token:
            logger.warning("HUGGINGFACE_TOKEN not set; diarisation unavailable.")
            return

        try:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                self.model_name, use_auth_token=self.hf_token
            )
            self.available = True
            logger.info("Loaded diarisation model: %s", self.model_name)
        except Exception as exc:
            logger.warning("Could not load diarisation model: %s", exc)

    def diarise(self, audio_path: str) -> DiarisationResult:
        if not self.available:
            raise RuntimeError("Diarisation model is not available.")

        diarization = self._pipeline(audio_path)
        segments = []
        speakers: set = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {"start": turn.start, "end": turn.end, "speaker": speaker}
            )
            speakers.add(speaker)

        segments.sort(key=lambda s: s["start"])
        return DiarisationResult(
            segments=segments,
            num_speakers=len(speakers),
            model_used=self.model_name,
        )

    def merge_with_transcript(
        self,
        transcription,  # TranscriptionResult
        diarisation: DiarisationResult,
    ) -> List[SpeakerTurn]:
        """
        Align ASR word timestamps with diarisation segments.
        Assigns the speaker who occupies the majority of each ASR segment.
        """
        turns: List[SpeakerTurn] = []

        for seg in transcription.segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_text = seg["text"]

            # Calculate overlap with each diarisation segment
            speaker_overlap: dict = {}
            for d_seg in diarisation.segments:
                overlap_start = max(seg_start, d_seg["start"])
                overlap_end = min(seg_end, d_seg["end"])
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > 0:
                    spk = d_seg["speaker"]
                    speaker_overlap[spk] = speaker_overlap.get(spk, 0.0) + overlap

            if speaker_overlap:
                dominant_speaker = max(speaker_overlap, key=speaker_overlap.get)
            else:
                dominant_speaker = "SPEAKER_UNKNOWN"

            turns.append(
                SpeakerTurn(
                    speaker_id=dominant_speaker,
                    role="",
                    text=seg_text,
                    start=seg_start,
                    end=seg_end,
                )
            )

        return turns
