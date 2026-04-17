"""Multi-model ASR transcription module."""

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    segments: List[dict]
    language: str
    model_used: str
    duration_seconds: float
    wer: Optional[float] = None


class ASRTranscriber:
    """
    Multi-backend ASR transcriber.

    Backends:
      whisper_local  — openai-whisper (runs locally)
      whisper_api    — OpenAI Whisper API
      medasr         — placeholder for MedASR
    """

    VALID_BACKENDS = {"whisper_local", "whisper_api", "medasr"}

    def __init__(self):
        self.backend = os.getenv("ASR_MODEL", "whisper_local")
        self.model_size = os.getenv("WHISPER_MODEL_SIZE", "base.en")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.available = True
        self._whisper_model = None

        if self.backend not in self.VALID_BACKENDS:
            logger.warning("Unknown ASR_MODEL '%s', defaulting to whisper_local", self.backend)
            self.backend = "whisper_local"

        if self.backend == "whisper_local":
            self._load_whisper_local()
        elif self.backend == "whisper_api":
            if not self.openai_api_key:
                logger.warning("OPENAI_API_KEY not set; whisper_api backend unavailable.")
                self.available = False
        elif self.backend == "medasr":
            logger.warning("MedASR backend is a placeholder — not yet implemented.")
            self.available = False

    def _load_whisper_local(self):
        try:
            import whisper
            self._whisper_model = whisper.load_model(self.model_size)
            logger.info("Loaded Whisper model: %s", self.model_size)
        except Exception as exc:
            logger.warning("Could not load Whisper model '%s': %s", self.model_size, exc)
            self.available = False

    def transcribe(
        self,
        audio_path: str,
        reference_text: Optional[str] = None,
    ) -> TranscriptionResult:
        if not self.available:
            raise RuntimeError(f"ASR backend '{self.backend}' is not available.")

        t0 = time.time()

        if self.backend == "whisper_local":
            result = self._transcribe_whisper_local(audio_path)
        elif self.backend == "whisper_api":
            result = self._transcribe_whisper_api(audio_path)
        else:
            raise NotImplementedError(f"Backend '{self.backend}' not implemented.")

        result.duration_seconds = time.time() - t0

        if reference_text:
            result.wer = self._compute_wer(result.text, reference_text)

        return result

    def _transcribe_whisper_local(self, audio_path: str) -> TranscriptionResult:
        raw = self._whisper_model.transcribe(audio_path)
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in raw.get("segments", [])
        ]
        return TranscriptionResult(
            text=raw["text"].strip(),
            segments=segments,
            language=raw.get("language", "en"),
            model_used=f"whisper_local/{self.model_size}",
            duration_seconds=0.0,
        )

    def _transcribe_whisper_api(self, audio_path: str) -> TranscriptionResult:
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_api_key)
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
            )
        segments = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in (response.segments or [])
        ]
        return TranscriptionResult(
            text=response.text.strip(),
            segments=segments,
            language=response.language or "en",
            model_used="whisper_api/whisper-1",
            duration_seconds=0.0,
        )

    @staticmethod
    def _compute_wer(hypothesis: str, reference: str) -> float:
        try:
            from jiwer import wer
            return float(wer(reference, hypothesis))
        except Exception:
            return -1.0

    def batch_transcribe(self, audio_dir: str) -> List[TranscriptionResult]:
        audio_dir = Path(audio_dir)
        extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        files = [p for p in audio_dir.iterdir() if p.suffix.lower() in extensions]
        results = []
        for f in sorted(files):
            try:
                results.append(self.transcribe(str(f)))
            except Exception as exc:
                logger.error("Failed to transcribe %s: %s", f, exc)
        return results
