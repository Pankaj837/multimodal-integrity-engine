"""Audio/video transcription with sentence-level timestamps.

Steps (the "pipeline integration" the brief's FAQ asks for):

1. Open the container with PyAV and confirm it has an audio stream
   (an MP4 without one is reported as NO_AUDIO_TRACK, not as an error).
2. Decode and transcribe the audio with faster-whisper, a CTranslate2
   re-implementation of OpenAI Whisper that loads the same open weights.
   It decodes audio through PyAV, so no system ffmpeg install is needed.
3. Guard against Whisper's best-known failure - hallucinating phrases such as
   "Thank you." on silence - with (a) Silero voice-activity detection and
   (b) Whisper's own no-speech probability / log-probability rule.
4. Re-assemble word-level timestamps into sentences, so every sentence
   carries an exact start and end time for the report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import Config
from ..models import Block, SentenceKind, Status
from ..normalize import normalize_text
from .base import ExtractionError

log = logging.getLogger(__name__)

_SENTENCE_END = (".", "?", "!")


class SpeechTranscriber:
    """Lazily loads the Whisper model once and reuses it for every file."""

    def __init__(self, config: Config):
        self.config = config
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            # CTranslate2 needs separate CUDA/cuDNN libraries; use the GPU only
            # when explicitly requested so CPU-only machines never fail here.
            device = "cuda" if self.config.models.device == "cuda" else "cpu"
            log.info("Loading Whisper '%s' on %s", self.config.models.whisper, device)
            self._model = WhisperModel(
                self.config.models.whisper,
                device=device,
                compute_type=self.config.models.whisper_compute_type,
            )
        return self._model

    def transcribe(self, path: Path) -> Tuple[List[Block], Optional[float], Optional[str], List[str]]:
        """Return (sentence_blocks, duration_seconds, language, warnings)."""
        duration = _probe_audio(path)
        asr = self.config.asr
        segments, info = self._load().transcribe(
            str(path),
            beam_size=asr.beam_size,
            temperature=0.0,  # deterministic decoding
            vad_filter=asr.vad_filter,
            word_timestamps=True,
            condition_on_previous_text=False,  # limits repetition loops
        )

        words = []
        for segment in segments:
            if segment.no_speech_prob > asr.no_speech_threshold and segment.avg_logprob < asr.log_prob_threshold:
                continue  # Whisper itself considers this segment silence
            words.extend(w for w in (segment.words or []) if w.word.strip())

        warnings: List[str] = []
        if len(words) < asr.min_words:
            raise ExtractionError(
                Status.NO_SPEECH,
                f"No intelligible speech detected (recognised {len(words)} word(s)); "
                "the file was flagged and skipped.",
            )
        language = info.language
        if language != "en" and info.language_probability >= 0.5:
            warnings.append(
                f"Detected spoken language '{language}' (p={info.language_probability:.2f}); "
                "the embedding model is English-only, so similarity may be under-estimated."
            )
        return _words_to_sentences(words), duration or info.duration, language, warnings


def _probe_audio(path: Path) -> Optional[float]:
    """Return the container duration, raising if there is no audio stream."""
    import av

    try:
        with av.open(str(path)) as container:
            has_audio = len(container.streams.audio) > 0
            duration = container.duration / 1_000_000 if container.duration else None
    except Exception as exc:  # av.error.InvalidDataError and friends
        raise ExtractionError(Status.CORRUPT_FILE, f"Media container could not be read: {exc}") from exc
    if not has_audio:
        raise ExtractionError(
            Status.NO_AUDIO_TRACK,
            "Video has no audio track; nothing to transcribe. The file was flagged and skipped.",
        )
    return duration


def _words_to_sentences(words) -> List[Block]:
    """Group timestamped words into sentence blocks at terminal punctuation."""
    blocks: List[Block] = []
    buffer = []

    def flush() -> None:
        if buffer:
            text = normalize_text("".join(w.word for w in buffer)).strip()
            blocks.append(
                Block(
                    text=text,
                    kind=SentenceKind.TEXT,
                    start=round(buffer[0].start, 2),
                    end=round(buffer[-1].end, 2),
                    is_sentence=True,
                )
            )
            buffer.clear()

    for word in words:
        buffer.append(word)
        if word.word.strip().endswith(_SENTENCE_END):
            flush()
    flush()
    return blocks
