"""Typed configuration with defaults, overridable from a YAML file.

Every tunable number in the pipeline lives here so that thresholds are
documented in one place and can be changed without touching code. Unknown
keys in the YAML file raise an error, which catches typos such as
``sentence_treshold`` that would otherwise be silently ignored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    whisper: str = "base"  # tiny | base | small ... (faster-whisper / CTranslate2 weights)
    whisper_compute_type: str = "int8"  # int8 keeps CPU inference fast and memory-light
    ai_perplexity: str = "openai-community/gpt2"
    ai_classifier: str = "Hello-SimpleAI/chatgpt-detector-roberta"
    device: str = "auto"  # auto | cpu | cuda


@dataclass
class AsrConfig:
    beam_size: int = 5
    vad_filter: bool = True  # Silero VAD: stops Whisper hallucinating text on silence
    no_speech_threshold: float = 0.6  # drop segments Whisper itself believes are silence
    log_prob_threshold: float = -1.0
    min_words: int = 3  # fewer recognised words than this => treated as NO_SPEECH


@dataclass
class ChunkingConfig:
    max_tokens: int = 200  # MiniLM truncates at 256 word pieces; 200 leaves headroom
    overlap_sentences: int = 1  # trailing sentences repeated at the start of the next chunk
    min_sentence_words: int = 6  # shorter sentences are too generic to count as evidence


@dataclass
class MatchingConfig:
    top_k: int = 10  # neighbours retrieved per sentence (after removing self-matches)
    sentence_threshold: float = 0.85  # cosine for a sentence to count as matched on its own
    context_chunk_threshold: float = 0.85  # chunk cosine that marks two passages as related
    context_sentence_threshold: float = 0.60  # relaxed sentence cosine inside related chunks
    single_sentence_min_similarity: float = 0.90  # an isolated 1-sentence semantic match needs this
    lexical_rescue_cosine: float = 0.70  # below sentence_threshold, accept if wording is near-verbatim
    passage_gap: int = 1  # max unmatched sentences allowed inside one merged passage
    verbatim_levenshtein: float = 0.90  # lexical similarity above which a match is "verbatim"
    near_verbatim_levenshtein: float = 0.70


@dataclass
class BoilerplateConfig:
    enabled: bool = True
    similarity_threshold: float = 0.90  # near-identical sentences
    min_documents: int = 3  # a sentence shared by >= this many documents is template text
    min_document_fraction: float = 0.10  # ...and by >= this share of the corpus (scales to 10k)
    template_file: Optional[str] = None  # optional known prompt/template to always discount


@dataclass
class AiConfig:
    enabled: bool = True
    min_words: int = 80  # below this the estimate is reported as "insufficient text"
    max_tokens_per_window: int = 512
    calibration_file: str = "integrity/resources/ai_calibration.json"


@dataclass
class RiskConfig:
    high_similarity: float = 30.0  # peer or external % at/above which risk is HIGH
    medium_similarity: float = 10.0
    high_ai: float = 85.0
    medium_ai: float = 60.0


@dataclass
class ReportConfig:
    max_matches_in_markdown: int = 8
    max_passage_chars: int = 700


@dataclass
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    boilerplate: BoilerplateConfig = field(default_factory=BoilerplateConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    reference_dir: str = "reference_corpus"
    cache_dir: str = ".cache"

    def to_dict(self) -> dict:
        return asdict(self)


def _apply(target: Any, overrides: dict, path: str = "") -> None:
    """Recursively copy ``overrides`` onto a dataclass, rejecting unknown keys."""
    known = {f.name: f for f in fields(target)}
    for key, value in overrides.items():
        if key not in known:
            raise ValueError(f"Unknown config key '{path}{key}'. Valid keys: {sorted(known)}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config section '{path}{key}' must be a mapping")
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(target, key, value)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Return defaults, overridden by the YAML file at ``path`` if given."""
    config = Config()
    if path is None:
        return config
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    _apply(config, data)
    return config
