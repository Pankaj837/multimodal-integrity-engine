"""End-to-end orchestration: folder of submissions -> scored results."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import __version__
from .ai_detect import AIDetector
from .boilerplate import mark_boilerplate
from .chunker import build_chunks
from .config import Config
from .embedder import Embedder, resolve_device
from .extractors import SpeechTranscriber, extract_document
from .matching import Collection, Passage, assemble_passages, find_sentence_pairs
from .models import Document, Modality
from .reference import cached_encode, load_reference_documents
from .scoring import coverage_percentage, risk_level, risk_reasons
from .segmentation import blocks_to_sentences

log = logging.getLogger(__name__)

# Files in the input folder that are metadata, not submissions.
IGNORED_FILE_NAMES = {"dataset_manifest.json", "readme.md", "readme.txt", "desktop.ini", "thumbs.db"}


@dataclass
class DocumentResult:
    document: Document
    peer_passages: List[Passage] = field(default_factory=list)
    external_passages: List[Passage] = field(default_factory=list)
    peer_percentage: Optional[float] = None
    external_percentage: Optional[float] = None
    ai: Dict = field(default_factory=dict)
    risk: str = "UNPROCESSABLE"
    reasons: Dict[str, str] = field(default_factory=dict)
    pair_coverage: Dict[int, float] = field(default_factory=dict)  # other doc index -> %


@dataclass
class RunResult:
    results: List[DocumentResult]
    reference_documents: List[Document]
    reference_metadata: List[Dict]
    timings: Dict[str, float]
    boilerplate_stats: Dict[str, int]
    config: Config
    device: str
    version: str = __version__


def discover_files(input_dir: Path) -> List[Path]:
    """All candidate submission files, recursively, in a stable order."""
    files = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.lower() in IGNORED_FILE_NAMES:
            continue
        files.append(path)
    return files


class IntegrityPipeline:
    def __init__(self, config: Config, root: Path):
        self.config = config
        self.root = root
        self.device = resolve_device(config.models.device)
        self.transcriber = SpeechTranscriber(config)
        self._embedder: Optional[Embedder] = None
        self._ai: Optional[AIDetector] = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.config.models.embedding, self.device)
        return self._embedder

    @property
    def ai(self) -> AIDetector:
        if self._ai is None:
            self._ai = AIDetector(self.config, self.root)
        return self._ai

    def _resolve(self, path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else self.root / path

    # ------------------------------------------------------------------ stages
    def run(self, input_dir: Path) -> RunResult:
        """Full pipeline: ingestion (extract) followed by analysis."""
        clock = time.perf_counter()
        documents = self.extract(input_dir)
        return self.analyse(documents, {"extraction": time.perf_counter() - clock})

    def extract(self, input_dir: Path) -> List[Document]:
        """Stage 1 - ingestion: every file -> Document (text blocks + status)."""
        files = discover_files(input_dir)
        log.info("Found %d file(s) in %s", len(files), input_dir)
        documents: List[Document] = []
        for path in files:
            doc_id = path.relative_to(input_dir).as_posix()
            started = time.perf_counter()
            document = extract_document(path, doc_id, self.transcriber)
            document.extraction_warnings = list(document.warnings)
            documents.append(document)
            log.info("Extracted %-28s %-18s %.1fs", doc_id, document.status.value, time.perf_counter() - started)
        return documents

    def analyse(self, documents: List[Document], timings: Optional[Dict[str, float]] = None) -> RunResult:
        """Stages 2-7. Re-entrant: can be re-run on the same extracted documents
        with a different configuration (used by scripts/evaluate.py for
        threshold sweeps and ablations without re-running ASR)."""
        cfg = self.config
        timings = dict(timings or {})

        # 2. Sentences -------------------------------------------------------
        for index, document in enumerate(documents):
            document.warnings = list(document.extraction_warnings)
            document.sentences, document.chunks = [], []
            if document.ok:
                document.sentences = blocks_to_sentences(document.blocks, index, cfg.chunking.min_sentence_words)
                if not any(s.eligible for s in document.sentences):
                    document.warnings.append(
                        "No prose sentence long enough to assess; similarity scores are 0 by definition."
                    )

        # 3. Sentence embeddings + boilerplate -------------------------------
        clock = time.perf_counter()
        keys = [(d_i, s.index) for d_i, d in enumerate(documents) if d.ok for s in d.sentences if s.eligible]
        texts = [documents[d].sentences[s].text for d, s in keys]
        vectors = self.embedder.encode(texts)
        vector_of = {key: vectors[row] for row, key in enumerate(keys)}
        template_vectors = self._template_vectors()
        boilerplate_stats = mark_boilerplate(documents, keys, vectors, cfg.boilerplate, template_vectors)

        # 4. Chunks over assessed (non-boilerplate) sentences -----------------
        for index, document in enumerate(documents):
            if document.ok:
                document.chunks = build_chunks(
                    document.sentences,
                    index,
                    self.embedder.count_tokens,
                    cfg.chunking.max_tokens,
                    cfg.chunking.overlap_sentences,
                )
                too_long = [c for c in document.chunks if c.n_tokens + 2 > self.embedder.max_seq_length]
                if too_long:
                    document.warnings.append(
                        f"{len(too_long)} chunk(s) exceed the model's {self.embedder.max_seq_length}-token "
                        "window (a single very long sentence); their tails are truncated when embedded."
                    )
        submissions = self._collection(documents, vector_of)
        timings["embedding_and_indexing"] = time.perf_counter() - clock

        # 5. Reference corpus --------------------------------------------------
        clock = time.perf_counter()
        ref_docs, ref_meta = load_reference_documents(
            self._resolve(cfg.reference_dir), cfg.chunking.min_sentence_words
        )
        references = self._reference_collection(ref_docs)
        timings["reference_corpus"] = time.perf_counter() - clock

        # 6. Matching ----------------------------------------------------------
        clock = time.perf_counter()
        peer_pairs = find_sentence_pairs(submissions, submissions, cfg.matching, same_corpus=True)
        peer_passages = assemble_passages(submissions, submissions, peer_pairs, cfg.matching)
        external_passages: Dict[int, List[Passage]] = {}
        if references is not None:
            external_pairs = find_sentence_pairs(submissions, references, cfg.matching, same_corpus=False)
            external_passages = assemble_passages(submissions, references, external_pairs, cfg.matching)
        timings["matching"] = time.perf_counter() - clock

        # 7. Scores, AI likelihood, risk ---------------------------------------
        clock = time.perf_counter()
        results = []
        for index, document in enumerate(documents):
            result = DocumentResult(document=document)
            if document.ok:
                result.peer_passages = peer_passages.get(index, [])
                result.external_passages = external_passages.get(index, [])
                result.peer_percentage = coverage_percentage(
                    document, (s.index for p in result.peer_passages for s in p.query_sentences)
                )
                result.external_percentage = coverage_percentage(
                    document, (s.index for p in result.external_passages for s in p.query_sentences)
                )
                for other in {p.target_doc for p in result.peer_passages}:
                    result.pair_coverage[other] = coverage_percentage(
                        document,
                        (s.index for p in result.peer_passages if p.target_doc == other for s in p.query_sentences),
                    )
                result.ai = self._assess_ai(document)
                result.risk = risk_level(
                    result.peer_percentage,
                    result.external_percentage,
                    result.ai.get("probability"),
                    result.ai.get("confidence", "normal"),
                    cfg.risk,
                )
                result.reasons = risk_reasons(
                    result.peer_percentage,
                    result.external_percentage,
                    result.ai.get("probability"),
                    result.ai.get("confidence", "normal"),
                    cfg.risk,
                )
            results.append(result)
        timings["ai_and_scoring"] = time.perf_counter() - clock

        return RunResult(
            results=results,
            reference_documents=ref_docs,
            reference_metadata=ref_meta,
            timings=timings,
            boilerplate_stats=boilerplate_stats,
            config=cfg,
            device=self.device,
        )

    # ----------------------------------------------------------------- helpers
    def _assess_ai(self, document: Document) -> Dict:
        if not self.config.ai.enabled:
            return {"probability": None, "status": "DISABLED", "confidence": "n/a", "signals": {}, "notes": []}
        try:
            return self.ai.assess(document)
        except Exception as exc:  # model download / runtime failure must not stop the batch
            log.exception("AI detection failed for %s", document.doc_id)
            return {
                "probability": None,
                "status": "ERROR",
                "confidence": "n/a",
                "signals": {},
                "notes": [f"AI detection failed: {type(exc).__name__}: {exc}"],
            }

    def _collection(self, documents: List[Document], vector_of: Dict) -> Collection:
        sentence_keys = [(d_i, s.index) for d_i, d in enumerate(documents) if d.ok for s in d.sentences if s.assessed]
        dim = self.embedder.dim
        sentence_vectors = (
            np.stack([vector_of[k] for k in sentence_keys]) if sentence_keys else np.zeros((0, dim), np.float32)
        )
        chunk_keys = [(d_i, c.index) for d_i, d in enumerate(documents) if d.ok for c in d.chunks]
        chunk_texts = [documents[d].chunks[c].text for d, c in chunk_keys]
        chunk_vectors = self.embedder.encode(chunk_texts)
        return Collection(documents, sentence_keys, sentence_vectors, chunk_keys, chunk_vectors)

    def _reference_collection(self, ref_docs: List[Document]) -> Optional[Collection]:
        if not ref_docs:
            return None
        cfg = self.config
        for index, document in enumerate(ref_docs):
            document.chunks = build_chunks(
                document.sentences, index, self.embedder.count_tokens,
                cfg.chunking.max_tokens, cfg.chunking.overlap_sentences,
            )
        cache_dir = self._resolve(cfg.cache_dir)
        sentence_keys = [(d_i, s.index) for d_i, d in enumerate(ref_docs) for s in d.sentences if s.assessed]
        sentence_texts = [ref_docs[d].sentences[s].text for d, s in sentence_keys]
        chunk_keys = [(d_i, c.index) for d_i, d in enumerate(ref_docs) for c in d.chunks]
        chunk_texts = [ref_docs[d].chunks[c].text for d, c in chunk_keys]
        model = cfg.models.embedding
        sentence_vectors = cached_encode(cache_dir, "ref_sentences", model, sentence_texts, self.embedder.encode)
        chunk_vectors = cached_encode(cache_dir, "ref_chunks", model, chunk_texts, self.embedder.encode)
        return Collection(ref_docs, sentence_keys, sentence_vectors, chunk_keys, chunk_vectors)

    def _template_vectors(self) -> Optional[np.ndarray]:
        template = self.config.boilerplate.template_file
        if not template:
            return None
        path = self._resolve(template)
        document = extract_document(path, path.name, self.transcriber)
        if not document.ok:
            log.warning("Template file %s could not be read (%s); ignoring it.", path, document.status.value)
            return None
        sentences = blocks_to_sentences(document.blocks, -1, self.config.chunking.min_sentence_words)
        texts = [s.text for s in sentences if s.eligible]
        log.info("Template file %s contributes %d sentence(s)", path.name, len(texts))
        return self.embedder.encode(texts)


def is_cross_modal(a: Document, b: Document) -> bool:
    """True when exactly one side is a speech transcript."""
    return (a.modality == Modality.SPEECH) != (b.modality == Modality.SPEECH)
