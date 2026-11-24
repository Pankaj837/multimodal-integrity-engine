"""External reference corpus ("simulated web") loading and embedding cache.

The brief allows simulating web search with a local knowledge base. Each
reference article becomes a Document whose sentences carry a paragraph
number, so external matches can be cited as "<article>, paragraph N" with a
link to the exact Wikipedia revision.

Reference embeddings are cached on disk, keyed by a hash of the model name
and the corpus text, so repeated runs do not re-embed ~2,000 sentences.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .models import Block, Document, SentenceKind
from .segmentation import blocks_to_sentences

log = logging.getLogger(__name__)


def load_reference_documents(reference_dir: Path, min_sentence_words: int) -> Tuple[List[Document], List[Dict]]:
    """Return (documents, metadata). Missing directory -> empty corpus + warning."""
    if not reference_dir.is_dir():
        log.warning("Reference corpus '%s' not found; external similarity will be 0.", reference_dir)
        return [], []
    manifest_path = reference_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    meta_by_file = {entry["file"]: entry for entry in manifest}

    documents, metadata = [], []
    for path in sorted(reference_dir.glob("*.txt")):
        paragraphs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        blocks = [Block(text=p, kind=SentenceKind.TEXT, paragraph=i + 1) for i, p in enumerate(paragraphs)]
        document = Document(doc_id=path.name, path=str(path), file_type="txt")
        document.blocks = blocks
        document.sentences = blocks_to_sentences(blocks, len(documents), min_sentence_words)
        documents.append(document)
        entry = meta_by_file.get(path.name, {})
        metadata.append(
            {
                "file": path.name,
                "title": entry.get("title", path.stem.replace("_", " ").title()),
                "url": entry.get("url"),
                "licence": entry.get("licence"),
            }
        )
    log.info("Loaded %d reference documents from %s", len(documents), reference_dir)
    return documents, metadata


def corpus_fingerprint(model_name: str, texts: List[str]) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for text in texts:
        digest.update(b"\x00" + text.encode("utf-8"))
    return digest.hexdigest()[:16]


def cached_encode(cache_dir: Path, tag: str, model_name: str, texts: List[str], encode) -> np.ndarray:
    """Encode texts, re-using a .npy cache keyed by model + content hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{tag}_{corpus_fingerprint(model_name, texts)}.npy"
    if path.exists():
        vectors = np.load(path)
        if vectors.shape[0] == len(texts):
            return vectors
    vectors = encode(texts)
    np.save(path, vectors)
    return vectors
