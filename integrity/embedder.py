"""Dense sentence/chunk embeddings with Sentence-Transformers.

Vectors are L2-normalised at encode time. For unit vectors the inner product
equals cosine similarity (a.b = |a||b|cos(theta) = cos(theta)), which lets the
FAISS index use a plain inner-product search to return exact cosine scores.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

log = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """'auto' -> 'cuda' when a CUDA build of PyTorch sees a GPU, else 'cpu'."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model '%s' on %s", model_name, device)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dim = int(self.model.encode(["probe"], convert_to_numpy=True).shape[1])
        self.max_seq_length = int(self.model.max_seq_length)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim) with unit-length rows."""
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        """Word-piece counts from the model's own tokenizer (no special tokens)."""
        if not texts:
            return []
        encoded = self.model.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]
