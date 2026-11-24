"""FAISS vector index with owner-aware (self-match-excluding) search.

Each vector has an *owner* - the document it came from. ``search`` returns the
top-k neighbours whose owner differs from the query's owner. Self-matches are
excluded at the document level, not just the vector level: overlapping chunks
of the same document are near-duplicates of each other and would otherwise
dominate every neighbour list.

``flat`` (exact, brute force) is the default and the right choice for a
corpus of this size; ``hnsw`` (approximate graph search) is available to
illustrate the scaling path discussed in the README.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int, kind: str = "flat", hnsw_m: int = 32, ef_search: int = 128):
        if kind == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = ef_search
        else:
            raise ValueError(f"Unknown index kind '{kind}' (use 'flat' or 'hnsw')")
        self.kind = kind
        self.dim = dim
        self.owners = np.zeros(0, dtype=np.int64)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray, owners: np.ndarray) -> None:
        if len(vectors) != len(owners):
            raise ValueError("vectors and owners must have the same length")
        if len(vectors):
            self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            self.owners = np.concatenate([self.owners, np.asarray(owners, dtype=np.int64)])

    def search(
        self, queries: np.ndarray, k: int, query_owners: Optional[np.ndarray] = None
    ) -> List[List[Tuple[int, float]]]:
        """Top-k (vector_id, cosine) per query, skipping vectors with the query's owner.

        To guarantee k results after filtering, we over-fetch by the number of
        vectors the query's own document contributes to the index.
        """
        if self.size == 0 or len(queries) == 0:
            return [[] for _ in range(len(queries))]
        extra = 0
        if query_owners is not None:
            owner_counts = np.bincount(self.owners)
            extra = int(max(owner_counts[o] if o < len(owner_counts) else 0 for o in query_owners))
        k_eff = min(self.size, k + extra)
        sims, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k_eff)

        results = []
        for row, (row_ids, row_sims) in enumerate(zip(ids, sims)):
            hits = []
            for vid, sim in zip(row_ids, row_sims):
                if vid < 0:
                    continue
                if query_owners is not None and self.owners[vid] == query_owners[row]:
                    continue  # self-match exclusion
                hits.append((int(vid), float(min(1.0, max(-1.0, sim)))))
                if len(hits) == k:
                    break
            results.append(hits)
        return results

    def range_search(self, queries: np.ndarray, threshold: float) -> List[List[Tuple[int, float]]]:
        """All neighbours with cosine >= threshold (exact for the flat index)."""
        if self.size == 0 or len(queries) == 0:
            return [[] for _ in range(len(queries))]
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self.kind == "flat":
            # FAISS returns results strictly above the radius; subtract a hair so
            # that ">= threshold" semantics hold for exact ties.
            lims, sims, ids = self.index.range_search(queries, threshold - 1e-6)
            return [
                [(int(ids[j]), float(min(1.0, sims[j]))) for j in range(lims[i], lims[i + 1])]
                for i in range(len(queries))
            ]
        sims, ids = self.index.search(queries, min(self.size, 256))
        return [
            [(int(v), float(min(1.0, s))) for v, s in zip(r_ids, r_sims) if v >= 0 and s >= threshold]
            for r_ids, r_sims in zip(ids, sims)
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        np.save(str(path) + ".owners.npy", self.owners)
