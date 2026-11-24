"""Shared-template (boilerplate) detection - the "judgment test" scenario.

Case-competition submissions legitimately repeat the organiser's prompt,
question wording and standard citations. Copying between *two* students is
suspicious; text shared by *many* independent submissions is almost always
institutional boilerplate.

Rule: an eligible sentence is boilerplate when near-identical sentences
(cosine >= ``similarity_threshold``) occur in at least

    max(min_documents, ceil(min_document_fraction * N))

distinct documents (its own included), where N is the number of successfully
processed submissions. The fraction term keeps the rule meaningful at scale:
in a 10,000-submission challenge, three students sharing a paragraph is a
collusion signal, not a template. Optionally, a known template file (e.g. the
official prompt) can be supplied and any sentence matching it is discounted.

Boilerplate sentences are excluded from evidence AND from the denominator of
the similarity percentages, and are listed separately in each report so the
decision stays transparent.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import BoilerplateConfig
from .indexer import VectorIndex
from .models import Document

log = logging.getLogger(__name__)

SentenceKey = Tuple[int, int]  # (doc_index, sentence_index)


def required_documents(cfg: BoilerplateConfig, n_documents: int) -> int:
    return max(cfg.min_documents, math.ceil(cfg.min_document_fraction * n_documents))


def mark_boilerplate(
    documents: List[Document],
    keys: List[SentenceKey],
    embeddings: np.ndarray,
    cfg: BoilerplateConfig,
    template_embeddings: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    """Flag boilerplate sentences in place. Returns summary statistics."""
    stats = {"threshold_documents": 0, "sentences_flagged": 0, "template_matches": 0}
    if not cfg.enabled or len(keys) == 0:
        return stats

    n_docs = sum(1 for d in documents if d.ok)
    min_docs = required_documents(cfg, n_docs)
    stats["threshold_documents"] = min_docs
    owners = np.array([doc for doc, _ in keys], dtype=np.int64)

    index = VectorIndex(embeddings.shape[1])
    index.add(embeddings, owners)
    neighbours = index.range_search(embeddings, cfg.similarity_threshold)

    for row, hits in enumerate(neighbours):
        sharing_docs = {int(owners[j]) for j, _ in hits}
        sharing_docs.add(int(owners[row]))
        if len(sharing_docs) >= min_docs:
            doc_index, sent_index = keys[row]
            sentence = documents[doc_index].sentences[sent_index]
            sentence.boilerplate = True
            sentence.boilerplate_docs = len(sharing_docs)
            stats["sentences_flagged"] += 1

    if template_embeddings is not None and len(template_embeddings):
        template_index = VectorIndex(embeddings.shape[1])
        template_index.add(template_embeddings, np.zeros(len(template_embeddings), dtype=np.int64))
        best = template_index.search(embeddings, 1)
        for row, hits in enumerate(best):
            if hits and hits[0][1] >= cfg.similarity_threshold:
                doc_index, sent_index = keys[row]
                sentence = documents[doc_index].sentences[sent_index]
                if not sentence.boilerplate:
                    stats["sentences_flagged"] += 1
                sentence.boilerplate = True
                sentence.boilerplate_docs = max(sentence.boilerplate_docs, 1)
                stats["template_matches"] += 1

    log.info(
        "Boilerplate: %d sentence(s) shared by >= %d documents were discounted",
        stats["sentences_flagged"],
        min_docs,
    )
    return stats
