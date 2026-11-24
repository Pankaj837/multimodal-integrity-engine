"""Group sentences into overlapping, token-bounded chunks.

Why ~200 tokens? all-MiniLM-L6-v2 silently truncates input beyond 256 word
pieces, so a 400-token chunk would have its second half ignored. Token counts
come from the embedding model's own tokenizer, not from a word count.

Chunks never split a sentence (except a single sentence longer than the
budget, which becomes its own chunk), and consecutive chunks share
``overlap_sentences`` sentences so that a copied passage straddling a chunk
boundary is still fully contained in at least one chunk.

Only *assessed* sentences (eligible prose that is not shared boilerplate) are
chunked, so template text and bibliographies cannot inflate chunk similarity.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

from .models import Chunk, Sentence


def build_chunks(
    sentences: Sequence[Sentence],
    doc_index: int,
    count_tokens: Callable[[List[str]], List[int]],
    max_tokens: int,
    overlap_sentences: int,
) -> List[Chunk]:
    units = [s for s in sentences if s.assessed]
    if not units:
        return []
    token_counts = count_tokens([s.text for s in units])

    chunks: List[Chunk] = []
    i = 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and (j == i or total + token_counts[j] <= max_tokens):
            total += token_counts[j]
            j += 1
        members = units[i:j]
        pages = [s.page for s in members if s.page is not None]
        starts = [s.start for s in members if s.start is not None]
        ends = [s.end for s in members if s.end is not None]
        chunks.append(
            Chunk(
                doc_index=doc_index,
                index=len(chunks),
                sentence_ids=[s.index for s in members],
                text=" ".join(s.text for s in members),
                n_tokens=total,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                start=min(starts) if starts else None,
                end=max(ends) if ends else None,
            )
        )
        if j >= len(units):
            break
        # Step forward, re-using the last `overlap_sentences` sentences, but
        # always advance by at least one sentence to guarantee termination.
        i = max(j - overlap_sentences, i + 1)
    return chunks
