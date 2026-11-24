"""Turn extracted blocks into located sentences.

Sentences are the unit of *evidence*: matches are reported sentence-by-sentence
so that a reviewer reads one precise quote instead of a 200-token chunk. Each
sentence inherits the block's page/paragraph or start/end timestamps.

A rule-based splitter is used instead of an NLP model so that segmentation is
deterministic, dependency-free and fast. It protects common abbreviations
(e.g., Ltd., Dr.), single-letter initials (M. E. Porter) and decimals (4.2%).
"""

from __future__ import annotations

import re
from typing import Iterable, List

from .models import Block, Sentence, SentenceKind
from .normalize import collapse_whitespace

_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "co", "corp",
    "no", "nos", "fig", "al", "approx", "est", "dept", "rs", "st", "jr", "sr", "pvt", "govt",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "vol", "pp", "ed", "eds", "cf", "u.s", "u.k", "a.m", "p.m",
}
# A sentence boundary candidate: terminal punctuation (+ optional closing
# quote/bracket), whitespace, then something that can start a sentence.
_BOUNDARY = re.compile(r"([.!?][\"')\]]*)\s+(?=[\"'(\[]?[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
_REFERENCE_HEADING = re.compile(
    r"^(references|reference list|bibliography|works cited|sources|citations)\s*:?$", re.I
)
# Bibliography-entry patterns, applied only to SHORT blocks (see is_citation):
#   "Porter, M. E. (1980). Competitive Strategy..."   (surname + initials)
#   "Kestrel Foods Ltd. (2026). Case Data Pack..."    (author/org, then "(year).")
#   any block with a DOI, ISBN, URL or "Retrieved from".
_CITATION_PATTERNS = [
    re.compile(r"^(\[\d+\]\s*)?[A-Z][\w'\-]+,\s(?:[A-Z]\.\s?)+"),
    re.compile(r"^(\[\d+\]\s*)?[A-Z][^()]{0,100}\(\d{4}[a-z]?\)\.\s+\S"),
    re.compile(r"\bdoi:|\bdoi\.org/|\bISBN\b|https?://|\bRetrieved from\b", re.I),
]
_CITATION_MAX_WORDS = 60


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def split_sentences(text: str) -> List[str]:
    """Split a paragraph into sentences (see module docstring for the rules)."""
    text = collapse_whitespace(text)
    if not text:
        return []
    sentences, start = [], 0
    for match in _BOUNDARY.finditer(text):
        candidate = text[start:match.end(1)]
        if match.group(1).startswith("."):
            last_token = candidate.split()[-1].rstrip(".!?\"')]").lower().lstrip("(\"'[")
            if (
                last_token in _ABBREVIATIONS
                or re.fullmatch(r"[a-z]", last_token)  # initials: "M. E. Porter"
                or re.fullmatch(r"(?:[a-z]\.)+[a-z]", last_token)  # "u.s", "e.g"
            ):
                continue
        sentences.append(candidate.strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def is_citation(text: str) -> bool:
    """Heuristic bibliography-entry detector for blocks outside a References
    section. Long blocks are never treated as citations so that ordinary
    prose mentioning "(2025)." or a URL is not silently excluded."""
    if count_words(text) > _CITATION_MAX_WORDS:
        return False
    return any(p.search(text) for p in _CITATION_PATTERNS)


def blocks_to_sentences(
    blocks: Iterable[Block], doc_index: int, min_sentence_words: int
) -> List[Sentence]:
    """Segment blocks into sentences and decide which are eligible evidence.

    * Blocks under a "References"/"Bibliography" heading, or that look like a
      bibliography entry, become one CITATION sentence each (never evidence).
    * Headings and table rows are kept for context but are not evidence.
    * Prose sentences shorter than ``min_sentence_words`` are not evidence:
      phrases such as "Thank you." match everything and prove nothing.
    """
    sentences: List[Sentence] = []
    in_references = False

    for block in blocks:
        text = collapse_whitespace(block.text)
        if not text:
            continue
        kind = block.kind
        if kind == SentenceKind.HEADING:
            in_references = bool(_REFERENCE_HEADING.match(text))
        elif in_references or is_citation(text):
            kind = SentenceKind.CITATION

        if kind == SentenceKind.TEXT and not block.is_sentence:
            pieces = split_sentences(text)
        else:
            pieces = [text]

        for piece in pieces:
            n_words = count_words(piece)
            sentences.append(
                Sentence(
                    doc_index=doc_index,
                    index=len(sentences),
                    text=piece,
                    n_words=n_words,
                    kind=kind,
                    page=block.page,
                    paragraph=block.paragraph,
                    start=block.start,
                    end=block.end,
                    eligible=kind == SentenceKind.TEXT and n_words >= min_sentence_words,
                )
            )
    return sentences
