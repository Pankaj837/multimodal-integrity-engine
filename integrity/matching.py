"""Semantic passage matching between document collections.

Two collections are compared: the submissions against themselves (peer and
cross-modal matching, with same-document exclusion) and the submissions
against the reference corpus (external matching).

Algorithm (coarse-to-fine, all similarities are cosine on unit vectors):

1. Direct sentence matching. Every assessed query sentence retrieves its
   top-k nearest sentences from the target index; a pair counts when
   cosine >= ``sentence_threshold`` (0.85). Retrieved candidates between
   ``lexical_rescue_cosine`` (0.70) and that threshold also count when their
   wording is near-identical (normalised Levenshtein >= 0.70): small edits,
   a dropped acronym or an ASR spelling ("fulfillment") lower the cosine of
   an obvious copy, but not its character-level overlap. Semantic retrieval
   proposes, lexical similarity confirms - a hybrid used by production
   plagiarism systems.
2. Context-supported matching. Every query chunk retrieves its nearest target
   chunks; when two ~200-token chunks are strongly related (cosine >=
   ``context_chunk_threshold``) the sentences inside them are aligned with a
   relaxed threshold (``context_sentence_threshold``). Heavy paraphrase lowers
   individual sentence scores, but the surrounding argument still lines up,
   so requiring BOTH a strong chunk match and a reasonable sentence match
   recovers paraphrase without flagging every on-topic sentence.
3. Passage assembly. Matched sentences that are adjacent in both documents
   are merged into passages, which are what reviewers read.
4. Evidence-strength filter (``is_reportable``). A lone sentence that is only
   semantically similar is dropped unless its similarity is very high or its
   wording also overlaps; same-topic answers share such sentences by chance.

Calibration (see EVALUATION.md): on the dataset, the true paraphrase pair has
a best chunk cosine of 0.89 while the most similar *independent* pair reaches
0.81, whereas single-sentence cosines overlap (0.89 vs 0.84). The chunk-level
gate therefore carries most of the paraphrase decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz.distance import Levenshtein

from .config import MatchingConfig
from .indexer import VectorIndex
from .models import Document

SentenceKey = Tuple[int, int]  # (doc_index, sentence_index)
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class Collection:
    """Assessed sentences and chunks of a set of documents, embedded and indexed."""

    documents: List[Document]
    sentence_keys: List[SentenceKey]
    sentence_vectors: np.ndarray
    chunk_keys: List[Tuple[int, int]]  # (doc_index, chunk_index)
    chunk_vectors: np.ndarray
    sentence_index: VectorIndex = field(init=False)
    chunk_index: VectorIndex = field(init=False)
    row_of: Dict[SentenceKey, int] = field(init=False)
    chunk_rows_of: Dict[SentenceKey, List[int]] = field(init=False)

    def __post_init__(self) -> None:
        dim = self.sentence_vectors.shape[1] if self.sentence_vectors.size else self.chunk_vectors.shape[1]
        self.sentence_owners = np.array([d for d, _ in self.sentence_keys], dtype=np.int64)
        self.chunk_owners = np.array([d for d, _ in self.chunk_keys], dtype=np.int64)
        self.sentence_index = VectorIndex(dim)
        self.sentence_index.add(self.sentence_vectors, self.sentence_owners)
        self.chunk_index = VectorIndex(dim)
        self.chunk_index.add(self.chunk_vectors, self.chunk_owners)
        self.row_of = {key: row for row, key in enumerate(self.sentence_keys)}
        self.chunk_rows_of = {}
        for row, (doc_index, chunk_index) in enumerate(self.chunk_keys):
            for sentence_index in self.documents[doc_index].chunks[chunk_index].sentence_ids:
                self.chunk_rows_of.setdefault((doc_index, sentence_index), []).append(row)

    def sentence(self, row: int):
        doc_index, sentence_index = self.sentence_keys[row]
        return self.documents[doc_index].sentences[sentence_index]


@dataclass
class SentencePair:
    query_row: int
    target_row: int  # primary aligned target sentence
    similarity: float  # cosine between the query sentence and the primary target
    method: str  # "direct" | "lexical" | "context"
    target_rows: List[int] = field(default_factory=list)  # 1 row, or 2 when sentences were merged


@dataclass
class Passage:
    query_doc: int
    target_doc: int
    pairs: List[SentencePair]
    query_text: str
    matched_text: str
    similarity: float  # word-weighted mean cosine of the sentence pairs
    chunk_similarity: Optional[float]
    jaccard: float
    levenshtein: float
    match_type: str  # verbatim | near_verbatim | paraphrase
    query_sentences: list
    matched_sentences: list
    query_words: int = 0
    alignment: List[Dict] = field(default_factory=list)  # per-sentence evidence for reviewers


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def lexical_scores(a: str, b: str) -> Tuple[float, float]:
    """(Jaccard over word sets, normalised Levenshtein similarity over word strings).

    Both ignore case and punctuation so that an ASR transcript ("4.2 percent")
    is compared on wording, not formatting.
    """
    ta, tb = _tokens(a), _tokens(b)
    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    levenshtein = Levenshtein.normalized_similarity(" ".join(ta), " ".join(tb))
    return round(jaccard, 3), round(levenshtein, 3)


def classify_match(levenshtein: float, cfg: MatchingConfig) -> str:
    if levenshtein >= cfg.verbatim_levenshtein:
        return "verbatim"
    if levenshtein >= cfg.near_verbatim_levenshtein:
        return "near_verbatim"
    return "paraphrase"


def find_sentence_pairs(
    query: Collection, target: Collection, cfg: MatchingConfig, same_corpus: bool
) -> Dict[Tuple[int, int], SentencePair]:
    """Best pair per (query sentence row, target document). Self-matches excluded.

    When several sentences of the same target document qualify, the one with
    the highest *combined* evidence (cosine + normalised Levenshtein) is kept.
    Ranking by cosine alone can pick a different sentence that merely says
    something similar (e.g. another definition in the same article) instead of
    the sentence that was actually copied, which also shares the wording.
    """
    best: Dict[Tuple[int, int], SentencePair] = {}
    rank: Dict[Tuple[int, int], float] = {}

    def offer(q_row: int, t_rows: List[int], sim: float, method: str, levenshtein: float) -> None:
        key = (q_row, int(target.sentence_owners[t_rows[0]]))
        score = sim + levenshtein
        if key not in best or score > rank[key]:
            best[key] = SentencePair(q_row, t_rows[0], sim, method, list(t_rows))
            rank[key] = score

    # 1. Direct sentence matching, with a lexical rescue for near-verbatim
    #    copies whose cosine is dragged down by small edits or ASR errors.
    owners = query.sentence_owners if same_corpus else None
    floor = min(cfg.sentence_threshold, cfg.lexical_rescue_cosine)
    for q_row, hits in enumerate(target.sentence_index.search(query.sentence_vectors, cfg.top_k, owners)):
        for t_row, sim in hits:
            if sim < floor:
                break  # hits are sorted by similarity
            levenshtein, rows = best_lexical_window(query, target, q_row, t_row)
            if sim >= cfg.sentence_threshold:
                offer(q_row, rows, sim, "direct", levenshtein)
            elif levenshtein >= cfg.near_verbatim_levenshtein:
                offer(q_row, rows, sim, "lexical", levenshtein)

    # 2. Context-supported matching inside strongly related chunk pairs.
    owners = query.chunk_owners if same_corpus else None
    chunk_hits = target.chunk_index.search(query.chunk_vectors, cfg.top_k, owners)
    for q_chunk_row, hits in enumerate(chunk_hits):
        q_doc, q_chunk = query.chunk_keys[q_chunk_row]
        q_rows = [query.row_of[(q_doc, s)] for s in query.documents[q_doc].chunks[q_chunk].sentence_ids]
        for t_chunk_row, chunk_sim in hits:
            if chunk_sim < cfg.context_chunk_threshold:
                break
            t_doc, t_chunk = target.chunk_keys[t_chunk_row]
            t_rows = [target.row_of[(t_doc, s)] for s in target.documents[t_doc].chunks[t_chunk].sentence_ids]
            sims = query.sentence_vectors[q_rows] @ target.sentence_vectors[t_rows].T
            for i, q_row in enumerate(q_rows):
                j = int(np.argmax(sims[i]))
                if sims[i, j] >= cfg.context_sentence_threshold and (q_row, t_doc) not in best:
                    levenshtein, rows = best_lexical_window(query, target, q_row, t_rows[j])
                    offer(q_row, rows, float(sims[i, j]), "context", levenshtein)
    return best


def best_lexical_window(query: Collection, target: Collection, q_row: int, t_row: int) -> Tuple[float, List[int]]:
    """Lexical alignment allowing one query sentence to cover TWO consecutive
    target sentences (and vice versa via the symmetric run).

    Copiers often merge or split sentences, and ASR segments speech
    differently from the written source; strict 1:1 sentence alignment would
    then miss an obvious copy. Returns (best Levenshtein, aligned target rows).
    """
    q_text = query.sentence(q_row).text
    _, best_lev = lexical_scores(q_text, target.sentence(t_row).text)
    best_rows = [t_row]
    owner = target.sentence_owners[t_row]
    for window in ((t_row - 1, t_row), (t_row, t_row + 1)):
        if window[0] < 0 or window[1] >= len(target.sentence_keys):
            continue
        if target.sentence_owners[window[0]] != owner or target.sentence_owners[window[1]] != owner:
            continue
        merged = target.sentence(window[0]).text + " " + target.sentence(window[1]).text
        _, lev = lexical_scores(q_text, merged)
        if lev > best_lev + 0.05:  # require a clear gain before claiming a merge
            best_lev, best_rows = lev, list(window)
    return best_lev, best_rows


def _chunk_similarity(query: Collection, target: Collection, pair: SentencePair) -> Optional[float]:
    q_rows = query.chunk_rows_of.get(query.sentence_keys[pair.query_row], [])
    t_rows = target.chunk_rows_of.get(target.sentence_keys[pair.target_row], [])
    if not q_rows or not t_rows:
        return None
    sims = query.chunk_vectors[q_rows] @ target.chunk_vectors[t_rows].T
    return float(min(1.0, sims.max()))


def assemble_passages(
    query: Collection,
    target: Collection,
    pairs: Dict[Tuple[int, int], SentencePair],
    cfg: MatchingConfig,
) -> Dict[int, List[Passage]]:
    """Merge adjacent sentence pairs into passages, grouped by query document."""
    by_doc_pair: Dict[Tuple[int, int], List[SentencePair]] = {}
    for (q_row, t_doc), pair in pairs.items():
        q_doc = query.sentence_keys[q_row][0]
        by_doc_pair.setdefault((q_doc, t_doc), []).append(pair)

    ordinal = _assessed_ordinals(query)
    t_ordinal = _assessed_ordinals(target)
    passages: Dict[int, List[Passage]] = {}
    for (q_doc, t_doc), doc_pairs in by_doc_pair.items():
        doc_pairs.sort(key=lambda p: ordinal[p.query_row])
        groups: List[List[SentencePair]] = [[doc_pairs[0]]]
        for pair in doc_pairs[1:]:
            previous = groups[-1][-1]
            q_gap = ordinal[pair.query_row] - ordinal[previous.query_row] - 1
            t_jump = abs(t_ordinal[pair.target_row] - t_ordinal[previous.target_row])
            if q_gap <= cfg.passage_gap and t_jump <= cfg.passage_gap + 2:
                groups[-1].append(pair)
            else:
                groups.append([pair])
        for group in groups:
            passage = _make_passage(query, target, q_doc, t_doc, group, cfg)
            if is_reportable(passage, cfg):
                passages.setdefault(q_doc, []).append(passage)

    for doc_passages in passages.values():
        doc_passages.sort(key=lambda p: (-p.similarity * p.query_words, -p.similarity))
    return passages


def is_reportable(passage: Passage, cfg: MatchingConfig) -> bool:
    """Evidence-strength rule.

    One isolated sentence that is only *semantically* close to another is weak
    evidence: two independent answers to the same case naturally contain
    similar topic sentences ("Kestrel's gross margin declined because ...").
    Such a match is reported only if the wording also overlaps (near-verbatim)
    or the semantic similarity is very high. Multi-sentence passages - a run
    of aligned sentences - are always reported.
    """
    if len(passage.pairs) >= 2:
        return True
    if passage.levenshtein >= cfg.near_verbatim_levenshtein:
        return True
    return passage.similarity >= cfg.single_sentence_min_similarity


def _assessed_ordinals(collection: Collection) -> Dict[int, int]:
    """Row -> position among the document's assessed sentences (for gap tests)."""
    ordinals: Dict[int, int] = {}
    counters: Dict[int, int] = {}
    for row, (doc_index, _) in enumerate(collection.sentence_keys):
        ordinals[row] = counters.get(doc_index, 0)
        counters[doc_index] = ordinals[row] + 1
    return ordinals


def _make_passage(
    query: Collection,
    target: Collection,
    q_doc: int,
    t_doc: int,
    group: Sequence[SentencePair],
    cfg: MatchingConfig,
) -> Passage:
    q_sentences = [query.sentence(p.query_row) for p in group]
    t_rows_unique = sorted({row for p in group for row in (p.target_rows or [p.target_row])})
    t_sentences = [target.sentence(r) for r in t_rows_unique]
    query_text = " ".join(s.text for s in q_sentences)
    matched_text = " ".join(s.text for s in t_sentences)
    weights = np.array([s.n_words for s in q_sentences], dtype=float)
    sims = np.array([p.similarity for p in group])
    similarity = float((weights * sims).sum() / weights.sum()) if weights.sum() else float(sims.mean())
    chunk_sims = [c for c in (_chunk_similarity(query, target, p) for p in group) if c is not None]
    jaccard, levenshtein = lexical_scores(query_text, matched_text)
    alignment = [
        {
            "query_sentence": query.sentence(p.query_row).text,
            "matched_sentence": " ".join(target.sentence(r).text for r in (p.target_rows or [p.target_row])),
            "similarity": round(p.similarity, 3),
            "method": p.method,
        }
        for p in group
    ]
    return Passage(
        query_doc=q_doc,
        target_doc=t_doc,
        pairs=list(group),
        query_text=query_text,
        matched_text=matched_text,
        similarity=round(similarity, 3),
        chunk_similarity=round(max(chunk_sims), 3) if chunk_sims else None,
        jaccard=jaccard,
        levenshtein=levenshtein,
        match_type=classify_match(levenshtein, cfg),
        query_sentences=q_sentences,
        matched_sentences=t_sentences,
        query_words=int(weights.sum()),
        alignment=alignment,
    )
