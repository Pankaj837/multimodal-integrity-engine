"""Vector index self-exclusion, boilerplate rule, lexical metrics, scoring."""

import numpy as np

from integrity.boilerplate import mark_boilerplate, required_documents
from integrity.config import BoilerplateConfig, MatchingConfig, RiskConfig
from integrity.indexer import VectorIndex
from integrity.matching import classify_match, lexical_scores
from integrity.models import Block, Document
from integrity.scoring import coverage_percentage, risk_level
from integrity.segmentation import blocks_to_sentences


def _unit(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_inner_product_on_unit_vectors_is_cosine():
    vectors = _unit([[1, 0], [1, 1], [0, 1]])
    index = VectorIndex(2)
    index.add(vectors, np.array([0, 1, 2]))
    hits = index.search(vectors[:1], k=3)[0]
    assert [h[0] for h in hits] == [0, 1, 2]
    assert np.isclose(hits[1][1], np.cos(np.pi / 4), atol=1e-6)


def test_search_excludes_every_vector_of_the_query_document():
    # Document 0 owns three near-identical vectors (overlapping chunks); without
    # document-level exclusion they would fill the whole top-3.
    vectors = _unit([[1, 0.00], [1, 0.01], [1, 0.02], [0.8, 0.6], [0, 1]])
    owners = np.array([0, 0, 0, 1, 2])
    index = VectorIndex(2)
    index.add(vectors, owners)
    hits = index.search(vectors[:1], k=3, query_owners=np.array([0]))[0]
    assert [owners[h[0]] for h in hits] == [1, 2]  # only other documents, best first
    assert all(owners[h[0]] != 0 for h in hits)


def test_range_search_includes_exact_threshold():
    vectors = _unit([[1, 0], [0.9, np.sqrt(1 - 0.81)]])
    index = VectorIndex(2)
    index.add(vectors, np.array([0, 1]))
    hits = index.range_search(vectors[:1], 0.9)[0]
    assert {h[0] for h in hits} == {0, 1}


def test_hnsw_index_returns_the_same_nearest_neighbour():
    rng = np.random.default_rng(0)
    vectors = _unit(rng.normal(size=(200, 16)))
    flat, hnsw = VectorIndex(16, "flat"), VectorIndex(16, "hnsw")
    owners = np.arange(200)
    flat.add(vectors, owners)
    hnsw.add(vectors, owners)
    query = vectors[:5]
    assert [h[0][0] for h in flat.search(query, 1)] == [h[0][0] for h in hnsw.search(query, 1)]


def _docs_with_shared_sentence(n_docs, shared_in):
    documents = []
    for d in range(n_docs):
        text = f"Unique analysis number {d} about a completely different matter here."
        if d in shared_in:
            text = "This is the official case prompt that every team copies verbatim. " + text
        document = Document(doc_id=f"d{d}.txt", path="", file_type="txt")
        document.sentences = blocks_to_sentences([Block(text)], d, 6)
        documents.append(document)
    return documents


def _fake_embeddings(documents):
    keys, rows = [], []
    for d, document in enumerate(documents):
        for s in document.sentences:
            keys.append((d, s.index))
            vec = np.zeros(32, dtype=np.float32)
            vec[0 if "official case prompt" in s.text else 1 + d] = 1.0
            rows.append(vec)
    return keys, np.stack(rows)


def test_boilerplate_needs_min_documents():
    cfg = BoilerplateConfig(min_documents=3, min_document_fraction=0.0)
    two = _docs_with_shared_sentence(6, {0, 1})
    mark_boilerplate(two, *_fake_embeddings(two), cfg)
    assert not any(s.boilerplate for d in two for s in d.sentences)  # 2 docs = possible collusion

    five = _docs_with_shared_sentence(6, {0, 1, 2, 3, 4})
    mark_boilerplate(five, *_fake_embeddings(five), cfg)
    flagged = [s for d in five for s in d.sentences if s.boilerplate]
    assert len(flagged) == 5 and all(s.boilerplate_docs == 5 for s in flagged)
    assert not any(s.boilerplate for s in five[5].sentences)


def test_boilerplate_threshold_scales_with_corpus():
    cfg = BoilerplateConfig(min_documents=3, min_document_fraction=0.10)
    assert required_documents(cfg, 15) == 3
    assert required_documents(cfg, 10_000) == 1_000


def test_lexical_scores_and_match_types():
    cfg = MatchingConfig()
    j, lev = lexical_scores("Margins fell 4.2% in Q3.", "margins fell 4.2 % in q3")
    assert j == 1.0 and lev == 1.0
    j, lev = lexical_scores("The firm bought cooking oil at market rates.",
                            "The company purchased edible oil on the spot market.")
    assert j < 0.5 and lev < 0.7
    assert classify_match(0.95, cfg) == "verbatim"
    assert classify_match(0.75, cfg) == "near_verbatim"
    assert classify_match(0.30, cfg) == "paraphrase"


def test_merge_window_aligns_one_sentence_to_two_source_sentences():
    """A copier merged two source sentences into one (failure F4 in EVALUATION.md)."""
    from integrity.matching import Collection, best_lexical_window

    def collection(texts):
        document = Document(doc_id="d", path="", file_type="txt")
        document.sentences = blocks_to_sentences([Block(" ".join(texts))], 0, 3)
        keys = [(0, s.index) for s in document.sentences]
        return Collection([document], keys, _unit(np.eye(len(keys), 8)), [], np.zeros((0, 8), np.float32))

    source = collection([
        "Gross margin is the difference between revenue and cost of goods sold, divided by revenue.",
        "Gross margin is expressed as a percentage.",
        "It is a kind of profit margin used by analysts.",
    ])
    query = collection(["Gross margin is the difference between revenue and the cost of goods sold, "
                        "divided by revenue, and it is always expressed as a percentage."])
    levenshtein, rows = best_lexical_window(query, source, 0, 0)
    assert rows == [0, 1] and levenshtein >= 0.75


def test_coverage_is_word_weighted_and_handles_empty_documents():
    document = Document(doc_id="x", path="", file_type="txt")
    document.sentences = blocks_to_sentences(
        [Block("One two three four five six. Seven eight nine ten eleven twelve thirteen fourteen.")], 0, 6
    )
    assert coverage_percentage(document, [0]) == round(100 * 6 / 14, 1)
    empty = Document(doc_id="y", path="", file_type="txt")
    assert coverage_percentage(empty, []) == 0.0


def test_risk_levels():
    cfg = RiskConfig()
    assert risk_level(35.0, 0.0, 5.0, "normal", cfg) == "HIGH"
    assert risk_level(12.0, 0.0, 5.0, "normal", cfg) == "MEDIUM"
    assert risk_level(0.0, 0.0, 90.0, "normal", cfg) == "HIGH"
    assert risk_level(0.0, 0.0, 90.0, "low", cfg) == "MEDIUM"  # unreliable AI signal is capped
    assert risk_level(0.0, 0.0, None, "n/a", cfg) == "LOW"
