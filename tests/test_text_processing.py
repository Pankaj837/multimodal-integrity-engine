"""Normalisation, sentence segmentation, citation handling and chunking."""

from integrity.chunker import build_chunks
from integrity.extractors.base import ParagraphCounter, classify_short_line, lines_to_blocks
from integrity.models import Block, SentenceKind
from integrity.normalize import normalize_text, remove_headers_footers
from integrity.segmentation import blocks_to_sentences, count_words, is_citation, split_sentences


def test_normalize_repairs_ligatures_hyphenation_and_controls():
    text = normalize_text("The ﬁle is compet-\nitive  here­.\x7f “Quote”")
    assert text == 'The file is competitive here. "Quote"'


def test_header_footer_removal_masks_page_numbers():
    pages = [
        ["ACME CASE | TEAM X", "Body text one.", "Page 1 of 3"],
        ["ACME CASE | TEAM X", "Body text two.", "Page 2 of 3"],
        ["ACME CASE | TEAM X", "Body text three.", "3"],
    ]
    cleaned, removed = remove_headers_footers(pages)
    assert cleaned == [["Body text one."], ["Body text two."], ["Body text three."]]
    assert removed == 6


def test_single_page_keeps_content_lines():
    cleaned, removed = remove_headers_footers([["Title line", "Only page body."]])
    assert cleaned == [["Title line", "Only page body."]] and removed == 0


def test_sentence_splitter_protects_abbreviations_initials_and_decimals():
    text = ("Kestrel Foods Ltd. grew 4.2% in Q3. Prices rose, e.g. edible oil. M. E. Porter (1980) disagrees! "
            "Did the U.S. market matter? Yes.")
    assert split_sentences(text) == [
        "Kestrel Foods Ltd. grew 4.2% in Q3.",
        "Prices rose, e.g. edible oil.",
        "M. E. Porter (1980) disagrees!",
        "Did the U.S. market matter?",
        "Yes.",
    ]


def test_citation_detection_is_conservative():
    assert is_citation("Porter, M. E. (1980). Competitive Strategy. New York: Free Press.")
    assert is_citation("Kestrel Foods Ltd. (2026). Case Competition Data Pack, Round 1.")
    long_prose = ("Sales peaked in the festive season of last year and then dropped sharply across all of the "
                  "regions we studied in detail and in every channel we looked at (2025). After that it fell.")
    assert not is_citation(long_prose)


def test_reference_section_sentences_are_citations_not_evidence():
    blocks = [
        Block("Our analysis shows that procurement drives the margin decline in every region.", SentenceKind.TEXT),
        Block("References", SentenceKind.HEADING),
        Block("Kotler, P., & Keller, K. L. (2016). Marketing Management. Pearson.", SentenceKind.TEXT),
    ]
    sentences = blocks_to_sentences(blocks, 0, 6)
    assert [s.kind for s in sentences] == [SentenceKind.TEXT, SentenceKind.HEADING, SentenceKind.CITATION]
    assert [s.eligible for s in sentences] == [True, False, False]


def test_short_sentences_are_not_evidence():
    sentences = blocks_to_sentences([Block("Thank you. This longer sentence certainly counts as evidence here.")], 0, 6)
    assert [s.eligible for s in sentences] == [False, True]


def test_full_width_line_with_few_hyphenated_words_is_not_a_heading():
    lines = [
        "Prices of edible oil and potatoes rose sharply, modern-trade retailers asked",
        "demanded deeper discounts, and two venture-funded direct-to-consumer brands",
        "entered its core categories over the past three years of the review.",
        "Next heading",
    ]
    blocks = lines_to_blocks(lines, 1, ParagraphCounter())
    assert blocks[0].kind == SentenceKind.TEXT and "venture-funded" in blocks[0].text
    assert blocks[-1].kind == SentenceKind.HEADING and blocks[-1].text == "Next heading"


def test_table_rows_are_classified_as_table():
    assert classify_short_line("2024 458 33% 111") == SentenceKind.TABLE
    assert classify_short_line("Financial impact") == SentenceKind.HEADING
    assert classify_short_line("This is a sentence.") is None


def test_chunker_respects_budget_overlap_and_order():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(12))
    sentences = blocks_to_sentences([Block(text, page=1)], 0, 6)
    chunks = build_chunks(sentences, 0, lambda texts: [count_words(t) for t in texts], 21, 1)
    assert all(c.n_tokens <= 21 for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.sentence_ids[-1] == current.sentence_ids[0]  # one-sentence overlap
    covered = {i for c in chunks for i in c.sentence_ids}
    assert covered == {s.index for s in sentences}


def test_chunker_skips_boilerplate_sentences():
    sentences = blocks_to_sentences([Block("Shared template sentence that appears everywhere in the corpus. "
                                           "Original analysis sentence written by this team only.")], 0, 6)
    sentences[0].boilerplate = True
    chunks = build_chunks(sentences, 0, lambda texts: [count_words(t) for t in texts], 200, 1)
    assert len(chunks) == 1 and chunks[0].sentence_ids == [1]
