"""End-to-end: CLI, report schema, self-match exclusion, batch robustness."""

import json

import pytest
from conftest import PARAGRAPH, make_docx, make_pdf

import run_integrity_check

SCHEMA_KEYS = ["document_id", "total_integrity_risk", "scores", "peer_matches", "multimodal_matches", "explanations"]
SCORE_KEYS = ["peer_similarity_percentage", "external_similarity_percentage", "ai_generated_probability"]
MATCH_KEYS = {"matched_file", "similarity_score", "query_passage", "matched_passage", "location"}

ORIGINAL = (
    "Our team believes the factory consolidation will save two margin points within eighteen months. "
    "Utilisation at the newer plant should climb above eighty percent after the second shift moves. "
    "Van sales in three districts would test whether small grocery stores can be served profitably."
)


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    folder = tmp_path_factory.mktemp("subs")
    make_pdf(folder / "source.pdf", [PARAGRAPH])
    make_docx(folder / "copy.docx", [PARAGRAPH])  # cross-format exact copy
    make_docx(folder / "original ü.docx", [ORIGINAL])  # unicode + space in name
    (folder / "nested").mkdir()
    (folder / "nested" / "empty.pdf").write_bytes(b"")
    (folder / "dataset_manifest.json").write_text("{}", encoding="utf-8")  # must be ignored
    out = tmp_path_factory.mktemp("reports")
    code = run_integrity_check.main(["--input_dir", str(folder), "--output_dir", str(out), "--no_ai",
                                     "--log_level", "WARNING"])
    return code, out


def test_cli_finishes_and_writes_every_report(batch):
    code, out = batch
    assert code == 0
    names = {p.name for p in out.iterdir()}
    for stem in ("source.pdf", "copy.docx", "original_ü.docx", "nested_empty.pdf"):
        assert f"{stem}.json" in names and f"{stem}.md" in names, stem
    assert {"summary.json", "summary.md"} <= names
    assert not any("dataset_manifest" in n for n in names)


def test_report_follows_brief_schema_with_ascii_keys(batch):
    _, out = batch
    report = json.loads((out / "copy.docx.json").read_text(encoding="utf-8"))
    assert list(report)[:6] == SCHEMA_KEYS
    assert list(report["scores"]) == SCORE_KEYS
    assert all(key.isascii() for key in json.dumps(report, ensure_ascii=True).split('"') if key.endswith("_file"))
    match = report["peer_matches"][0]
    assert MATCH_KEYS <= set(match) and "matched_file" in match  # no U+FB01 ligature in keys
    assert set(match["location"]) >= {"query_page", "matched_page"}


def test_exact_copy_is_found_and_self_matches_are_excluded(batch):
    _, out = batch
    report = json.loads((out / "copy.docx.json").read_text(encoding="utf-8"))
    assert report["scores"]["peer_similarity_percentage"] == 100.0
    assert report["total_integrity_risk"] == "HIGH"
    assert {m["matched_file"] for m in report["peer_matches"]} == {"source.pdf"}  # never itself
    assert report["peer_matches"][0]["match_type"] == "verbatim"


def test_original_document_is_clean(batch):
    _, out = batch
    report = json.loads((out / "original_ü.docx.json").read_text(encoding="utf-8"))
    assert report["scores"]["peer_similarity_percentage"] == 0.0
    assert report["peer_matches"] == [] and report["total_integrity_risk"] == "LOW"


def test_empty_file_in_subfolder_is_flagged_not_fatal(batch):
    _, out = batch
    report = json.loads((out / "nested_empty.pdf.json").read_text(encoding="utf-8"))
    assert report["total_integrity_risk"] == "UNPROCESSABLE"
    assert report["processing"]["status"] == "EMPTY_FILE"
    assert report["scores"]["peer_similarity_percentage"] is None


def test_missing_input_folder_exits_with_code_2(tmp_path):
    assert run_integrity_check.main(["--input_dir", str(tmp_path / "nope"), "--output_dir", str(tmp_path)]) == 2


def test_empty_input_folder_still_writes_summary(tmp_path):
    folder, out = tmp_path / "in", tmp_path / "out"
    folder.mkdir()
    assert run_integrity_check.main(["--input_dir", str(folder), "--output_dir", str(out), "--no_ai",
                                     "--log_level", "WARNING"]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["files_total"] == 0


def test_unknown_config_key_is_rejected(tmp_path):
    from integrity.config import load_config

    bad = tmp_path / "bad.yaml"
    bad.write_text("matching:\n  sentence_treshold: 0.8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sentence_treshold"):
        load_config(bad)
