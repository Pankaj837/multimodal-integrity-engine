"""Evaluate the engine against the dataset's ground truth (dataset_manifest.json).

    python scripts/evaluate.py

What is measured
----------------
1. Pair-level peer/cross-modal detection over every unordered pair of
   analysable submissions (13 files -> 78 pairs): predicted positive when at
   least one passage is reported between the two files.
2. Document-level flags: copying involvement (peer or external >= MEDIUM
   threshold), external-source detection, AI-generated text.
3. AI-detector benchmark (from integrity/resources/ai_calibration.json).
4. Automated pass/fail checks for the brief's five scenarios.
5. Threshold sweep and ablations (each component switched off), plus a
   lexical-only baseline, all on the same extracted documents. ASR runs
   once; only the analysis stages are repeated.

Outputs: evaluation/results.json and the block between the
``<!-- BEGIN GENERATED -->`` / ``<!-- END GENERATED -->`` markers in
EVALUATION.md (the rest of EVALUATION.md is hand-written analysis).
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

from integrity.config import load_config  # noqa: E402
from integrity.matching import lexical_scores  # noqa: E402
from integrity.pipeline import IntegrityPipeline, RunResult  # noqa: E402

SAMPLES = ROOT / "sample_submissions"
OUT_DIR = ROOT / "evaluation"
EVALUATION_MD = ROOT / "EVALUATION.md"


# ------------------------------------------------------------------ metrics
def prf(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3)}


def predicted_pairs(run: RunResult) -> dict:
    """{frozenset(a, b): max coverage %} for every pair with >= 1 reported passage."""
    names = [r.document.doc_id for r in run.results]
    pairs = {}
    for result in run.results:
        for other, coverage in result.pair_coverage.items():
            key = frozenset((result.document.doc_id, names[other]))
            pairs[key] = max(pairs.get(key, 0.0), coverage)
    return pairs


def pair_metrics(run: RunResult, manifest: dict, predicted: dict = None) -> dict:
    ok = [r.document.doc_id for r in run.results if r.document.ok]
    truth = {frozenset(p) for p in manifest["positive_pairs"]}
    predicted = predicted_pairs(run) if predicted is None else predicted
    tp = fp = fn = tn = 0
    false_pos, false_neg = [], []
    for a, b in itertools.combinations(ok, 2):
        key = frozenset((a, b))
        hit, real = key in predicted, key in truth
        tp += hit and real
        fp += hit and not real
        fn += real and not hit
        tn += not hit and not real
        if hit and not real:
            false_pos.append(sorted(key))
        if real and not hit:
            false_neg.append(sorted(key))
    result = prf(tp, fp, fn, tn)
    result.update({"pairs_evaluated": tp + fp + fn + tn, "false_positive_pairs": false_pos,
                   "false_negative_pairs": false_neg})
    return result


def doc_metrics(run: RunResult, positives: set, predicate) -> dict:
    tp = fp = fn = tn = 0
    errors = []
    for result in run.results:
        if not result.document.ok:
            continue
        flagged = predicate(result)
        if flagged is None:  # not assessable (e.g. AI on a short transcript)
            continue
        flagged = bool(flagged)
        real = result.document.doc_id in positives
        tp += flagged and real
        fp += flagged and not real
        fn += real and not flagged
        tn += not flagged and not real
        if flagged != real:
            errors.append(result.document.doc_id)
    out = prf(tp, fp, fn, tn)
    out["misclassified"] = errors
    return out


def lexical_baseline(run: RunResult, threshold: float = 0.7) -> dict:
    """Naive string matching: a pair is flagged when any two assessed sentences
    have normalised Levenshtein similarity >= threshold (no embeddings)."""
    docs = [r.document for r in run.results if r.document.ok]
    predicted = {}
    for a, b in itertools.combinations(docs, 2):
        sa = [s.text for s in a.sentences if s.assessed]
        sb = [s.text for s in b.sentences if s.assessed]
        best = max((lexical_scores(x, y)[1] for x in sa for y in sb), default=0.0)
        if best >= threshold:
            predicted[frozenset((a.doc_id, b.doc_id))] = round(best, 3)
    return predicted


# ---------------------------------------------------------------- scenarios
def scenario_checks(run: RunResult, manifest: dict) -> list:
    by_id = {r.document.doc_id: r for r in run.results}
    checks = []

    # Scenario 1 - cross-modal duplication with page + timestamp.
    truth = next(r for r in manifest["relationships"] if r["type"] == "cross_modal_copy")
    pitch = by_id["pitch_01.mp4"]
    names = [r.document.doc_id for r in run.results]
    hits = [p for p in pitch.peer_passages if names[p.target_doc] == "submission_05.pdf"]
    ok1 = False
    detail = "no cross-modal match found"
    if hits:
        p = hits[0]
        pages = {s.page for s in p.matched_sentences}
        start = min(s.start for s in p.query_sentences)
        end = max(s.end for s in p.query_sentences)
        t_start, t_end = truth["copy_location"]["start_seconds"], truth["copy_location"]["end_seconds"]
        overlap = max(0.0, min(end, t_end) - max(start, t_start)) / (t_end - t_start)
        ok1 = truth["source_location"]["page"] in pages and overlap >= 0.9
        detail = (f"matched page(s) {sorted(pages)} (truth {truth['source_location']['page']}); "
                  f"detected {start:.1f}-{end:.1f}s vs truth {t_start:.1f}-{t_end:.1f}s "
                  f"({overlap:.0%} overlap); cosine {p.similarity:.2f}")
    checks.append({"scenario": "1. Cross-modal duplication", "passed": bool(ok1), "detail": detail})

    # Scenario 2 - paraphrase: high semantic, low lexical similarity.
    s3 = by_id["submission_03.docx"]
    hits = [p for p in s3.peer_passages if names[p.target_doc] == "submission_04.docx"]
    ok2, detail = False, "no match found"
    if hits:
        p = max(hits, key=lambda x: x.query_words)
        semantic = p.chunk_similarity or p.similarity
        ok2 = semantic > 0.80 and p.jaccard < 0.5 and p.levenshtein < 0.6
        detail = (f"chunk cosine {semantic:.3f} (> 0.80) vs Jaccard {p.jaccard:.2f}, Levenshtein "
                  f"{p.levenshtein:.2f}; {len(p.pairs)} aligned sentences; peer {s3.peer_percentage:.1f}%")
    checks.append({"scenario": "2. Paraphrased content", "passed": bool(ok2), "detail": detail})

    # Scenario 3 - corrupted / empty submissions.
    expected = manifest["unprocessable"]
    statuses = {doc: by_id[doc].document.status.value for doc in expected}
    others_ok = all(r.document.ok for r in run.results if r.document.doc_id not in expected)
    ok3 = statuses == expected and others_ok
    checks.append({"scenario": "3. Corrupted / empty submissions", "passed": bool(ok3),
                   "detail": f"statuses {statuses}; all {len(run.results) - len(expected)} other files analysed: "
                             f"{others_ok}"})

    # Scenario 4 - shared template must not be flagged as plagiarism.
    group = next(r for r in manifest["relationships"] if r["type"] == "shared_template")["files"]
    planted = {frozenset(p) for p in manifest["positive_pairs"]}
    predicted = predicted_pairs(run)
    leaked = [sorted(k) for k in predicted if k <= set(group) and k not in planted]
    boiler = {d: sum(s.n_words for s in by_id[d].document.sentences if s.boilerplate) for d in group}
    clean = by_id["submission_11.docx"]
    ok4 = not leaked and min(boiler.values()) >= 140 and clean.risk == "LOW"
    checks.append({"scenario": "4. Benign common formatting", "passed": bool(ok4),
                   "detail": f"template words discounted per file {boiler}; non-planted pairs flagged inside the "
                             f"template group: {leaked or 'none'}; submission_11 (template + original) risk "
                             f"{clean.risk}, peer {clean.peer_percentage}%"})

    # Scenario 5 - whole batch processed with summary.
    ok5 = len(run.results) == len(manifest["files"])
    checks.append({"scenario": "5. Multi-document batch", "passed": bool(ok5),
                   "detail": f"{len(run.results)}/{len(manifest['files'])} files produced a result; summary matrix "
                             f"and per-file JSON/MD written by run_integrity_check.py"})
    return checks


# ---------------------------------------------------------------- reporting
def table(rows, headers) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def cm_table(m: dict) -> str:
    return table([["Actual positive", m["tp"], m["fn"]], ["Actual negative", m["fp"], m["tn"]]],
                 ["", "Predicted positive", "Predicted negative"])


def render(results: dict) -> str:
    main = results["main"]
    lines = ["## Results (generated by `python scripts/evaluate.py`)", ""]
    lines += ["### Scenario checks", "", table(
        [[c["scenario"], "PASS" if c["passed"] else "FAIL", c["detail"]] for c in results["scenarios"]],
        ["Scenario", "Result", "Evidence"]), ""]

    pm = main["pairs"]
    lines += ["### Pair-level detection (peer + cross-modal)", "",
              f"All {pm['pairs_evaluated']} unordered pairs of the 13 analysable files; 3 planted positive pairs "
              "(exact copy, paraphrase, cross-modal).", "", cm_table(pm), "",
              f"Precision **{pm['precision']:.3f}** · Recall **{pm['recall']:.3f}** · F1 **{pm['f1']:.3f}** · "
              f"false positives: {pm['false_positive_pairs'] or 'none'} · false negatives: "
              f"{pm['false_negative_pairs'] or 'none'}", ""]
    lines += ["Coverage of each planted pair (how much of the copying file was recovered):", "", table(
        [[k.replace(" | ", " ↔ "), f"{v:.1f}%"] for k, v in main["positive_pair_coverage"].items()],
        ["Pair", "Max coverage"]), ""]

    rows = []
    for name, m in main["documents"].items():
        rows.append([name, m["tp"], m["fp"], m["fn"], m["tn"], m["precision"], m["recall"], m["f1"],
                     ", ".join(m["misclassified"]) or "–"])
    lines += ["### Document-level flags", "", table(
        rows, ["Signal", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Misclassified"]), ""]

    ai = results["ai_benchmark"]
    loo, clf = ai["combined_leave_one_out"], ai["classifier_alone_at_0.5"]
    lines += ["### AI-detector benchmark", "",
              f"{ai['benchmark']['human_samples']} human paragraphs (pre-2022 Wikipedia) vs "
              f"{ai['benchmark']['ai_samples']} LLM-generated paragraphs on the same topics.", "", table(
                  [["RoBERTa classifier alone (p ≥ 0.5)", clf["roc_auc"], clf["accuracy"], clf["precision"],
                    clf["recall"], f"{clf['confusion_matrix']}"],
                   ["Calibrated combination (leave-one-out)", loo["roc_auc"], loo["accuracy"], loo["precision"],
                    loo["recall"], f"{loo['confusion_matrix']}"]],
                  ["Detector", "ROC-AUC", "Accuracy", "Precision", "Recall", "Confusion"]), "",
              "Stand-alone ROC-AUC per feature: " + ", ".join(
                  f"{k} {v}" for k, v in ai["single_feature_auc"].items() if not k.endswith("direction")), ""]

    lines += ["### Threshold sweep (pair level, AI disabled)", "", table(
        [[s["setting"], s["precision"], s["recall"], s["f1"], s["fp"], s["fn"],
          ", ".join(f"{v:.0f}%" for v in s["positive_coverage"])] for s in results["sweep"]],
        ["Setting", "Precision", "Recall", "F1", "FP pairs", "FN pairs", "Coverage of planted pairs"]), ""]

    def coverage(a, key):
        return ", ".join(f"{v:.0f}%" for v in a.get(key, [])) or "n/a"

    lines += ["### Ablations and baseline", "",
              "Pair-level metrics plus recovered coverage: planted peer pairs (01↔02, 03↔04, 05↔pitch_01) and "
              "external copies (submission_09, pitch_02).", "", table(
                  [[a["setting"], a["precision"], a["recall"], a["fp"], a["fn"], coverage(a, "positive_coverage"),
                    coverage(a, "external_coverage"), len(a.get("external_false_positive_docs", [])), a["note"]]
                   for a in results["ablations"]],
                  ["Variant", "Precision", "Recall", "FP", "FN", "Peer coverage", "External coverage",
                   "External FP docs", "What changed"]), ""]
    return "\n".join(lines)


def update_evaluation_md(block: str) -> None:
    begin, end = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
    text = EVALUATION_MD.read_text(encoding="utf-8") if EVALUATION_MD.exists() else f"{begin}\n{end}\n"
    if begin not in text:
        text += f"\n{begin}\n{end}\n"
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    EVALUATION_MD.write_text(f"{head}{begin}\n{block}\n{end}{tail}", encoding="utf-8")


# --------------------------------------------------------------------- main
def main() -> int:
    manifest = json.loads((SAMPLES / "dataset_manifest.json").read_text(encoding="utf-8"))
    base_cfg = load_config(ROOT / "config.yaml")
    pipeline = IntegrityPipeline(base_cfg, ROOT)
    documents = pipeline.extract(SAMPLES)

    print("Main run (all components, AI enabled) ...")
    run = pipeline.analyse(documents)
    risk = base_cfg.risk
    copying = {f for p in manifest["positive_pairs"] for f in p} | set(manifest["external_positives"])
    predicted = predicted_pairs(run)
    main_results = {
        "pairs": pair_metrics(run, manifest),
        "positive_pair_coverage": {
            " | ".join(p): predicted.get(frozenset(p), 0.0) for p in manifest["positive_pairs"]
        },
        "documents": {
            "Copying involvement (peer or external ≥ 10%)": doc_metrics(
                run, copying,
                lambda r: max(r.peer_percentage, r.external_percentage) >= risk.medium_similarity),
            "External-source copying (external ≥ 10%)": doc_metrics(
                run, set(manifest["external_positives"]),
                lambda r: r.external_percentage >= risk.medium_similarity),
            "AI-generated (AI probability ≥ 60%)": doc_metrics(
                run, set(manifest["ai_generated"]),
                lambda r: None if r.ai.get("probability") is None else r.ai["probability"] >= risk.medium_ai),
            "Overall risk ≥ MEDIUM vs any planted issue": doc_metrics(
                run, copying | set(manifest["ai_generated"]), lambda r: r.risk in ("MEDIUM", "HIGH")),
        },
    }
    scenarios = scenario_checks(run, manifest)
    # Baseline must be computed now: later variants re-segment the same Document objects.
    lm = pair_metrics(run, manifest, predicted=lexical_baseline(run))

    def variant(label, note, mutate):
        cfg = copy.deepcopy(base_cfg)
        cfg.ai.enabled = False
        mutate(cfg)
        pipeline.config = cfg
        result = pipeline.analyse(documents)
        m = pair_metrics(result, manifest)
        pred = predicted_pairs(result)
        print(f"  {label:45} P={m['precision']:.3f} R={m['recall']:.3f} FP={m['fp']} FN={m['fn']}")
        externals = {r.document.doc_id: r.external_percentage for r in result.results if r.document.ok}
        return {"setting": label, "note": note, **{k: m[k] for k in ("precision", "recall", "f1", "fp", "fn")},
                "false_positive_pairs": m["false_positive_pairs"],
                "positive_coverage": [pred.get(frozenset(p), 0.0) for p in manifest["positive_pairs"]],
                "external_coverage": [externals.get(d, 0.0) for d in manifest["external_positives"]],
                "external_false_positive_docs": sorted(
                    d for d, v in externals.items() if v > 0 and d not in manifest["external_positives"])}

    print("Threshold sweep ...")
    sweep = []
    for t in (0.75, 0.80, 0.85, 0.90, 0.95):
        sweep.append(variant(f"sentence_threshold = {t:.2f}", "",
                             lambda c, t=t: setattr(c.matching, "sentence_threshold", t)))
    for t in (0.75, 0.80, 0.90):
        sweep.append(variant(f"context_chunk_threshold = {t:.2f}", "",
                             lambda c, t=t: setattr(c.matching, "context_chunk_threshold", t)))

    print("Ablations ...")
    ablations = [
        variant("Full engine (defaults)", "reference point", lambda c: None),
        variant("No boilerplate detection", "template text treated as normal content",
                lambda c: setattr(c.boilerplate, "enabled", False)),
        variant("No context (chunk-supported) matching", "context_chunk_threshold = 1.01",
                lambda c: setattr(c.matching, "context_chunk_threshold", 1.01)),
        variant("No lexical rescue", "lexical_rescue_cosine = 1.0",
                lambda c: setattr(c.matching, "lexical_rescue_cosine", 1.0)),
        variant("No evidence-strength filter", "isolated 1-sentence semantic matches kept",
                lambda c: setattr(c.matching, "single_sentence_min_similarity", 0.0)),
        variant("No evidence filter + sentence_threshold 0.80", "shows what the filter protects against",
                lambda c: (setattr(c.matching, "single_sentence_min_similarity", 0.0),
                           setattr(c.matching, "sentence_threshold", 0.80))),
        variant("Old defaults (0.80 / 0.75 / 0.65, no filter)", "pre-calibration thresholds",
                lambda c: (setattr(c.matching, "sentence_threshold", 0.80),
                           setattr(c.matching, "context_chunk_threshold", 0.75),
                           setattr(c.matching, "context_sentence_threshold", 0.65),
                           setattr(c.matching, "single_sentence_min_similarity", 0.0))),
    ]
    pipeline.config = base_cfg
    ablations.append({"setting": "Lexical-only baseline (Levenshtein ≥ 0.7, no embeddings)",
                      "note": "classic string matching on the same sentences",
                      **{k: lm[k] for k in ("precision", "recall", "f1", "fp", "fn")},
                      "false_positive_pairs": lm["false_positive_pairs"],
                      "false_negative_pairs": lm["false_negative_pairs"]})

    ai_benchmark = json.loads((ROOT / "integrity/resources/ai_calibration.json").read_text(encoding="utf-8"))
    results = {"main": main_results, "scenarios": scenarios, "sweep": sweep, "ablations": ablations,
               "ai_benchmark": ai_benchmark,
               "per_document": [{"document_id": r.document.doc_id, "status": r.document.status.value,
                                 "risk": r.risk, "peer": r.peer_percentage, "external": r.external_percentage,
                                 "ai": r.ai.get("probability")} for r in run.results]}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    update_evaluation_md(render(results))
    print("\n".join(f"{'PASS' if c['passed'] else 'FAIL'}  {c['scenario']}" for c in scenarios))
    print(f"Pair-level: P={main_results['pairs']['precision']} R={main_results['pairs']['recall']}")
    print("Wrote evaluation/results.json and updated EVALUATION.md")
    return 0 if all(c["passed"] for c in scenarios) else 1


if __name__ == "__main__":
    sys.exit(main())
