"""Benchmark and calibrate the AI-generated-text detector.

Data (benchmarks/ai_detection/):
  human.jsonl         genuinely human prose: paragraphs of pinned Wikipedia
                      revisions saved before ChatGPT's release (2022-11-30).
                      Selected deterministically: the first two paragraphs of
                      90-180 words from each reference article.
  ai_generated.jsonl  LLM-generated paragraphs on the same 14 topics and in
                      the same encyclopedic register (generator recorded per line).

Steps:
  1. compute the detector's raw features for every text
     (RoBERTa ChatGPT-detector logit, GPT-2 mean token NLL, burstiness);
  2. report the stand-alone ROC-AUC of each feature;
  3. fit a standardised logistic regression on the three features and report
     honest leave-one-out (LOO) metrics - each text is scored by a model that
     never saw it;
  4. refit on all data and write integrity/resources/ai_calibration.json.

    python scripts/calibrate_ai.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score  # noqa: E402

from integrity.ai_detect import FEATURES, AIDetector  # noqa: E402
from integrity.config import load_config  # noqa: E402
from integrity.segmentation import split_sentences  # noqa: E402

BENCH = ROOT / "benchmarks" / "ai_detection"
OUT = ROOT / "integrity" / "resources" / "ai_calibration.json"
MIN_WORDS, MAX_WORDS, PER_ARTICLE = 90, 180, 2
C_REGULARISATION = 1.0  # sklearn default; small data -> keep the default L2 penalty


def build_human_set() -> list:
    manifest = json.loads((ROOT / "reference_corpus" / "manifest.json").read_text(encoding="utf-8"))
    samples = []
    for article in manifest:
        paragraphs = (ROOT / "reference_corpus" / article["file"]).read_text(encoding="utf-8").split("\n\n")
        chosen = [p.strip() for p in paragraphs if MIN_WORDS <= len(p.split()) <= MAX_WORDS][:PER_ARTICLE]
        for text in chosen:
            samples.append({
                "id": f"human_{len(samples) + 1:02d}",
                "topic": article["title"],
                "source": article["url"],
                "revision_timestamp": article["revision_timestamp"],
                "text": text,
            })
    with open(BENCH / "human.jsonl", "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return samples


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    human = build_human_set()
    ai = load_jsonl(BENCH / "ai_generated.jsonl")
    texts = [s["text"] for s in human] + [s["text"] for s in ai]
    labels = np.array([0] * len(human) + [1] * len(ai))
    ids = [s["id"] for s in human] + [s["id"] for s in ai]
    print(f"Benchmark: {len(human)} human (pre-2022 Wikipedia) + {len(ai)} AI-generated paragraphs")

    detector = AIDetector(load_config(ROOT / "config.yaml"), ROOT)
    detector.calibration = None  # raw features only
    rows = []
    for text, sample_id in zip(texts, ids):
        features = detector.features(split_sentences(text))
        rows.append([features[name] for name in FEATURES])
        print(f"  {sample_id:9} clf={features['classifier_probability']:.3f} nll={features['mean_nll']:.3f} "
              f"burst={features['burstiness']:.3f}")
    X = np.array(rows)

    single_auc = {}
    for j, name in enumerate(FEATURES):
        auc = roc_auc_score(labels, X[:, j])
        single_auc[name] = round(float(max(auc, 1 - auc)), 3)
        single_auc[name + "_direction"] = "higher=AI" if auc >= 0.5 else "lower=AI"

    means, stds = X.mean(axis=0), X.std(axis=0)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds

    loo_prob = np.zeros(len(labels))
    for i in range(len(labels)):
        mask = np.arange(len(labels)) != i
        model = LogisticRegression(C=C_REGULARISATION, max_iter=1000).fit(Z[mask], labels[mask])
        loo_prob[i] = model.predict_proba(Z[i:i + 1])[0, 1]
    loo_pred = (loo_prob >= 0.5).astype(int)
    classifier_only = (X[:, 0] >= 0.0).astype(int)  # logit >= 0  <=> probability >= 0.5

    def metrics(pred, score):
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        return {
            "roc_auc": round(float(roc_auc_score(labels, score)), 3),
            "accuracy": round(float(accuracy_score(labels, pred)), 3),
            "precision": round(float(precision_score(labels, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(labels, pred, zero_division=0)), 3),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

    final = LogisticRegression(C=C_REGULARISATION, max_iter=1000).fit(Z, labels)
    calibration = {
        "description": "Standardised logistic regression over detector features; see scripts/calibrate_ai.py",
        "features": FEATURES,
        "weights": [round(float(w), 6) for w in final.coef_[0]],
        "bias": round(float(final.intercept_[0]), 6),
        "means": [round(float(m), 6) for m in means],
        "stds": [round(float(s), 6) for s in stds],
        "benchmark": {
            "human_samples": len(human),
            "ai_samples": len(ai),
            "human_source": "Wikipedia revisions saved before 2022-11-01 (CC BY-SA 4.0)",
            "ai_source": "LLM-generated paragraphs on the same topics (see ai_generated.jsonl)",
        },
        "single_feature_auc": single_auc,
        "classifier_alone_at_0.5": metrics(classifier_only, X[:, 0]),
        "combined_leave_one_out": metrics(loo_pred, loo_prob),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    with open(BENCH / "features.csv", "w", encoding="utf-8") as handle:
        handle.write("id,label," + ",".join(FEATURES) + ",loo_probability\n")
        for sample_id, label, row, prob in zip(ids, labels, X, loo_prob):
            handle.write(f"{sample_id},{'ai' if label else 'human'}," + ",".join(f"{v:.4f}" for v in row)
                         + f",{prob:.4f}\n")

    print("\nSingle-feature AUC:", single_auc)
    print("Classifier alone  :", calibration["classifier_alone_at_0.5"])
    print("Combined (LOO)    :", calibration["combined_leave_one_out"])
    print(f"Wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
