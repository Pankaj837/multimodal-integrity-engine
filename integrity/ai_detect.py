"""AI-generated text likelihood.

Three signals, all from open models running locally:

1. Perplexity (GPT-2). Language models assign high probability (low
   negative log-likelihood, NLL) to text that looks like their own output.
   We compute the mean per-token NLL with a context-aware sliding window.
2. Burstiness. Human writing mixes plain and surprising sentences, so the
   per-sentence NLL varies more. We use the standard deviation of the mean
   NLL of each sentence.
3. A RoBERTa classifier fine-tuned to separate ChatGPT answers from human
   answers (Hello-SimpleAI/chatgpt-detector-roberta, trained on HC3).

The three features are combined with a logistic-regression calibration
(``integrity/resources/ai_calibration.json``) fitted by
``scripts/calibrate_ai.py`` on a separate benchmark: human paragraphs from
pre-ChatGPT Wikipedia revisions vs. LLM-generated paragraphs. If the
calibration file is missing, the classifier probability is used alone.

Limitations are real and documented in EVALUATION.md: detectors are
unreliable on short texts, on non-native or highly formulaic human writing,
and on speech transcripts. The output is a *signal for review*, never proof.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .embedder import resolve_device
from .models import Document, Modality, SentenceKind

log = logging.getLogger(__name__)

_AI_LABEL_HINTS = ("chatgpt", "gpt", "ai", "fake", "machine", "generated", "llm")
_HUMAN_LABEL_HINTS = ("human", "real")
FEATURES = ["classifier_logit", "mean_nll", "burstiness"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class AIDetector:
    def __init__(self, config: Config, root: Optional[Path] = None):
        self.config = config
        self.device = resolve_device(config.models.device)
        self._lm = self._lm_tokenizer = None
        self._clf = self._clf_tokenizer = None
        self._ai_label = None
        calibration_path = Path(config.ai.calibration_file)
        if root is not None and not calibration_path.is_absolute():
            calibration_path = root / calibration_path
        self.calibration = None
        if calibration_path.exists():
            self.calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        else:
            log.warning("AI calibration file %s not found; using classifier probability only.", calibration_path)

    # ------------------------------------------------------------------ models
    def _load_lm(self):
        if self._lm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            name = self.config.models.ai_perplexity
            log.info("Loading perplexity model '%s'", name)
            self._lm_tokenizer = AutoTokenizer.from_pretrained(name)
            self._lm = AutoModelForCausalLM.from_pretrained(name).to(self.device).eval()
            torch.manual_seed(0)
        return self._lm, self._lm_tokenizer

    def _load_classifier(self):
        if self._clf is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            name = self.config.models.ai_classifier
            log.info("Loading AI-text classifier '%s'", name)
            self._clf_tokenizer = AutoTokenizer.from_pretrained(name)
            self._clf = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
            self._ai_label = _find_ai_label(self._clf.config.id2label)
        return self._clf, self._clf_tokenizer

    # ---------------------------------------------------------------- features
    def features(self, sentences: List[str]) -> Dict[str, float]:
        """Compute raw features for a list of sentences (one text)."""
        text, spans = _join_with_spans(sentences)
        mean_nll, sentence_nll = self._perplexity(text, spans)
        classifier_prob = self._classify(text)
        return {
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "burstiness": float(np.std(sentence_nll)) if len(sentence_nll) > 1 else 0.0,
            "classifier_probability": classifier_prob,
            "classifier_logit": _logit(classifier_prob),
        }

    def _perplexity(self, text: str, spans: List[tuple]):
        import torch

        model, tokenizer = self._load_lm()
        encoding = tokenizer(text, return_offsets_mapping=True)
        ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        max_len = min(1024, getattr(model.config, "n_positions", 1024))
        stride = max_len // 2
        token_nll = np.full(len(ids), np.nan)

        previous_end = 0
        for begin in range(0, len(ids), stride):
            end = min(begin + max_len, len(ids))
            window = torch.tensor([ids[begin:end]], device=self.device)
            with torch.no_grad():
                logits = model(window).logits[0]
            log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
            targets = window[0, 1:]
            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
            # token at absolute position begin+1+i is predicted by logits[i]
            for i, value in enumerate(nll):
                position = begin + 1 + i
                if position >= previous_end:  # score each token once, with max context
                    token_nll[position] = value
            previous_end = end
            if end == len(ids):
                break

        valid = ~np.isnan(token_nll)
        mean_nll = float(np.nanmean(token_nll)) if valid.any() else 0.0
        sentence_nll = []
        for start, stop in spans:
            members = [
                token_nll[i]
                for i, (a, b) in enumerate(offsets)
                if valid[i] and a >= start and b <= stop and b > a
            ]
            if members:
                sentence_nll.append(float(np.mean(members)))
        return mean_nll, sentence_nll

    def _classify(self, text: str) -> float:
        import torch

        model, tokenizer = self._load_classifier()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        window = self.config.ai.max_tokens_per_window - 2
        # Wrap each window in the model's own start/end tokens (<s> ... </s> for RoBERTa).
        start_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        end_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        probabilities, weights = [], []
        for begin in range(0, max(len(ids), 1), window):
            piece = ids[begin:begin + window]
            if not piece:
                break
            wrapped = ([start_id] if start_id is not None else []) + piece + ([end_id] if end_id is not None else [])
            inputs = torch.tensor([wrapped], device=self.device)
            with torch.no_grad():
                logits = model(input_ids=inputs).logits[0]
            probabilities.append(float(torch.softmax(logits.float(), dim=-1)[self._ai_label]))
            weights.append(len(piece))
        return float(np.average(probabilities, weights=weights)) if probabilities else 0.5

    # ------------------------------------------------------------- combination
    def combine(self, features: Dict[str, float]) -> float:
        """Calibrated probability (0-1) that the text is AI-generated."""
        if not self.calibration:
            return features["classifier_probability"]
        z = self.calibration["bias"]
        for name, weight, mean, std in zip(
            self.calibration["features"],
            self.calibration["weights"],
            self.calibration["means"],
            self.calibration["stds"],
        ):
            z += weight * (features[name] - mean) / (std or 1.0)
        return _sigmoid(z)

    def assess(self, document: Document) -> Dict:
        """AI-likelihood result for one document (dict ready for the report)."""
        sentences = [
            s.text for s in document.sentences if s.kind == SentenceKind.TEXT and not s.boilerplate
        ]
        n_words = sum(len(s.split()) for s in sentences)
        result = {
            "probability": None,
            "status": "OK",
            "assessed_words": n_words,
            "confidence": "normal",
            "signals": {},
            "notes": [],
        }
        if n_words < self.config.ai.min_words:
            result["status"] = "INSUFFICIENT_TEXT"
            result["notes"].append(
                f"Only {n_words} words of non-template prose; at least {self.config.ai.min_words} "
                "are needed for a meaningful estimate."
            )
            return result
        features = self.features(sentences)
        probability = self.combine(features)
        result["probability"] = round(100.0 * probability, 1)
        result["signals"] = {
            "perplexity": round(features["perplexity"], 2),
            "mean_token_nll": round(features["mean_nll"], 3),
            "burstiness": round(features["burstiness"], 3),
            "classifier_probability": round(100.0 * features["classifier_probability"], 1),
        }
        if document.modality == Modality.SPEECH:
            result["confidence"] = "low"
            result["notes"].append(
                "Speech transcript: ASR normalises disfluencies and punctuation, which makes "
                "statistical AI detection unreliable."
            )
        elif n_words < 200:
            result["confidence"] = "low"
            result["notes"].append("Short text (< 200 words): estimate has high variance.")
        return result


def _find_ai_label(id2label: Dict) -> int:
    labels = {int(k): str(v).lower() for k, v in id2label.items()}
    for index, label in labels.items():
        if any(h in label for h in _AI_LABEL_HINTS) and not any(h in label for h in _HUMAN_LABEL_HINTS):
            return index
    for index, label in labels.items():  # binary model with only a "human/real" label named
        if any(h in label for h in _HUMAN_LABEL_HINTS) and len(labels) == 2:
            return 1 - index
    raise ValueError(f"Cannot identify the AI label among {id2label}")


def _join_with_spans(sentences: List[str]):
    """Join sentences with spaces; return text and each sentence's char span."""
    spans, parts, cursor = [], [], 0
    for sentence in sentences:
        spans.append((cursor, cursor + len(sentence)))
        parts.append(sentence)
        cursor += len(sentence) + 1
    return " ".join(parts), spans
