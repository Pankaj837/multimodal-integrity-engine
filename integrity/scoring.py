"""Similarity percentages and the overall risk level.

Definitions (all word-weighted, so long sentences count more than short ones):

    assessed words A   = words in eligible prose sentences that are NOT boilerplate
    peer %             = 100 * (words of assessed sentences matched in >= 1 other submission) / A
    external %         = 100 * (words of assessed sentences matched in the reference corpus) / A
    pair coverage[i,j] = 100 * (words of i's assessed sentences matched in submission j) / A_i

Risk level (thresholds in config.yaml -> risk):

    HIGH   if max(peer %, external %) >= high_similarity
           or (AI probability >= high_ai and the AI estimate is normal-confidence)
    MEDIUM if max(peer %, external %) >= medium_similarity or AI probability >= medium_ai
    LOW    otherwise
    UNPROCESSABLE when the file could not be analysed (empty, corrupt, silent ...)

A low-confidence AI estimate (speech transcript / short text) can raise the
risk to MEDIUM at most: an unreliable signal alone must not trigger a penalty.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set

from .config import RiskConfig
from .models import Document


def coverage_percentage(document: Document, matched_sentence_ids: Iterable[int]) -> float:
    total = document.assessed_words
    if total == 0:
        return 0.0
    matched: Set[int] = set(matched_sentence_ids)
    words = sum(s.n_words for s in document.sentences if s.assessed and s.index in matched)
    return round(100.0 * words / total, 1)


def risk_level(
    peer: Optional[float],
    external: Optional[float],
    ai_probability: Optional[float],
    ai_confidence: str,
    cfg: RiskConfig,
) -> str:
    similarity = max(peer or 0.0, external or 0.0)
    ai = ai_probability if ai_probability is not None else 0.0
    if similarity >= cfg.high_similarity or (ai >= cfg.high_ai and ai_confidence == "normal"):
        return "HIGH"
    if similarity >= cfg.medium_similarity or ai >= cfg.medium_ai:
        return "MEDIUM"
    return "LOW"


def risk_reasons(
    peer: float, external: float, ai_probability: Optional[float], ai_confidence: str, cfg: RiskConfig
) -> Dict[str, str]:
    """Human-readable reason for each signal that crossed a threshold."""
    reasons = {}
    for name, value in (("peer similarity", peer), ("external similarity", external)):
        if value >= cfg.high_similarity:
            reasons[name] = f"{name} {value:.1f}% >= {cfg.high_similarity:.0f}% (HIGH threshold)"
        elif value >= cfg.medium_similarity:
            reasons[name] = f"{name} {value:.1f}% >= {cfg.medium_similarity:.0f}% (MEDIUM threshold)"
    if ai_probability is not None:
        if ai_probability >= cfg.high_ai and ai_confidence == "normal":
            reasons["ai"] = f"AI-generation probability {ai_probability:.1f}% >= {cfg.high_ai:.0f}% (HIGH threshold)"
        elif ai_probability >= cfg.medium_ai:
            note = " (low-confidence estimate, capped at MEDIUM)" if ai_confidence != "normal" else ""
            reasons["ai"] = f"AI-generation probability {ai_probability:.1f}% >= {cfg.medium_ai:.0f}%{note}"
    return reasons
