"""Per-document JSON + Markdown audit reports and the batch summary.

JSON reports keep the exact keys of the brief's schema, in the same order
(``document_id``, ``total_integrity_risk``, ``scores``, ``peer_matches``,
``multimodal_matches``, ``explanations``), followed by clearly separated
extensions (``external_matches``, ``boilerplate_passages``, ``ai_assessment``,
``processing``, ``score_details``, ``pipeline``). Keys are plain ASCII; the
brief's PDF renders "file" with a U+FB01 ligature, which must not leak in.

Markdown reports are written for a reviewer who has 30 seconds: verdict and
scores first, then an evidence table, then each match with both passages
quoted side by side and the exact page / timestamp to open.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .matching import Passage
from .models import Document, Modality, Sentence, Status
from .pipeline import DocumentResult, RunResult, is_cross_modal

RISK_BADGE = {"HIGH": "🔴 HIGH", "MEDIUM": "🟠 MEDIUM", "LOW": "🟢 LOW", "UNPROCESSABLE": "⚪ UNPROCESSABLE"}
_STOPWORDS = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its this that "
    "these those we our they their he she his her you your i not no so than then there here which who whom "
    "what when where why how all any each more most other some such can could should would may might must "
    "will shall do does did has have had into over under about also only very just".split()
)
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")


# --------------------------------------------------------------------------- utils
def format_timestamp(seconds: Optional[float], round_up: bool = False) -> str:
    if seconds is None:
        return "??:??:??"
    total = int(math.ceil(seconds)) if round_up else int(math.floor(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _span(sentences: List[Sentence]) -> Dict:
    """Location of a group of sentences: page range, paragraph range or time range."""
    pages = sorted({s.page for s in sentences if s.page is not None})
    paragraphs = sorted({s.paragraph for s in sentences if s.paragraph is not None})
    starts = [s.start for s in sentences if s.start is not None]
    ends = [s.end for s in sentences if s.end is not None]
    span: Dict = {}
    if starts:
        span["timestamp"] = f"{format_timestamp(min(starts))} - {format_timestamp(max(ends), round_up=True)}"
        span["seconds"] = [round(min(starts), 2), round(max(ends), 2)]
    if pages:
        span["page"] = pages[0]
        if pages[-1] != pages[0]:
            span["page_end"] = pages[-1]
    if paragraphs:
        span["paragraph"] = paragraphs[0]
    return span


def _location(passage: Passage, matched_is_reference: bool) -> Dict:
    """Location object in the brief's style (query_page / matched_page / matched_timestamp)."""
    q, m = _span(passage.query_sentences), _span(passage.matched_sentences)
    location: Dict = {}
    for prefix, span in (("query", q), ("matched", m)):
        if "timestamp" in span:
            location[f"{prefix}_timestamp"] = span["timestamp"]
        elif "page" in span and not (prefix == "matched" and matched_is_reference):
            location[f"{prefix}_page"] = span["page"]
            if "page_end" in span:
                location[f"{prefix}_page_end"] = span["page_end"]
        if "paragraph" in span and ("page" not in span or prefix == "matched" and matched_is_reference):
            location[f"{prefix}_paragraph"] = span["paragraph"]
    return location


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + " …"


def _human_location(span: Dict, is_reference: bool = False) -> str:
    if "timestamp" in span:
        return span["timestamp"]
    if is_reference and "paragraph" in span:
        return f"para. {span['paragraph']}"
    if "page" in span:
        return f"p. {span['page']}" + (f"–{span['page_end']}" if "page_end" in span else "")
    if "paragraph" in span:
        return f"para. {span['paragraph']}"
    return "n/a"


def _highlight(text: str, other: str) -> str:
    """Bold content words shared with the other passage (lexical overlap)."""
    shared = {t.lower() for t in _TOKEN.findall(other)} - _STOPWORDS
    return _TOKEN.sub(lambda m: f"**{m.group(0)}**" if m.group(0).lower() in shared else m.group(0), text)


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


# ------------------------------------------------------------------- report body
class ReportBuilder:
    def __init__(self, run: RunResult):
        self.run = run
        self.cfg = run.config
        self.documents = [r.document for r in run.results]

    def _target(self, passage: Passage, external: bool) -> Document:
        return self.run.reference_documents[passage.target_doc] if external else self.documents[passage.target_doc]

    def _match_json(self, passage: Passage, external: bool) -> Dict:
        target = self._target(passage, external)
        entry = {
            "matched_file": target.doc_id,
            "similarity_score": round(passage.similarity, 2),
            "query_passage": passage.query_text,
            "matched_passage": passage.matched_text,
            "location": _location(passage, matched_is_reference=external),
            "match_type": passage.match_type,
            "chunk_similarity": passage.chunk_similarity,
            "lexical_similarity": {"jaccard": passage.jaccard, "levenshtein": passage.levenshtein},
            "matched_sentences": len(passage.pairs),
            "query_words": passage.query_words,
            "evidence": sorted({p.method for p in passage.pairs}),
            "sentence_alignment": passage.alignment,
        }
        if external:
            meta = self.run.reference_metadata[passage.target_doc]
            entry = {"matched_source": meta["title"], **entry, "source_url": meta.get("url")}
        return entry

    def explanation(self, result: DocumentResult) -> str:
        document = result.document
        if not document.ok:
            reason = document.warnings[-1] if document.warnings else document.status.value
            return f"File could not be analysed ({document.status.value}): {reason} No scores were computed."

        parts: List[str] = []
        if result.peer_passages:
            by_target: Dict[int, List[Passage]] = {}
            for passage in result.peer_passages:
                by_target.setdefault(passage.target_doc, []).append(passage)
            ranked = sorted(by_target, key=lambda t: -result.pair_coverage.get(t, 0.0))
            pieces = []
            for target in ranked[:3]:
                passages = by_target[target]
                types = sorted({p.match_type for p in passages}, key=["verbatim", "near_verbatim", "paraphrase"].index)
                where = ", ".join(sorted({_human_location(_span(p.query_sentences)) for p in passages}))
                kind = "cross-modal " if is_cross_modal(document, self.documents[target]) else ""
                detail = ""
                if types == ["paraphrase"]:
                    strongest = max(passages, key=lambda p: p.query_words)
                    detail = (
                        "; semantic paraphrase - chunk cosine "
                        f"{strongest.chunk_similarity or strongest.similarity:.2f} "
                        f"despite low wording overlap (Jaccard {strongest.jaccard:.2f}, "
                        f"Levenshtein {strongest.levenshtein:.2f})"
                    )
                pieces.append(
                    f"{self.documents[target].doc_id} ({result.pair_coverage.get(target, 0):.1f}% of this file, "
                    f"{kind}{'/'.join(t.replace('_', '-') for t in types)}, at {where}{detail})"
                )
            parts.append(
                f"{result.peer_percentage:.1f}% of the assessed text overlaps with other submissions: "
                + "; ".join(pieces) + "."
            )
        else:
            parts.append("No significant overlap with other submissions.")

        if result.external_passages:
            sources = sorted({self.run.reference_metadata[p.target_doc]["title"] for p in result.external_passages})
            parts.append(
                f"{result.external_percentage:.1f}% matches external reference sources ({', '.join(sources)})."
            )
        else:
            parts.append("No match with the external reference corpus.")

        boilerplate = [s for s in document.sentences if s.boilerplate]
        if boilerplate:
            words = sum(s.n_words for s in boilerplate)
            shared = max(s.boilerplate_docs for s in boilerplate)
            parts.append(
                f"{words} words of shared template/boilerplate text (present in up to {shared} submissions) "
                "were recognised and excluded from scoring."
            )

        ai = result.ai
        if ai.get("probability") is not None:
            confidence = " (low-confidence estimate)" if ai.get("confidence") == "low" else ""
            parts.append(f"AI-generation likelihood is {ai['probability']:.1f}%{confidence}.")
        elif ai.get("status") == "INSUFFICIENT_TEXT":
            parts.append("AI-generation likelihood not estimated: too little non-template prose.")

        if result.reasons:
            parts.append(f"Risk {result.risk} because " + "; ".join(result.reasons.values()) + ".")
        else:
            parts.append(f"Risk {result.risk}: no signal crossed a review threshold.")
        if result.peer_passages:
            parts.append("Note: similarity is symmetric - this report cannot tell which file is the original.")
        return " ".join(parts)

    def document_json(self, result: DocumentResult) -> Dict:
        document = result.document
        peer, multimodal = [], []
        for passage in result.peer_passages:
            target = self.documents[passage.target_doc]
            (multimodal if is_cross_modal(document, target) else peer).append(self._match_json(passage, False))
        boilerplate = [
            {
                "text": s.text,
                "shared_by_documents": s.boilerplate_docs,
                "location": {k: v for k, v in _span([s]).items() if k != "seconds"},
            }
            for s in document.sentences
            if s.boilerplate
        ]
        ai = result.ai or {}
        return {
            "document_id": document.doc_id,
            "total_integrity_risk": result.risk,
            "scores": {
                "peer_similarity_percentage": result.peer_percentage,
                "external_similarity_percentage": result.external_percentage,
                "ai_generated_probability": ai.get("probability"),
            },
            "peer_matches": peer,
            "multimodal_matches": multimodal,
            "explanations": self.explanation(result),
            # ---- extensions beyond the brief's example schema -------------
            "external_matches": [self._match_json(p, True) for p in result.external_passages],
            "boilerplate_passages": boilerplate,
            "ai_assessment": ai,
            "risk_reasons": list(result.reasons.values()),
            "processing": {
                "status": document.status.value,
                "file_type": document.file_type,
                "modality": document.modality.value,
                "pages": document.n_pages,
                "duration_seconds": round(document.duration_seconds, 2) if document.duration_seconds else None,
                "language": document.language,
                "sentences": len(document.sentences),
                "chunks": len(document.chunks),
                "header_footer_lines_removed": document.removed_header_footer_lines,
                "warnings": document.warnings,
            },
            "score_details": {
                "assessed_words": document.assessed_words if document.ok else 0,
                "boilerplate_words": sum(s.n_words for s in document.sentences if s.boilerplate),
                "pair_coverage_percentage": {
                    self.documents[k].doc_id: v for k, v in sorted(result.pair_coverage.items(), key=lambda x: -x[1])
                },
            },
        }

    # ------------------------------------------------------------- markdown
    def document_markdown(self, result: DocumentResult, data: Dict) -> str:
        document = result.document
        scores = data["scores"]
        md: List[str] = [f"# Integrity report: `{document.doc_id}`", ""]
        if not document.ok:
            md += [
                f"**Overall risk:** {RISK_BADGE['UNPROCESSABLE']}  ",
                f"**Status:** `{document.status.value}`",
                "",
                f"> {data['explanations']}",
                "",
                "The rest of the batch was processed normally. Re-submit a readable file to obtain scores.",
                "",
            ]
            return "\n".join(md)

        ai = result.ai or {}
        ai_value = scores["ai_generated_probability"]
        ai_cell = f"{ai_value:.1f}%" if ai_value is not None else "n/a"
        if ai.get("confidence") == "low":
            ai_cell += " (low confidence)"
        md += [
            "| Overall risk | Peer similarity | External similarity | AI-generated probability |",
            "|---|---|---|---|",
            f"| **{RISK_BADGE[result.risk]}** | **{scores['peer_similarity_percentage']:.1f}%** | "
            f"**{scores['external_similarity_percentage']:.1f}%** | **{ai_cell}** |",
            "",
            f"**Verdict.** {data['explanations']}",
            "",
        ]
        info = [f"{document.file_type.upper()}"]
        if document.n_pages:
            info.append(f"{document.n_pages} page(s)")
        if document.duration_seconds:
            info.append(f"{document.duration_seconds:.0f} s of media")
        info.append(f"{data['score_details']['assessed_words']} assessed words")
        if data["score_details"]["boilerplate_words"]:
            info.append(f"{data['score_details']['boilerplate_words']} template words excluded")
        md += [f"**File:** {' · '.join(info)}", ""]

        evidence = [("Peer", m, False) for m in data["peer_matches"]]
        evidence += [("Cross-modal", m, False) for m in data["multimodal_matches"]]
        evidence += [("External", m, True) for m in data["external_matches"]]
        evidence.sort(key=lambda e: -(e[1]["similarity_score"] * e[1]["query_words"]))
        limit = self.cfg.report.max_matches_in_markdown

        if evidence:
            md += [
                "## Evidence at a glance",
                "",
                "| # | Type | Matched source | Here | There | Similarity | Lexical (Lev.) | Kind |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for i, (label, match, is_ref) in enumerate(evidence[:limit], 1):
                source = match.get("matched_source", match["matched_file"])
                here, there = self._md_locations(match["location"], is_ref)
                md.append(
                    f"| {i} | {label} | `{_md_escape(source)}` | {here} | {there} | "
                    f"{match['similarity_score']:.2f} | {match['lexical_similarity']['levenshtein']:.2f} | "
                    f"{match['match_type'].replace('_', '-')} |"
                )
            if len(evidence) > limit:
                md.append(f"\n_{len(evidence) - limit} further match(es) are listed in the JSON report._")
            md.append("")

            md += ["## Matched passages", ""]
            for i, (label, match, is_ref) in enumerate(evidence[:limit], 1):
                md += self._md_passage(i, label, match, is_ref)
        else:
            md += ["## Evidence", "", "No passage crossed the similarity threshold.", ""]

        if data["boilerplate_passages"]:
            md += [
                "## Discounted template / boilerplate text",
                "",
                "These sentences also appear (near-identically) in many other submissions, so they are treated "
                "as shared case-prompt or institutional text and **not** counted as plagiarism:",
                "",
            ]
            for entry in data["boilerplate_passages"][:6]:
                md.append(f"- _{_truncate(entry['text'], 160)}_ (shared by {entry['shared_by_documents']} files)")
            if len(data["boilerplate_passages"]) > 6:
                md.append(f"- … and {len(data['boilerplate_passages']) - 6} more (see JSON).")
            md.append("")

        md += ["## AI-generation signals", ""]
        if ai.get("probability") is not None:
            sig = ai["signals"]
            md += [
                f"- Combined estimate: **{ai['probability']:.1f}%** (confidence: {ai['confidence']})",
                f"- RoBERTa ChatGPT-detector probability: {sig['classifier_probability']:.1f}%",
                f"- GPT-2 perplexity: {sig['perplexity']:.1f} (lower = more machine-like)",
                f"- Burstiness (std. of sentence NLL): {sig['burstiness']:.2f} (lower = more uniform)",
            ]
        else:
            md.append(f"- Not estimated ({ai.get('status', 'n/a')}).")
        md += [f"- {note}" for note in ai.get("notes", [])]
        md += ["- AI detectors are probabilistic; treat this as a prompt for review, never as proof.", ""]

        if document.warnings:
            md += ["## Processing notes", ""] + [f"- {w}" for w in document.warnings] + [""]
        return "\n".join(md)

    @staticmethod
    def _md_locations(location: Dict, is_reference: bool):
        def pick(prefix):
            if f"{prefix}_timestamp" in location:
                return location[f"{prefix}_timestamp"]
            if f"{prefix}_page" in location:
                end = location.get(f"{prefix}_page_end")
                return f"p. {location[f'{prefix}_page']}" + (f"–{end}" if end else "")
            if f"{prefix}_paragraph" in location:
                return f"para. {location[f'{prefix}_paragraph']}"
            return "n/a"

        return pick("query"), pick("matched")

    def _md_passage(self, i: int, label: str, match: Dict, is_reference: bool) -> List[str]:
        limit = self.cfg.report.max_passage_chars
        source = match.get("matched_source", match["matched_file"])
        here, there = self._md_locations(match["location"], is_reference)
        verbatim = match["match_type"] == "verbatim"
        query = _truncate(match["query_passage"], limit)
        matched = _truncate(match["matched_passage"], limit)
        if not verbatim:
            query, matched = _highlight(query, match["matched_passage"]), _highlight(matched, match["query_passage"])
        lines = [
            f"### {i}. {label} match: `{source}`",
            "",
            f"Similarity **{match['similarity_score']:.2f}** (chunk {match['chunk_similarity'] or 'n/a'}) · "
            f"Jaccard {match['lexical_similarity']['jaccard']:.2f} · "
            f"Levenshtein {match['lexical_similarity']['levenshtein']:.2f} · "
            f"**{match['match_type'].replace('_', '-')}**"
            + (" (identical wording)" if verbatim else " (shared content words in bold)"),
            "",
            f"**This file ({here}):**",
            f"> {query}",
            "",
            f"**{source} ({there}):**",
            f"> {matched}",
            "",
        ]
        alignment = match.get("sentence_alignment") or []
        if len(alignment) > 1 or not verbatim:
            shown = sorted(alignment, key=lambda a: -a["similarity"])[:4]
            lines += [
                f"Sentence alignment (strongest {len(shown)} of {len(alignment)}):",
                "",
                "| This file | Matched source | Cosine | Evidence |",
                "|---|---|---|---|",
            ]
            for row in shown:
                lines.append(
                    f"| {_md_escape(_truncate(row['query_sentence'], 220))} | "
                    f"{_md_escape(_truncate(row['matched_sentence'], 220))} | "
                    f"{row['similarity']:.2f} | {row['method']} |"
                )
            lines.append("")
        if match.get("source_url"):
            lines += [f"Source: <{match['source_url']}>", ""]
        return lines


# ------------------------------------------------------------------- writers
def _write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _matrix_cell(value: Optional[float]) -> str:
    if value is None:
        return "·"  # diagonal: a document is never compared with itself
    if value == 0:
        return ""
    return f"**{value:.0f}**" if value >= 30 else f"{value:.0f}"


def _safe_name(doc_id: str) -> str:
    """File-system-safe report name. Keeps Unicode letters (``\\w``) so that
    'résumé.pdf' and 'resume.pdf' cannot collide; path separators and spaces
    become underscores (``nested/a.pdf`` -> ``nested_a.pdf``)."""
    return re.sub(r"[^\w.\-]+", "_", doc_id)


def write_reports(run: RunResult, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    builder = ReportBuilder(run)
    rows = []
    for result in run.results:
        data = builder.document_json(result)
        base = _safe_name(result.document.doc_id)
        _write_json(output_dir / f"{base}.json", data)
        (output_dir / f"{base}.md").write_text(builder.document_markdown(result, data), encoding="utf-8")
        rows.append(data)
    summary = _summary(run, rows)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(_summary_markdown(run, summary), encoding="utf-8")
    return summary


def _short_labels(documents: List[Document]) -> List[str]:
    labels = []
    for document in documents:
        stem = Path(document.doc_id).stem
        digits = re.findall(r"\d+", stem)
        prefix = "P" if document.modality == Modality.SPEECH else "S"
        labels.append(f"{prefix}{digits[-1]}" if digits else stem[:6])
    if len(set(labels)) != len(labels):  # fall back to positional labels if ambiguous
        labels = [f"D{i + 1:02d}" for i in range(len(documents))]
    return labels


def _summary(run: RunResult, rows: List[Dict]) -> Dict:
    ok_results = [r for r in run.results if r.document.ok]
    all_docs = [r.document for r in run.results]
    matrix = []
    for result in ok_results:
        row = []
        for other in ok_results:
            if other is result:
                row.append(None)
            else:
                other_index = all_docs.index(other.document)
                row.append(result.pair_coverage.get(other_index, 0.0))
        matrix.append(row)
    labels = _short_labels([r.document for r in ok_results])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": run.version,
        "device": run.device,
        "files_total": len(run.results),
        "files_processed": len(ok_results),
        "files_flagged_unprocessable": len(run.results) - len(ok_results),
        "risk_counts": {
            level: sum(1 for r in run.results if r.risk == level)
            for level in ("HIGH", "MEDIUM", "LOW", "UNPROCESSABLE")
        },
        "documents": [
            {
                "document_id": row["document_id"],
                "status": row["processing"]["status"],
                "modality": row["processing"]["modality"],
                "total_integrity_risk": row["total_integrity_risk"],
                **row["scores"],
                "top_peer_match": (row["peer_matches"] + row["multimodal_matches"] or [{}])[0].get("matched_file"),
                "warnings": row["processing"]["warnings"],
            }
            for row in rows
        ],
        "peer_coverage_matrix": {
            "description": "matrix[i][j] = % of document i's assessed words matched in document j",
            "labels": labels,
            "documents": [r.document.doc_id for r in ok_results],
            "values": matrix,
        },
        "boilerplate": run.boilerplate_stats,
        "timings_seconds": {k: round(v, 2) for k, v in run.timings.items()},
        "config": run.config.to_dict(),
    }


def _summary_markdown(run: RunResult, summary: Dict) -> str:
    md = [
        "# Batch integrity summary",
        "",
        f"Generated {summary['generated_at']} · pipeline v{summary['pipeline_version']} · device `{summary['device']}`",
        "",
        f"**{summary['files_total']} files** · {summary['files_processed']} analysed · "
        f"{summary['files_flagged_unprocessable']} flagged as unprocessable · risk counts: "
        + ", ".join(f"{k} {v}" for k, v in summary["risk_counts"].items()),
        "",
        "## Per-file results",
        "",
        "| File | Status | Risk | Peer % | External % | AI % | Top peer match | Report |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def pct(value):
        return "–" if value is None else f"{value:.1f}"

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNPROCESSABLE": 3}
    for doc in sorted(summary["documents"], key=lambda d: (order[d["total_integrity_risk"]], d["document_id"])):
        md.append(
            f"| `{doc['document_id']}` | {doc['status']} | {RISK_BADGE[doc['total_integrity_risk']]} | "
            f"{pct(doc['peer_similarity_percentage'])} | {pct(doc['external_similarity_percentage'])} | "
            f"{pct(doc['ai_generated_probability'])} | {doc['top_peer_match'] or '–'} | "
            f"[md]({_safe_name(doc['document_id'])}.md) |"
        )

    matrix = summary["peer_coverage_matrix"]
    if matrix["labels"]:
        md += [
            "",
            "## Peer similarity matrix",
            "",
            "Cell = % of the **row** file's assessed (non-template) words that match the **column** file. "
            "Asymmetric by design: a short file copied into a long one scores high in its own row only.",
            "",
            "| | " + " | ".join(matrix["labels"]) + " |",
            "|---" * (len(matrix["labels"]) + 1) + "|",
        ]
        for label, values in zip(matrix["labels"], matrix["values"]):
            md.append(f"| **{label}** | " + " | ".join(_matrix_cell(v) for v in values) + " |")
        legend = " · ".join(f"{label} = `{doc}`" for label, doc in zip(matrix["labels"], matrix["documents"]))
        md += ["", "Legend: " + legend]

    flagged = [d for d in summary["documents"] if d["status"] != Status.OK.value]
    if flagged:
        md += ["", "## Files that could not be analysed", ""]
        for d in flagged:
            reason = d["warnings"][-1] if d["warnings"] else ""
            md.append(f"- `{d['document_id']}`: **{d['status']}** – {reason}")

    md += [
        "",
        "## Run details",
        "",
        f"- Boilerplate rule: sentences shared by ≥ {summary['boilerplate']['threshold_documents']} files "
        f"were discounted ({summary['boilerplate']['sentences_flagged']} sentences).",
        "- Timings (s): " + ", ".join(f"{k} {v}" for k, v in summary["timings_seconds"].items()),
        "",
    ]
    return "\n".join(md)
