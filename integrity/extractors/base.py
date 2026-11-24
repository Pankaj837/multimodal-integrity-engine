"""Shared helpers for the format-specific extractors."""

from __future__ import annotations

import re
from typing import List, Optional

from ..models import Block, SentenceKind, Status
from ..segmentation import count_words


class ExtractionError(Exception):
    """Raised by an extractor when a file cannot be turned into text.

    Carries the :class:`Status` that should be reported for the file; the
    pipeline catches it, records a warning and moves on to the next file.
    """

    def __init__(self, status: Status, message: str):
        super().__init__(message)
        self.status = status


_TERMINAL = re.compile(r"[.!?:;,][\"')\]]?$")
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?$")
_BULLET = re.compile(r"^([•▪●◦–*-]|\(?\d{1,2}[.)]|[a-z][.)])\s+")
_NUMERIC = re.compile(r"\d")
MAX_LABEL_WORDS = 12


def classify_short_line(text: str) -> Optional[SentenceKind]:
    """Return HEADING/TABLE for a stand-alone short line, or None for prose.

    A line of at most 12 words with no terminal punctuation is a heading,
    label or table row. Table rows are recognised by containing several
    numbers or an explicit cell separator.
    """
    if count_words(text) > MAX_LABEL_WORDS or _TERMINAL.search(text):
        return None
    numeric_tokens = sum(1 for tok in text.split() if _NUMERIC.search(tok))
    if " | " in text or numeric_tokens >= 2:
        return SentenceKind.TABLE
    return SentenceKind.HEADING


class ParagraphCounter:
    def __init__(self) -> None:
        self.value = 0

    def next(self) -> int:
        self.value += 1
        return self.value


def lines_to_blocks(lines: List[str], page: Optional[int], counter: ParagraphCounter) -> List[Block]:
    """Rebuild paragraphs from the visual lines of one page.

    * Blank lines end a paragraph.
    * Short lines without terminal punctuation are headings / table rows.
    * Bullet markers start a new paragraph.
    * A line ending a sentence that is clearly shorter than a typical line is
      the last line of a paragraph.
    """
    # "Full-width" line length of this page. A high percentile (not the median)
    # is used because pages dominated by short table rows or bullets would
    # otherwise make every normal text line look "short".
    lengths = sorted(len(line.strip()) for line in lines if line.strip())
    typical = lengths[min(len(lengths) - 1, int(0.9 * len(lengths)))] if lengths else 0
    blocks: List[Block] = []
    buffer: List[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(Block(" ".join(buffer), SentenceKind.TEXT, page=page, paragraph=counter.next()))
            buffer.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        # A heading / label / table row must also be visually short: a
        # full-width line is always running text, even if it has few (long,
        # hyphenated) words and no punctuation at the wrap point.
        visually_short = typical == 0 or len(line) < 0.75 * typical
        label_kind = classify_short_line(line) if visually_short else None
        if label_kind is not None and not buffer:
            blocks.append(Block(line, label_kind, page=page, paragraph=counter.next()))
            continue
        if (
            label_kind is not None
            and buffer
            and not _SENTENCE_END.search(buffer[-1])
            and len(buffer[-1]) >= 0.75 * typical
        ):
            # A short unpunctuated line right after a full-width line that ends
            # mid-sentence is that sentence's wrapped tail, not a heading.
            buffer.append(line)
            flush()
            continue
        if label_kind is not None:
            flush()
            blocks.append(Block(line, label_kind, page=page, paragraph=counter.next()))
            continue
        if _BULLET.match(line):
            flush()
        buffer.append(line)
        if _SENTENCE_END.search(line) and len(line) < 0.75 * typical:
            flush()
    flush()
    return blocks
