"""DOCX text extraction: body paragraphs and tables, in document order.

Notes
-----
* Headers/footers live in separate package parts and are not part of the
  body, so they are excluded automatically.
* ``python-docx``'s ``Document.paragraphs`` skips tables; we walk the body
  XML directly so table text is not lost and reading order is preserved.
* A .docx file stores no layout, so "page" cannot be known exactly. We count
  explicit page breaks (``<w:br w:type="page">`` / ``pageBreakBefore``) and,
  if a file has none, the ``<w:lastRenderedPageBreak>`` markers Word writes
  when it saves. Paragraph numbers are always reported as well.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from zipfile import BadZipFile

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..models import Block, SentenceKind, Status
from ..normalize import normalize_text
from ..segmentation import count_words
from .base import ExtractionError, ParagraphCounter, classify_short_line

_W_P, _W_TBL, _W_T = qn("w:p"), qn("w:tbl"), qn("w:t")
_W_BR, _W_TYPE, _W_LRPB = qn("w:br"), qn("w:type"), qn("w:lastRenderedPageBreak")
_W_PAGE_BREAK_BEFORE, _W_VAL = qn("w:pageBreakBefore"), qn("w:val")
_PROSE_CELL_WORDS = 12


def _page_breaks(p_element, use_rendered: bool) -> Tuple[int, int]:
    """Count page breaks before and after the first visible text of a paragraph."""
    before = after = 0
    seen_text = False
    ppr = p_element.find(qn("w:pPr"))
    if ppr is not None:
        pbb = ppr.find(_W_PAGE_BREAK_BEFORE)
        if pbb is not None and pbb.get(_W_VAL, "true").lower() not in ("false", "0", "off"):
            before += 1
    for element in p_element.iter():
        if element.tag == _W_T and element.text and element.text.strip():
            seen_text = True
        elif (element.tag == _W_BR and element.get(_W_TYPE) == "page") or (
            use_rendered and element.tag == _W_LRPB
        ):
            if seen_text:
                after += 1
            else:
                before += 1
    return before, after


def _paragraph_kind(paragraph: Paragraph, text: str) -> SentenceKind:
    try:
        style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
    except (KeyError, AttributeError):
        style = ""
    if style.startswith(("heading", "title", "subtitle")):
        return SentenceKind.HEADING
    return classify_short_line(text) or SentenceKind.TEXT


def extract_docx(path: Path) -> Tuple[List[Block], int]:
    """Return (blocks, page_count). Raises ExtractionError on unreadable files."""
    try:
        document = docx.Document(str(path))
    except BadZipFile as exc:
        raise ExtractionError(Status.CORRUPT_FILE, "DOCX is not a valid ZIP package") from exc
    except Exception as exc:  # python-docx raises several unrelated types
        raise ExtractionError(Status.CORRUPT_FILE, f"DOCX could not be opened: {exc}") from exc

    body = document.element.body
    use_rendered = not body.xpath('.//w:br[@w:type="page"]') and not body.xpath(".//w:pageBreakBefore")
    counter = ParagraphCounter()
    blocks: List[Block] = []
    page = 1

    for child in body.iterchildren():
        if child.tag == _W_P:
            paragraph = Paragraph(child, document)
            before, after = _page_breaks(child, use_rendered)
            text = normalize_text(paragraph.text).strip()
            if text and blocks:
                page += before
            elif not text:
                after += before  # an empty paragraph that only carries a break
            if text:
                blocks.append(Block(text, _paragraph_kind(paragraph, text), page=page, paragraph=counter.next()))
            page += after
        elif child.tag == _W_TBL:
            blocks.extend(_table_blocks(Table(child, document), page, counter))

    if not any(count_words(b.text) for b in blocks):
        raise ExtractionError(Status.NO_TEXT, "DOCX contains no text")
    return blocks, page


def _table_blocks(table: Table, page: int, counter: ParagraphCounter) -> List[Block]:
    """Prose cells become TEXT blocks; short cells of a row become one TABLE block."""
    blocks: List[Block] = []
    for row in table.rows:
        seen, short_cells = set(), []
        for cell in row.cells:
            if id(cell._tc) in seen:  # merged cells repeat the same element
                continue
            seen.add(id(cell._tc))
            text = normalize_text(cell.text).strip()
            if not text:
                continue
            if count_words(text) >= _PROSE_CELL_WORDS and classify_short_line(text) is None:
                blocks.append(Block(text, SentenceKind.TEXT, page=page, paragraph=counter.next()))
            else:
                short_cells.append(" ".join(text.split()))
        if short_cells:
            blocks.append(Block(" | ".join(short_cells), SentenceKind.TABLE, page=page, paragraph=counter.next()))
    return blocks
