"""PDF text extraction with page numbers and header/footer removal."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..models import Block, Status
from ..normalize import normalize_text, remove_headers_footers
from .base import ExtractionError, ParagraphCounter, lines_to_blocks


def extract_pdf(path: Path) -> Tuple[List[Block], int, int]:
    """Return (blocks, page_count, removed_header_footer_lines).

    Raises ExtractionError for unreadable, password-protected or text-less PDFs.
    """
    try:
        reader = PdfReader(str(path), strict=False)
    except (PdfReadError, ValueError, OSError) as exc:
        raise ExtractionError(Status.CORRUPT_FILE, f"PDF could not be parsed: {exc}") from exc

    if reader.is_encrypted:
        # Many PDFs are encrypted only with an *owner* password (print/copy
        # restrictions) and open with an empty user password.
        try:
            unlocked = bool(reader.decrypt(""))
        except Exception:  # unsupported cipher or missing crypto backend
            unlocked = False
        if not unlocked:
            raise ExtractionError(Status.PASSWORD_PROTECTED, "PDF is password-protected")

    try:
        pages = [normalize_text(page.extract_text() or "").split("\n") for page in reader.pages]
    except Exception as exc:  # malformed content streams
        raise ExtractionError(Status.CORRUPT_FILE, f"PDF text extraction failed: {exc}") from exc

    if not pages:
        raise ExtractionError(Status.CORRUPT_FILE, "PDF has no pages")

    cleaned, removed = remove_headers_footers(pages)
    counter = ParagraphCounter()
    blocks: List[Block] = []
    for page_number, lines in enumerate(cleaned, start=1):
        blocks.extend(lines_to_blocks(lines, page_number, counter))

    if not any(block.text.strip() for block in blocks):
        raise ExtractionError(
            Status.NO_TEXT,
            "PDF contains no extractable text (likely scanned images; OCR is out of scope)",
        )
    return blocks, len(pages), removed
