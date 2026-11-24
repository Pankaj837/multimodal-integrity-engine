"""Format detection and dispatch to the right extractor.

The file type is decided from the file's *content* (magic bytes), not only its
extension, so a PDF renamed to ``.docx`` is still read correctly and a random
binary renamed to ``.pdf`` is reported as corrupt instead of crashing a parser.
Every failure is converted into a :class:`Document` with a non-OK status and a
human-readable warning, so one bad file never stops the batch.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Optional

from ..models import Document, Modality, Status
from ..normalize import normalize_text
from .base import ExtractionError, ParagraphCounter, lines_to_blocks
from .docx_extractor import extract_docx
from .media_extractor import SpeechTranscriber
from .pdf_extractor import extract_pdf

log = logging.getLogger(__name__)

TEXT_TYPES = {"pdf", "docx", "txt"}
MEDIA_TYPES = {"mp4", "m4v", "mov", "avi", "m4a", "mkv", "webm", "mp3", "wav", "flac", "ogg"}
SUPPORTED_TYPES = TEXT_TYPES | MEDIA_TYPES

__all__ = ["ExtractionError", "SpeechTranscriber", "detect_file_type", "extract_document", "SUPPORTED_TYPES"]


def detect_file_type(path: Path) -> Optional[str]:
    """Identify a file from its leading bytes. Returns None when unknown."""
    with open(path, "rb") as handle:
        head = handle.read(2048)
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile:
            return "zip"
        if "word/document.xml" in names:
            return "docx"
        if any(n.startswith("ppt/") for n in names):
            return "pptx"
        if any(n.startswith("xl/") for n in names):
            return "xlsx"
        return "zip"
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole"  # legacy .doc / .ppt / .xls
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand.startswith(b"M4A"):
            return "m4a"
        if brand == b"qt  ":
            return "mov"
        return "mp4"
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "wav"
    if head.startswith(b"RIFF") and head[8:12] == b"AVI ":
        return "avi"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if head.startswith(b"fLaC"):
        return "flac"
    if head.startswith(b"OggS"):
        return "ogg"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "mkv"
    return None


def _extract_txt(path: Path, document: Document) -> None:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExtractionError(Status.CORRUPT_FILE, "Text file is not valid UTF-8") from exc
    if "\x00" in text:
        raise ExtractionError(Status.CORRUPT_FILE, "File looks binary, not plain text")
    document.blocks = lines_to_blocks(normalize_text(text).split("\n"), None, ParagraphCounter())
    if not any(b.text.strip() for b in document.blocks):
        raise ExtractionError(Status.NO_TEXT, "Text file contains only whitespace")


def extract_document(path: Path, doc_id: str, transcriber: SpeechTranscriber) -> Document:
    """Extract one file into a Document. Never raises for bad input files."""
    extension = path.suffix.lower().lstrip(".")
    document = Document(doc_id=doc_id, path=str(path), file_type=extension or "unknown")

    try:
        if path.stat().st_size == 0:
            raise ExtractionError(Status.EMPTY_FILE, "File is empty (0 bytes); nothing to analyse.")

        sniffed = detect_file_type(path)
        if sniffed in ("ole",):
            raise ExtractionError(
                Status.UNSUPPORTED_TYPE,
                "Legacy binary Office format (.doc/.ppt/.xls); please convert to .docx/.pdf.",
            )
        if sniffed is not None and sniffed not in SUPPORTED_TYPES:
            raise ExtractionError(Status.UNSUPPORTED_TYPE, f"Unsupported file type '{sniffed}'.")
        if sniffed is None:
            if extension in ("pdf", "docx") or extension in MEDIA_TYPES:
                raise ExtractionError(
                    Status.CORRUPT_FILE,
                    f"Content does not match the .{extension} format (file is corrupt or mislabelled).",
                )
            if extension != "txt":
                raise ExtractionError(Status.UNSUPPORTED_TYPE, f"Unsupported file type '.{extension}'.")
            sniffed = "txt"
        same_family = sniffed in MEDIA_TYPES and extension in MEDIA_TYPES
        if sniffed != extension and not same_family:
            document.warnings.append(
                f"Extension '.{extension}' does not match the file content ({sniffed}); processed as {sniffed}."
            )
        document.file_type = sniffed

        if sniffed == "pdf":
            document.blocks, document.n_pages, document.removed_header_footer_lines = extract_pdf(path)
        elif sniffed == "docx":
            document.blocks, document.n_pages = extract_docx(path)
        elif sniffed == "txt":
            _extract_txt(path, document)
        else:
            document.modality = Modality.SPEECH
            blocks, duration, language, warnings = transcriber.transcribe(path)
            document.blocks, document.duration_seconds, document.language = blocks, duration, language
            document.warnings.extend(warnings)
    except ExtractionError as exc:
        document.status = exc.status
        document.warnings.append(str(exc))
        log.warning("%s: %s - %s", doc_id, exc.status.value, exc)
    except Exception as exc:  # defensive: an unexpected bug must not stop the batch
        document.status = Status.ERROR
        document.warnings.append(f"Unexpected error during extraction: {type(exc).__name__}: {exc}")
        log.exception("%s: unexpected extraction failure", doc_id)

    if document.file_type in MEDIA_TYPES:
        document.modality = Modality.SPEECH
    return document
