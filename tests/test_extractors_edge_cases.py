"""Malformed and unusual inputs must be flagged with a clear status, never crash."""

import zipfile

import pytest
from conftest import PARAGRAPH, make_docx, make_pdf, make_video

from integrity.extractors import detect_file_type, extract_document
from integrity.models import Modality, Status


def _extract(path, transcriber):
    return extract_document(path, path.name, transcriber)


def test_valid_pdf_with_pages_and_header_removal(tmp_path, transcriber):
    path = make_pdf(tmp_path / "ok.pdf", [PARAGRAPH, "<pagebreak>", PARAGRAPH], header="CASE 2026 | TEAM Z")
    document = _extract(path, transcriber)
    assert document.status == Status.OK and document.n_pages == 2
    assert {b.page for b in document.blocks} == {1, 2}
    assert not any("CASE 2026" in b.text for b in document.blocks)
    assert document.removed_header_footer_lines >= 2


def test_docx_page_numbers_follow_explicit_breaks(tmp_path, transcriber):
    path = make_docx(tmp_path / "ok.docx", [PARAGRAPH, PARAGRAPH], page_break_after=0)
    document = _extract(path, transcriber)
    assert document.status == Status.OK
    assert [b.page for b in document.blocks] == [1, 2]


def test_zero_byte_file(tmp_path, transcriber):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    document = _extract(path, transcriber)
    assert document.status == Status.EMPTY_FILE and document.warnings


def test_random_bytes_named_pdf_is_corrupt(tmp_path, transcriber):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"\x00\x01garbage" * 100)
    assert _extract(path, transcriber).status == Status.CORRUPT_FILE


def test_truncated_pdf_is_corrupt_or_textless(tmp_path, transcriber):
    good = make_pdf(tmp_path / "good.pdf", [PARAGRAPH])
    path = tmp_path / "truncated.pdf"
    path.write_bytes(good.read_bytes()[:300])
    assert _extract(path, transcriber).status in (Status.CORRUPT_FILE, Status.NO_TEXT)


def test_password_protected_pdf(tmp_path, transcriber):
    from pypdf import PdfReader, PdfWriter

    source = make_pdf(tmp_path / "plain.pdf", [PARAGRAPH])
    writer = PdfWriter()
    for page in PdfReader(str(source)).pages:
        writer.add_page(page)
    writer.encrypt(user_password="secret", owner_password="owner", algorithm="RC4-128")
    path = tmp_path / "locked.pdf"
    with open(path, "wb") as handle:
        writer.write(handle)
    assert _extract(path, transcriber).status == Status.PASSWORD_PROTECTED


def test_image_only_pdf_reports_no_text(tmp_path, transcriber):
    from reportlab.pdfgen import canvas

    path = tmp_path / "scan.pdf"
    page = canvas.Canvas(str(path))
    page.rect(100, 100, 300, 300, fill=1)  # graphics only, like a scanned page
    page.save()
    assert _extract(path, transcriber).status == Status.NO_TEXT


def test_corrupt_docx(tmp_path, transcriber):
    path = tmp_path / "bad.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("hello.txt", "not a word document")
    assert _extract(path, transcriber).status in (Status.CORRUPT_FILE, Status.UNSUPPORTED_TYPE)


def test_pdf_with_wrong_extension_is_read_by_content(tmp_path, transcriber):
    source = make_pdf(tmp_path / "real.pdf", [PARAGRAPH])
    path = tmp_path / "actually_pdf.docx"
    path.write_bytes(source.read_bytes())
    document = _extract(path, transcriber)
    assert document.status == Status.OK and document.file_type == "pdf"
    assert any("does not match" in w for w in document.warnings)


def test_unsupported_and_plain_text_files(tmp_path, transcriber):
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert _extract(image, transcriber).status == Status.UNSUPPORTED_TYPE
    text = tmp_path / "notes.txt"
    text.write_text(PARAGRAPH, encoding="utf-8")
    assert _extract(text, transcriber).status == Status.OK


def test_magic_byte_detection(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", [PARAGRAPH])
    docx_path = make_docx(tmp_path / "a.docx", [PARAGRAPH])
    assert detect_file_type(pdf) == "pdf"
    assert detect_file_type(docx_path) == "docx"


def test_video_without_audio_track(tmp_path, transcriber):
    path = make_video(tmp_path / "mute.mp4", audio="none")
    document = _extract(path, transcriber)
    assert document.status == Status.NO_AUDIO_TRACK
    assert document.modality == Modality.SPEECH


@pytest.mark.slow
def test_video_with_silent_audio_track_does_not_hallucinate(tmp_path, transcriber):
    """Whisper is known to invent "Thank you." on silence; VAD must stop that."""
    path = make_video(tmp_path / "silent.mp4", seconds=4.0, audio="silent")
    document = _extract(path, transcriber)
    assert document.status == Status.NO_SPEECH
    assert not document.blocks
