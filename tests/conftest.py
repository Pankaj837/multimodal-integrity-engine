"""Shared fixtures. Test files are generated on the fly (no binaries committed)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PARAGRAPH = (
    "Kestrel should sign forward contracts for edible oil covering half of its annual requirement. "
    "It should also pilot contract farming for potatoes with two farmer producer organisations. "
    "The board must track gross margin by product family and trade spend as a share of revenue."
)


def make_pdf(path: Path, paragraphs, header: str = None) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

    styles = getSampleStyleSheet()
    story = []
    for item in paragraphs:
        story.append(PageBreak() if item == "<pagebreak>" else Paragraph(item, styles["BodyText"]))

    def decorate(canvas, document):
        if header:
            canvas.drawString(72, A4[1] - 40, header)
            canvas.drawString(72, 30, f"Page {document.page}")

    SimpleDocTemplate(str(path), pagesize=A4).build(story, onFirstPage=decorate, onLaterPages=decorate)
    return path


def make_docx(path: Path, paragraphs, page_break_after: int = None) -> Path:
    import docx

    document = docx.Document()
    for i, text in enumerate(paragraphs):
        document.add_paragraph(text)
        if page_break_after is not None and i == page_break_after:
            document.add_page_break()
    document.save(str(path))
    return path


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def make_video(path: Path, seconds: float = 3.0, audio: str = "none") -> Path:
    """audio: 'none' (no audio stream) or 'silent' (audio stream of digital silence)."""
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=black:s=320x240:d={seconds}"]
    if audio == "silent":
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", str(seconds), "-c:a", "aac"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture(scope="session")
def config():
    from integrity.config import load_config

    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def transcriber(config):
    from integrity.extractors import SpeechTranscriber

    return SpeechTranscriber(config)
