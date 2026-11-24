"""Build the synthetic multi-format evaluation dataset and its ground truth.

Creates sample_submissions/ with 15 files (7 PDF, 5 DOCX, 3 MP4) and
dataset_manifest.json describing every planted relationship.

    python scripts/build_dataset.py

Requirements: reportlab (PDF), python-docx (DOCX), pyttsx3 + Windows SAPI /
espeak (offline text-to-speech), Pillow (video title card), imageio-ffmpeg
(bundled ffmpeg binary), PyAV (duration check). Builds are reproducible:
PDFs are rendered in reportlab's invariant mode and DOCX core properties
use a fixed timestamp. TTS audio depends on the voices installed locally,
which is why the generated MP4s are committed to the repository.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import docx  # noqa: E402
from docx.shared import Pt  # noqa: E402
from reportlab import rl_config  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from scripts import dataset_content as content  # noqa: E402

rl_config.invariant = 1  # deterministic PDF bytes (no timestamps / random IDs)
OUT = ROOT / "sample_submissions"
FIXED_TIME = dt.datetime(2026, 11, 20, 12, 0, 0)
MAX_CLIP_SECONDS = 30.0
MIN_CLIP_SECONDS = 10.0


# ------------------------------------------------------------------------ PDF
def _styles():
    sheet = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=sheet["BodyText"], fontName="Helvetica", fontSize=10.5, leading=14.5,
                          spaceAfter=7)
    return {
        "title": ParagraphStyle("T", parent=sheet["Title"], fontSize=16, leading=20, spaceAfter=10),
        "heading": ParagraphStyle("H", parent=sheet["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4),
        "para": body,
        "ref": ParagraphStyle("R", parent=body, leftIndent=18, firstLineIndent=-18, fontSize=9.5, leading=12.5),
    }


def build_pdf(path: Path, items, header: str = None) -> None:
    styles = _styles()
    story = []
    for kind, value in items:
        if kind in ("title", "heading", "para"):
            story.append(Paragraph(value, styles[kind]))
        elif kind == "pagebreak":
            story.append(PageBreak())
        elif kind == "refs":
            story.extend(Paragraph(ref, styles["ref"]) for ref in value)
        elif kind == "bullets":
            story.append(ListFlowable(
                [ListItem(Paragraph(b, styles["para"]), leftIndent=12) for b in value],
                bulletType="bullet", start="•", leftIndent=12,
            ))
        elif kind == "table":
            table = Table(value, hAlign="LEFT", colWidths=[2.4 * cm, 4.2 * cm, 3.2 * cm, 3.0 * cm])
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]))
            story += [Spacer(1, 4), table, Spacer(1, 8)]
        else:
            raise ValueError(f"unknown item kind {kind}")

    def decorate(canvas, document):
        """Running header + "Page X of N"-style footer for the heavily formatted PDF."""
        if not header:
            return
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 8.5)
        canvas.setFillColor(colors.HexColor("#1f3b57"))
        canvas.drawString(2 * cm, A4[1] - 1.3 * cm, header)
        canvas.line(2 * cm, A4[1] - 1.45 * cm, A4[0] - 2 * cm, A4[1] - 1.45 * cm)
        canvas.setFont("Helvetica", 8.5)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {document.page} of 3")
        canvas.drawString(2 * cm, 1.2 * cm, "Kestrel Foods Ltd. - Round 1 submission")
        canvas.restoreState()

    document = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                 topMargin=2.2 * cm, bottomMargin=2 * cm, title=path.stem,
                                 author="Synthetic dataset", creator="build_dataset.py")
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)


# ----------------------------------------------------------------------- DOCX
def build_docx(path: Path, items) -> None:
    document = docx.Document()
    document.styles["Normal"].font.size = Pt(11)
    for kind, value in items:
        if kind == "title":
            document.add_heading(value, level=0)
        elif kind == "heading":
            document.add_heading(value, level=2)
        elif kind == "para":
            document.add_paragraph(value)
        elif kind == "pagebreak":
            document.add_page_break()
        elif kind == "refs":
            for ref in value:
                document.add_paragraph(ref)
        elif kind == "bullets":
            for bullet in value:
                document.add_paragraph(bullet, style="List Bullet")
        elif kind == "table":
            table = document.add_table(rows=len(value), cols=len(value[0]))
            table.style = "Table Grid"
            for r, row in enumerate(value):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell
        else:
            raise ValueError(f"unknown item kind {kind}")
    props = document.core_properties
    props.author = "Synthetic dataset"
    props.created = props.modified = FIXED_TIME
    props.last_modified_by = "build_dataset.py"
    document.save(str(path))


# ---------------------------------------------------------------------- video
def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _title_card(path: Path, lines) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1280, 720), (24, 44, 66))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 54)
        small = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((80, 260), lines[0], fill=(255, 255, 255), font=font)
    for i, line in enumerate(lines[1:]):
        draw.text((80, 350 + 50 * i), line, fill=(190, 210, 230), font=small)
    image.save(path)


def _tts(text: str, wav: Path, voice_hint: str, rate: int) -> None:
    import pyttsx3

    engine = pyttsx3.init()
    for voice in engine.getProperty("voices"):
        if voice_hint.lower() in voice.name.lower():
            engine.setProperty("voice", voice.id)
            break
    engine.setProperty("rate", rate)
    engine.save_to_file(text, str(wav))
    engine.runAndWait()
    engine.stop()


def _duration(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        return container.duration / 1_000_000


def build_speech_video(path: Path, text: str, voice_hint: str, card_lines, work: Path) -> float:
    """TTS -> WAV, still title card -> H.264/AAC MP4. Speeds up speech if > 30 s."""
    card = work / f"{path.stem}.png"
    _title_card(card, card_lines)
    wav = work / f"{path.stem}.wav"
    for rate in (175, 190, 205, 220):
        _tts(text, wav, voice_hint, rate)
        # keep 1 s of padding so the clip does not end mid-word
        if _duration(wav) + 1.0 <= MAX_CLIP_SECONDS:
            break
    # An explicit -t is used instead of -shortest, which does not reliably stop
    # a looped still image and can leave a long silent video tail.
    total = f"{_duration(wav) + 1.0:.2f}"
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-loop", "1", "-framerate", "2", "-i", str(card), "-i", str(wav),
         "-af", "apad", "-t", total, "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "96k", "-ar", "16000", "-movflags", "+faststart", str(path)],
        check=True,
    )
    return _duration(path)


def build_silent_video(path: Path, seconds: float, work: Path) -> float:
    """A video stream only - no audio track at all (Scenario 3)."""
    card = work / f"{path.stem}.png"
    _title_card(card, ["Team Juliet - Pitch", "(recording failed: microphone was muted)"])
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-loop", "1", "-framerate", "2", "-i", str(card),
         "-t", str(seconds), "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-an",
         "-movflags", "+faststart", str(path)],
        check=True,
    )
    return _duration(path)


# ------------------------------------------------------------- ground truth
def _pdf_page_of(path: Path, phrase: str) -> int:
    """1-based page of the first page containing ``phrase`` (whitespace-insensitive)."""
    from pypdf import PdfReader

    needle = " ".join(phrase.split())
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        if needle in " ".join((page.extract_text() or "").split()):
            return number
    raise ValueError(f"phrase not found in {path.name}: {phrase[:40]}")


def _speech_window(path: Path, first_words: str, last_words: str) -> dict:
    """Locate a spoken passage in a clip by transcribing it (word timestamps)."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), word_timestamps=True, vad_filter=True, temperature=0.0)
    words = [w for s in segments for w in s.words]
    tokens = ["".join(ch for ch in w.word.lower() if ch.isalnum()) for w in words]

    def find(phrase):
        target = ["".join(ch for ch in t.lower() if ch.isalnum()) for t in phrase.split()]
        for i in range(len(tokens) - len(target) + 1):
            if tokens[i:i + len(target)] == target:
                return i, i + len(target) - 1
        raise ValueError(f"'{phrase}' not found in transcript of {path.name}")

    start_index, _ = find(first_words)
    _, end_index = find(last_words)
    return {"start_seconds": round(words[start_index].start, 2), "end_seconds": round(words[end_index].end, 2)}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for old in OUT.iterdir():
        if old.is_file():
            old.unlink()

    for name, items in content.DOCUMENTS.items():
        path = OUT / name
        if name.endswith(".pdf"):
            header = content.FORMATTED_HEADER if name == "submission_10.pdf" else None
            build_pdf(path, items, header)
        else:
            build_docx(path, items)
        print(f"[OK] {name}")
    (OUT / "submission_12.pdf").write_bytes(b"")  # zero-byte "PDF" (Scenario 3)
    print("[OK] submission_12.pdf (0 bytes)")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        durations = {
            "pitch_01.mp4": build_speech_video(
                OUT / "pitch_01.mp4", content.PITCH_01_SCRIPT, "David",
                ["Team Kilo - Video Pitch", "Kestrel case"], work),
            "pitch_02.mp4": build_speech_video(
                OUT / "pitch_02.mp4", content.PITCH_02_SCRIPT, "Zira",
                ["Team Lima - Video Pitch", "Kestrel case"], work),
            "pitch_03.mp4": build_silent_video(OUT / "pitch_03.mp4", 12.0, work),
        }
    for name, seconds in durations.items():
        status = "OK" if MIN_CLIP_SECONDS <= seconds <= MAX_CLIP_SECONDS else "OUT OF RANGE"
        print(f"[{status}] {name}: {seconds:.1f}s")
        if status != "OK":
            return 1

    manifest = make_manifest(durations)
    (OUT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] dataset_manifest.json ({len(manifest['files'])} files)")
    return 0


def make_manifest(durations: dict) -> dict:
    copied_page = _pdf_page_of(OUT / "submission_05.pdf", content.ANSWER_C_CHANNEL_PARAGRAPH[:80])
    pitch_window = _speech_window(OUT / "pitch_01.mp4", "We looked at the", "already under pressure")
    wiki_pages = sorted({_pdf_page_of(OUT / "submission_09.pdf", content.WIKI_VERTICAL_P1[:70]),
                         _pdf_page_of(OUT / "submission_09.pdf", content.WIKI_VERTICAL_P3[:70])})
    pitch2_window = _speech_window(OUT / "pitch_02.mp4", "In commerce supply", "point of consumption")
    template_group = ["submission_01.pdf", "submission_02.docx", "submission_03.docx",
                      "submission_05.pdf", "submission_11.docx"]

    def entry(name, label, description, **extra):
        path = OUT / name
        record = {"file": name, "format": path.suffix.lstrip("."), "bytes": path.stat().st_size,
                  "label": label, "description": description}
        if name in durations:
            record["duration_seconds"] = round(durations[name], 1)
        record.update(extra)
        return record

    return {
        "name": "Kestrel Foods synthetic case-competition dataset",
        "version": "1.0",
        "created": "2026-11-20",
        "domain": "Fictional business case competition (packaged-snacks maker 'Kestrel Foods Ltd.')",
        "data_compliance": (
            "Fully synthetic: fictional company, teams and figures. External-copy passages are quoted from pinned "
            "pre-ChatGPT Wikipedia revisions (CC BY-SA 4.0, see reference_corpus/manifest.json). No real student "
            "data, no proprietary or copyrighted case material."
        ),
        "provenance": (
            "All texts were drafted with AI assistance for this assessment. 'ai_generated' = unedited "
            "default-style LLM output from a plain instruction. 'human_style' = written to emulate student "
            "writing. Because human_style files are not genuinely human-authored, AI detection is evaluated "
            "primarily on benchmarks/ai_detection (genuinely human pre-2022 Wikipedia vs LLM text). Speech was "
            "synthesised with offline Windows SAPI voices (David, Zira)."
        ),
        "files": [
            entry("submission_01.pdf", "original_with_template",
                  "Team Alpha answer. Shared case prompt + reference list. SOURCE of the exact copy in "
                  "submission_02.docx.", authorship="human_style"),
            entry("submission_02.docx", "exact_copy",
                  "Team Sigma: verbatim copy of submission_01.pdf (only the team name changed), PDF->DOCX.",
                  authorship="human_style"),
            entry("submission_03.docx", "original_with_template",
                  "Team Bravo answer with the shared prompt. SOURCE of the paraphrase in submission_04.docx.",
                  authorship="human_style"),
            entry("submission_04.docx", "paraphrase",
                  "Team Charlie: same arguments as submission_03.docx reworded with synonyms and restructured "
                  "sentences (Scenario 2). No template.", authorship="human_style"),
            entry("submission_05.pdf", "original_with_template",
                  "Team Delta answer with the shared prompt. Its channel paragraph is read aloud in pitch_01.mp4.",
                  authorship="human_style"),
            entry("submission_06.pdf", "ai_generated",
                  "Answer generated by an LLM from a plain instruction, unedited.", authorship="ai_generated"),
            entry("submission_07.pdf", "original",
                  "Team Echo: independent answer to the same case (hard negative: same topic, no copying).",
                  authorship="human_style"),
            entry("submission_08.docx", "original",
                  "Team Foxtrot: independent answer to a different case (bike-share) - easy negative.",
                  authorship="human_style"),
            entry("submission_09.pdf", "external_copy",
                  "Team Golf: two uncited verbatim paragraphs from Wikipedia 'Vertical integration' and a "
                  "lightly edited sentence from 'Gross margin'.", authorship="human_style"),
            entry("submission_10.pdf", "original_heavily_formatted",
                  "Team Hotel: original answer; running header, 'Page X of 3' footer, table and bullet list on "
                  "3 pages (degraded-formatting test).", authorship="human_style"),
            entry("submission_11.docx", "original_with_template",
                  "Team India: original answer with the shared prompt and reference list.",
                  authorship="human_style"),
            entry("submission_12.pdf", "degraded_empty",
                  "Zero-byte PDF (Scenario 3).", expected_status="EMPTY_FILE"),
            entry("pitch_01.mp4", "cross_modal_copy",
                  "Team Kilo video: short spoken intro, then reads submission_05.pdf's channel paragraph aloud "
                  "(Scenario 1).", authorship="human_style"),
            entry("pitch_02.mp4", "external_copy",
                  "Team Lima video: reads Wikipedia 'Supply chain management' definition aloud, lightly edited "
                  "for speech ('reading an online article verbatim in an MP4').", authorship="human_style"),
            entry("pitch_03.mp4", "degraded_silent",
                  "Video with NO audio track (Scenario 3).", expected_status="NO_AUDIO_TRACK"),
        ],
        "relationships": [
            {"type": "exact_copy", "files": ["submission_01.pdf", "submission_02.docx"], "scenario": "Exact copy",
             "expected": "flag both, verbatim, peer_matches"},
            {"type": "paraphrase", "files": ["submission_03.docx", "submission_04.docx"], "scenario": "Scenario 2",
             "expected": "semantic match (cosine high) despite low Jaccard/Levenshtein"},
            {"type": "cross_modal_copy", "files": ["submission_05.pdf", "pitch_01.mp4"], "scenario": "Scenario 1",
             "source_location": {"file": "submission_05.pdf", "page": copied_page},
             "copy_location": {"file": "pitch_01.mp4", **pitch_window},
             "expected": "multimodal_matches with PDF page and video timestamp"},
            {"type": "external_copy", "files": ["submission_09.pdf"], "scenario": "External",
             "source": ["reference_corpus/vertical_integration.txt", "reference_corpus/gross_margin.txt"],
             "copy_location": {"file": "submission_09.pdf", "pages": wiki_pages},
             "expected": "external_matches citing the Wikipedia articles"},
            {"type": "external_copy", "files": ["pitch_02.mp4"], "scenario": "External (spoken)",
             "source": ["reference_corpus/supply_chain_management.txt"],
             "copy_location": {"file": "pitch_02.mp4", **pitch2_window},
             "expected": "external_matches with video timestamp"},
            {"type": "shared_template", "files": template_group, "scenario": "Scenario 4",
             "template_words": len(content.TEMPLATE_PROMPT.split()),
             "expected": "150-word prompt + reference list discounted as boilerplate; NOT plagiarism between these "
                         "files (except the planted 01-02 exact copy)"},
        ],
        "positive_pairs": [
            ["submission_01.pdf", "submission_02.docx"],
            ["submission_03.docx", "submission_04.docx"],
            ["submission_05.pdf", "pitch_01.mp4"],
        ],
        "external_positives": ["submission_09.pdf", "pitch_02.mp4"],
        "ai_generated": ["submission_06.pdf"],
        "unprocessable": {"submission_12.pdf": "EMPTY_FILE", "pitch_03.mp4": "NO_AUDIO_TRACK"},
    }


if __name__ == "__main__":
    sys.exit(main())
