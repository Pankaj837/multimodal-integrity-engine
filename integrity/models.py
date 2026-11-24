"""Plain data containers passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Status(str, Enum):
    """Processing outcome for one input file. Only ``OK`` documents are scored."""

    OK = "OK"
    EMPTY_FILE = "EMPTY_FILE"  # zero bytes on disk
    CORRUPT_FILE = "CORRUPT_FILE"  # parser could not read the container
    PASSWORD_PROTECTED = "PASSWORD_PROTECTED"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    NO_TEXT = "NO_TEXT"  # e.g. a scanned, image-only PDF (OCR is out of scope)
    NO_AUDIO_TRACK = "NO_AUDIO_TRACK"  # video container without an audio stream
    NO_SPEECH = "NO_SPEECH"  # audio present but silent / no recognisable speech
    ERROR = "ERROR"  # unexpected failure; the batch still continues


class Modality(str, Enum):
    TEXT = "text"  # PDF / DOCX
    SPEECH = "speech"  # transcript of an audio/video file


class SentenceKind(str, Enum):
    TEXT = "text"  # prose sentence: eligible as evidence
    HEADING = "heading"  # titles, section headings, short labels
    TABLE = "table"  # table rows / short cells
    CITATION = "citation"  # bibliography entries: never counted as plagiarism


@dataclass
class Block:
    """A unit of extracted text with its source location.

    For PDF/DOCX a block is a paragraph, heading or table row. For speech a
    block is one transcribed sentence carrying start/end times.
    """

    text: str
    kind: SentenceKind = SentenceKind.TEXT
    page: Optional[int] = None
    paragraph: Optional[int] = None
    start: Optional[float] = None  # seconds (speech only)
    end: Optional[float] = None
    is_sentence: bool = False  # True when already sentence-segmented (ASR output)


@dataclass
class Sentence:
    doc_index: int
    index: int  # position within the document
    text: str
    n_words: int
    kind: SentenceKind
    page: Optional[int] = None
    paragraph: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    eligible: bool = False  # prose and long enough to be evidence
    boilerplate: bool = False  # shared template text (set by boilerplate stage)
    boilerplate_docs: int = 0  # how many documents share it

    @property
    def assessed(self) -> bool:
        """Counts towards similarity percentages."""
        return self.eligible and not self.boilerplate


@dataclass
class Chunk:
    doc_index: int
    index: int
    sentence_ids: List[int]  # indices into the document's sentence list
    text: str
    n_tokens: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class Document:
    doc_id: str  # file name relative to the input folder
    path: str
    file_type: str  # pdf | docx | mp4 | ...
    modality: Modality = Modality.TEXT
    status: Status = Status.OK
    warnings: List[str] = field(default_factory=list)
    extraction_warnings: List[str] = field(default_factory=list)  # snapshot after ingestion
    blocks: List[Block] = field(default_factory=list)
    sentences: List[Sentence] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    n_pages: Optional[int] = None
    duration_seconds: Optional[float] = None
    language: Optional[str] = None
    removed_header_footer_lines: int = 0

    @property
    def ok(self) -> bool:
        return self.status == Status.OK

    @property
    def assessed_words(self) -> int:
        return sum(s.n_words for s in self.sentences if s.assessed)
