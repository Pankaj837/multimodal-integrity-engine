"""Text normalisation and header/footer removal.

Normalisation is deliberately conservative: it repairs extraction artefacts
(ligatures, soft hyphens, broken line-end hyphenation, irregular whitespace)
but never rewrites wording, because reports must quote passages verbatim.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import List, Tuple

_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)
# C0/C1 control characters except tab and newline (e.g. bullet glyphs that a
# PDF font maps to U+007F).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_SPACES = re.compile(r"[ \t\f\v ]+")
_PAGE_NUMBER = re.compile(r"^(page\s*)?#+(\s*(of|/)\s*#+)?$")
_ZONE_LINES = 3  # how many lines at the top/bottom of a page may be header/footer
_MAX_HEADER_WORDS = 15  # running headers/footers are short labels, not prose lines


def normalize_text(text: str) -> str:
    """Unicode-normalise and tidy whitespace while keeping line breaks.

    NFKC folds compatibility characters - most importantly PDF ligatures such
    as U+FB01 ("fi") - into plain letters so that "ﬁle" and "file" compare equal.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text.translate(_INVISIBLE).translate(_QUOTES))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub("", text)  # "compet-\nitive" -> "competitive"
    text = _SPACES.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n"))


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _line_key(line: str) -> str:
    """Comparable form of a line: lower-case, digits masked, spaces collapsed.

    Masking digits lets "Page 1 of 3" and "Page 2 of 3" share the same key.
    """
    return re.sub(r"\d+", "#", collapse_whitespace(line).lower())


def remove_headers_footers(pages: List[List[str]]) -> Tuple[List[List[str]], int]:
    """Drop running headers/footers and bare page numbers.

    A short line (<= 15 words) in the top or bottom ``_ZONE_LINES`` of a page
    is treated as a header/footer when its digit-masked form appears on at
    least half of the pages (minimum two). Bare page-number lines ("3",
    "Page 3 of 10") in those zones are removed on any page. If the rule would
    delete more than half of a page, only its first/last line and bare page
    numbers are removed there.
    Returns the cleaned pages and the number of lines removed.
    """
    n_pages = len(pages)

    def zone_indices(lines: List[str]) -> List[int]:
        non_empty = [i for i, line in enumerate(lines) if line.strip()]
        return sorted(set(non_empty[:_ZONE_LINES] + non_empty[-_ZONE_LINES:]))

    counts: Counter = Counter()
    for lines in pages:
        counts.update({_line_key(lines[i]) for i in zone_indices(lines)})

    min_pages = max(2, math.ceil(0.5 * n_pages))
    repeated = {
        key for key, c in counts.items()
        if c >= min_pages and n_pages >= 2 and len(key.split()) <= _MAX_HEADER_WORDS
    }

    cleaned, removed = [], 0
    for lines in pages:
        zone = set(zone_indices(lines))
        drop = {i for i in zone if _line_key(lines[i]) in repeated or _PAGE_NUMBER.match(_line_key(lines[i]))}
        non_empty = [i for i, line in enumerate(lines) if line.strip()]
        if non_empty and len(drop) > 0.5 * len(non_empty):
            # Safety valve: running headers/footers never make up most of a page
            # (a short page whose body repeats on other pages must not vanish).
            # Keep only the outermost lines - where headers/footers live - and
            # bare page numbers.
            edges = {non_empty[0], non_empty[-1]}
            drop = {i for i in drop if i in edges or _PAGE_NUMBER.match(_line_key(lines[i]))}
        cleaned.append([line for i, line in enumerate(lines) if i not in drop])
        removed += len(drop)
    return cleaned, removed
