"""Build the local external-reference knowledge base from Wikipedia.

The brief allows simulating web search with "a local reference knowledge base
(e.g., a set of Wikipedia articles)". This script downloads a fixed set of
articles and pins each one to the last revision saved *before* 2022-11-01
(ChatGPT was released on 2022-11-30). Pinning gives us two properties:

1. Reproducibility - the same revision IDs always produce the same text.
2. A human-written control set - pre-ChatGPT Wikipedia prose is used as the
   "human" class when benchmarking the AI-text detector (see
   scripts/build_ai_benchmark.py).

Output:
    reference_corpus/<slug>.txt      one paragraph per block, blank-line separated
    reference_corpus/manifest.json   title, URL, revision id, timestamp, licence

Wikipedia text is licensed CC BY-SA 4.0; attribution is kept in manifest.json.

Usage:
    python scripts/fetch_reference_corpus.py
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "integrity-engine-assessment/1.0 (educational prototype)"
CUTOFF = "2022-11-01T00:00:00Z"  # pre-ChatGPT revisions only
MIN_PARAGRAPH_WORDS = 25

ARTICLES = [
    "Vertical integration",
    "Supply chain management",
    "Direct-to-consumer",
    "Gross margin",
    "Porter's five forces analysis",
    "SWOT analysis",
    "Snack food",
    "Marketing mix",
    "Inflation",
    "E-commerce",
    "Brand management",
    "Economies of scale",
    "Market segmentation",
    "Pricing strategies",
]

OUT_DIR = Path(__file__).resolve().parent.parent / "reference_corpus"


def _get(params: dict, retries: int = 5) -> dict:
    """Call the MediaWiki API and return parsed JSON.

    Retries with exponential back-off on HTTP 429 / 5xx, honouring the
    Retry-After header when the server sends one.
    """
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500 or attempt == retries - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 ** (attempt + 1)
            time.sleep(min(delay, 60.0))
    raise RuntimeError("unreachable")


class _ParagraphExtractor(HTMLParser):
    """Collect the text of top-level <p> elements, skipping tables,
    footnote markers (<sup>), and inline styles."""

    _SKIP = {"table", "sup", "style", "script"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._skip_depth = 0
        self._in_p = False
        self._buffer: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "p" and self._skip_depth == 0:
            self._in_p = True
            self._buffer = []

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "p" and self._in_p:
            self._in_p = False
            self.paragraphs.append("".join(self._buffer))

    def handle_data(self, data):
        if self._in_p and self._skip_depth == 0:
            self._buffer.append(data)


def _clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\[(?:\d+|[a-z]|citation needed|note \d+)\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def fetch_article(title: str) -> dict:
    """Return metadata and cleaned paragraphs for the pinned revision."""
    revision_info = _get(
        {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "redirects": 1,
            "rvlimit": 1,
            "rvstart": CUTOFF,
            "rvdir": "older",
            "rvprop": "ids|timestamp",
        }
    )
    page = revision_info["query"]["pages"][0]
    if "missing" in page or "revisions" not in page:
        raise ValueError(f"No pre-cutoff revision for '{title}'")
    revision = page["revisions"][0]

    parsed = _get({"action": "parse", "oldid": revision["revid"], "prop": "text"})
    extractor = _ParagraphExtractor()
    extractor.feed(parsed["parse"]["text"])
    paragraphs = [_clean(p) for p in extractor.paragraphs]
    paragraphs = [p for p in paragraphs if len(p.split()) >= MIN_PARAGRAPH_WORDS]

    resolved_title = page["title"]
    return {
        "title": resolved_title,
        "file": f"{_slug(resolved_title)}.txt",
        "url": "https://en.wikipedia.org/w/index.php?oldid=" + str(revision["revid"]),
        "revision_id": revision["revid"],
        "revision_timestamp": revision["timestamp"],
        "licence": "CC BY-SA 4.0 (Wikipedia contributors)",
        "paragraphs": paragraphs,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    force = "--force" in sys.argv
    existing = {}
    if manifest_path.exists() and not force:
        for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
            if (OUT_DIR / entry["file"]).exists():
                existing[entry["requested_title"]] = entry

    manifest = []
    for title in ARTICLES:
        if title in existing:
            manifest.append(existing[title])
            print(f"[SKIP] {title}: already downloaded (use --force to refresh)")
            continue
        try:
            article = fetch_article(title)
        except Exception as exc:  # network or API error: report and continue
            print(f"[WARN] {title}: {exc}", file=sys.stderr)
            continue
        paragraphs = article.pop("paragraphs")
        article = {"requested_title": title, **article}
        (OUT_DIR / article["file"]).write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
        article["paragraph_count"] = len(paragraphs)
        article["word_count"] = sum(len(p.split()) for p in paragraphs)
        manifest.append(article)
        print(f"[OK] {article['title']}: rev {article['revision_id']} "
              f"({article['revision_timestamp']}), {len(paragraphs)} paragraphs")
        time.sleep(2.0)  # be polite to the public API

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} articles to {OUT_DIR}")
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
