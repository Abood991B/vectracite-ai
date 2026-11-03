"""Render a PDF page to PNG with query-term highlights.

Owns the rendering decision (DPI, highlight cap, ETag derivation) and the bounded page cache. The
controller only decides status codes and headers.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

HIGHLIGHT_DPI = 150
MAX_HIGHLIGHTS = 8
DEFAULT_CACHE_LIMIT = 128

STOPWORDS = {
    "about",
    "according",
    "and",
    "compared",
    "from",
    "into",
    "that",
    "the",
    "their",
    "what",
    "were",
    "with",
}
TERM_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9%$.-]{2,}")


class PageOutOfRange(ValueError):
    """Raised when the requested page does not exist in the document."""

    def __init__(self, doc_name: str, page_number: int, page_count: int) -> None:
        self.page_number = page_number
        self.page_count = page_count
        super().__init__(
            f"Page {page_number} is outside '{doc_name}' ({page_count} pages)."
        )


class InvalidPdf(ValueError):
    """Raised when bytes do not form a readable PDF."""


def count_pdf_pages(raw_bytes: bytes) -> int:
    """Return the page count of an in-memory PDF, or raise ``InvalidPdf``."""
    try:
        with fitz.open(stream=raw_bytes, filetype="pdf") as document:
            return document.page_count
    except Exception as exc:
        raise InvalidPdf("Uploaded content is not a valid PDF.") from exc


@dataclass(frozen=True)
class RenderedPage:
    """A rendered page image and the ETag that identifies its bytes."""

    png: bytes
    etag: str


def _highlight_terms(highlight: str) -> list[str]:
    terms: list[str] = []
    for term in TERM_PATTERN.findall(highlight):
        normalized = term.casefold()
        if normalized not in STOPWORDS and normalized not in terms:
            terms.append(normalized)
    return terms


class PageImageRenderer:
    """Render pages on demand and keep a bounded cache of recent images."""

    def __init__(self, cache_limit: int = DEFAULT_CACHE_LIMIT) -> None:
        self._cache: "OrderedDict[str, RenderedPage]" = OrderedDict()
        self._cache_limit = cache_limit

    def render(self, pdf_path: Path, doc_name: str, page_number: int, highlight: str = "") -> RenderedPage:
        """Return the PNG for one page, reusing a cached render when the file is unchanged."""
        stat = pdf_path.stat()
        cache_key = (
            f"{doc_name}:{page_number}:{highlight.strip().casefold()}:"
            f"{stat.st_mtime_ns}:{stat.st_size}"
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        document = fitz.open(pdf_path)
        try:
            if page_number > len(document):
                raise PageOutOfRange(doc_name, page_number, len(document))
            page = document[page_number - 1]
            self._apply_highlights(page, highlight)
            png = page.get_pixmap(dpi=HIGHLIGHT_DPI, alpha=False).tobytes("png")
        finally:
            document.close()

        rendered = RenderedPage(
            png=png,
            etag='"' + hashlib.sha256(png).hexdigest()[:32] + '"',
        )
        self._cache[cache_key] = rendered
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return rendered

    @staticmethod
    def _apply_highlights(page: fitz.Page, highlight: str) -> None:
        applied = 0
        for term in _highlight_terms(highlight):
            for rect in page.search_for(term):
                page.add_highlight_annot(rect).update()
                applied += 1
                if applied >= MAX_HIGHLIGHTS:
                    return
