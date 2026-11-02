"""PDF parsing, section-aware chunking, and deterministic PDF rendering."""

import hashlib
import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List

import pymupdf as fitz

from src.domain.documents import DocumentChunk
from src.domain.ocr import OCRBackend
from src.domain.sections import SECTION_HEADER_REGEX

logger = logging.getLogger(__name__)

TABLE_DIVIDER_REGEX = re.compile(r"^\|[\s\-:|]+\|$")
TABLE_MANIFEST_PREFIX = "VECTRACITE_MANIFEST:"
PARSER_CACHE_VERSION = 4

# Structural defaults. Deployment values are supplied by the composition root; these keep the parser
# usable standalone (tests, scripts) without importing configuration.
DEFAULT_OCR_DPI = 200
DEFAULT_OCR_MIN_CHARS = 30
DEFAULT_OCR_MAX_PAGES = 50
DEFAULT_MAX_RENDERED_PAGES = 5000


class PDFRenderError(ValueError):
    """Raised when structured pages cannot be rendered faithfully."""


class PDFRenderLimitError(PDFRenderError):
    """Raised when structured input exceeds the configured render limit."""


class MultimodalFinancialParser:
    """
    Parse filings, CVs, academic documents, trials, and technical PDFs.

    The parser treats a PDF as untrusted input: malformed streams, password
    protection, and image-only pages receive explicit user-facing errors.
    Generated PDFs carry a compact table manifest in metadata so a render and
    reload cannot silently lose table cells.
    """

    def __init__(
        self,
        chunk_size: int = 1400,
        chunk_overlap: int = 150,
        *,
        ocr_backend: OCRBackend | None = None,
        ocr_dpi: int = DEFAULT_OCR_DPI,
        ocr_min_chars: int = DEFAULT_OCR_MIN_CHARS,
        ocr_max_pages: int = DEFAULT_OCR_MAX_PAGES,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_backend = ocr_backend
        self.ocr_dpi = ocr_dpi
        self.ocr_min_chars = ocr_min_chars
        self.ocr_max_pages = ocr_max_pages

    def parse_pdf(self, pdf_path: Path, max_pages: int | None = None) -> List[DocumentChunk]:
        """Parse a PDF from disk in strict reading order."""
        if not pdf_path.exists():
            raise ValueError(f"PDF file not found: {pdf_path}")
        return self.parse_pdf_bytes(
            pdf_path.read_bytes(),
            doc_name=pdf_path.stem,
            max_pages=max_pages,
        )

    def parse_pdf_bytes(
        self, pdf_bytes: bytes, doc_name: str, max_pages: int | None = None,
        *, coverage: list[dict[str, Any]] | None = None,
        cancel_event: threading.Event | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> List[DocumentChunk]:
        """Parse a PDF byte stream and recover native and manifest-backed tables."""
        if not pdf_bytes or len(pdf_bytes) < 8:
            raise ValueError("Empty or truncated PDF file uploaded.")
        if not pdf_bytes.lstrip().startswith(b"%PDF-"):
            raise ValueError("Invalid or corrupted PDF file: missing PDF header.")

        clean_name = doc_name[:-4] if doc_name.lower().endswith(".pdf") else doc_name
        try:
            document = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted PDF file: {exc}") from exc

        try:
            if document.is_encrypted and not document.authenticate(""):
                raise ValueError(
                    "PDF is password-protected/encrypted. Please unlock the PDF before uploading."
                )
            if len(document) == 0:
                raise ValueError("PDF contains 0 pages.")

            manifest_tables = self._load_manifest(document)
            total_pages = len(document) if max_pages is None else min(len(document), max_pages)
            chunks: List[DocumentChunk] = []
            total_images_seen = 0
            ocr_pages = 0
            blank_image_pages = 0

            for page_index in range(total_pages):
                if cancel_event is not None and cancel_event.is_set():
                    raise ValueError("Document processing cancelled.")
                page = document[page_index]
                total_images_seen += len(page.get_images())
                try:
                    blank_image = self._is_blank_image_page(page)
                    blank_image_pages += int(blank_image)
                    ocr_attempted = (
                        not blank_image
                        and self.ocr_backend is not None
                        and len(page.get_text("text", sort=True).strip()) < self.ocr_min_chars
                        and bool(page.get_images())
                    )
                    page_chunks = [] if blank_image else self._parse_page(
                        page=page,
                        doc_name=clean_name,
                        physical_page_index=page_index,
                        manifest_tables=manifest_tables,
                        allow_ocr=ocr_pages < self.ocr_max_pages,
                    )
                    if ocr_attempted:
                        ocr_pages += 1
                    if coverage is not None:
                        has_images = bool(page.get_images())
                        ocr_used = any(chunk.metadata.get("ocr_used") for chunk in page_chunks)
                        native_chars = len(page.get_text("text", sort=True).strip())
                        status = (
                            "blank_image" if blank_image else
                            "ocr_text" if ocr_used else
                            "native_text" if page_chunks else
                            "image_without_text" if has_images else "no_text"
                        )
                        coverage.append({
                            "page": page_index + 1,
                            "status": status,
                            "native_characters": native_chars,
                            "contains_images": has_images,
                        })
                    chunks.extend(page_chunks)
                    if progress is not None:
                        progress(page_index + 1, total_pages)
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError(f"Invalid or corrupted PDF file: {exc}") from exc
        finally:
            document.close()

        if not chunks:
            if total_pages and blank_image_pages == total_pages:
                raise ValueError("PDF contains only blank image pages.")
            if total_images_seen > 0:
                raise ValueError(
                    "Scanned or image-only PDF detected (no extractable text layer found). "
                    "Please run OCR on the PDF before uploading."
                )
            raise ValueError("No readable text or tables found in PDF.")
        return chunks

    def _is_blank_image_page(self, page: fitz.Page) -> bool:
        if page.get_text("text", sort=True).strip() or not page.get_images():
            return False
        pixmap = page.get_pixmap(dpi=self.ocr_dpi, colorspace=fitz.csGRAY, alpha=False)
        return bool(pixmap.samples) and min(pixmap.samples) == 255

    @staticmethod
    def _load_manifest(document: fitz.Document) -> Dict[str, List[str]]:
        raw_keywords = (document.metadata or {}).get("keywords", "") or ""
        if not raw_keywords.startswith(TABLE_MANIFEST_PREFIX):
            return {}
        try:
            decoded = json.loads(raw_keywords[len(TABLE_MANIFEST_PREFIX) :])
            return {str(page): list(blocks) for page, blocks in decoded.items()}
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring malformed VectraCite table manifest")
            return {}

    def _parse_page(
        self,
        page: fitz.Page,
        doc_name: str,
        physical_page_index: int,
        manifest_tables: Dict[str, List[str]] | None = None,
        *,
        allow_ocr: bool = True,
    ) -> List[DocumentChunk]:
        raw_text = page.get_text("text", sort=True).strip()
        page_number = physical_page_index + 1

        ocr_used = False
        ocr_engine = None
        if self.ocr_backend is not None and len(raw_text) < self.ocr_min_chars and page.get_images():
            if not allow_ocr:
                raise ValueError(f"OCR page limit reached at page {physical_page_index + 1}.")
            try:
                pixmap = page.get_pixmap(dpi=self.ocr_dpi, alpha=False)
                ocr_result = self.ocr_backend.recognize(pixmap.tobytes("png"))
            except Exception as exc:
                logger.warning(
                    "OCR failed for %s page %s: %s",
                    doc_name,
                    page_number,
                    type(exc).__name__,
                )
                raise ValueError(f"OCR failed on page {physical_page_index + 1}.") from exc
            else:
                ocr_text = ocr_result.text.strip()
                if not ocr_text:
                    raise ValueError(f"OCR returned no text on page {physical_page_index + 1}.")
                raw_text = ocr_text
                ocr_used = True
                ocr_engine = ocr_result.engine

        manifest_blocks = [
            block
            for key, blocks in (manifest_tables or {}).items()
            if key.endswith(f":{physical_page_index}")
            for block in blocks
        ]
        legacy_label = re.search(r"\(SOURCE PAGE:\s*(\d+)\)", raw_text.split("\n", 1)[0])
        if not manifest_blocks and legacy_label:
            manifest_blocks = (manifest_tables or {}).get(legacy_label.group(1), [])
        table_markdown_blocks = self._extract_table_blocks(
            page=page,
            raw_text=raw_text,
            manifest_tables=manifest_blocks,
        )
        has_inline_markdown_table = self._has_inline_markdown_table(raw_text)
        contains_table = bool(table_markdown_blocks) or has_inline_markdown_table
        document_header = f"[DOCUMENT: {doc_name} | PAGE: {page_number}]"

        if table_markdown_blocks and not has_inline_markdown_table:
            combined_content = (
                f"{document_header}\n{raw_text}\n\n"
                f"EXTRACTED TABLES (PAGE {page_number})\n"
                + "\n\n".join(table_markdown_blocks)
            )
        else:
            combined_content = f"{document_header}\n{raw_text}"

        return self._chunk_text_content(
            text=combined_content,
            doc_name=doc_name,
            page_number=page_number,
            contains_table=contains_table,
            ocr_used=ocr_used,
            ocr_engine=ocr_engine,
        )

    @classmethod
    def _extract_table_blocks(
        cls,
        page: fitz.Page,
        raw_text: str,
        manifest_tables: List[str],
    ) -> List[str]:
        """Combine lossless metadata, native tables, and aligned text tables."""
        blocks: List[str] = []
        seen: set[str] = set()

        def add_block(block: str) -> None:
            normalized = re.sub(r"\s+", "", block).casefold()
            if normalized and normalized not in seen and len(block.strip()) > 20:
                blocks.append(block.strip())
                seen.add(normalized)

        for block in manifest_tables:
            add_block(str(block))

        if not manifest_tables:
            try:
                for table in page.find_tables().tables:
                    markdown = table.to_markdown()
                    if markdown:
                        add_block(markdown)
            except Exception as exc:
                logger.debug("Native table detection unavailable: %s", exc)

        for block in cls._extract_pipe_tables(raw_text):
            add_block(block)
        if not blocks:
            for block in cls._extract_whitespace_tables(raw_text):
                add_block(block)
        return blocks

    @staticmethod
    def _has_inline_markdown_table(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines()]
        return any(
            line.startswith("|")
            and line.endswith("|")
            and TABLE_DIVIDER_REGEX.fullmatch(line)
            for line in lines
        )

    @staticmethod
    def _extract_pipe_tables(text: str) -> List[str]:
        groups: List[List[str]] = []
        current: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            is_table_line = line.startswith("|") and line.endswith("|")
            if is_table_line:
                current.append(line)
            elif current:
                if any(TABLE_DIVIDER_REGEX.fullmatch(item) for item in current):
                    groups.append(current)
                current = []
        if current and any(TABLE_DIVIDER_REGEX.fullmatch(item) for item in current):
            groups.append(current)
        return ["\n".join(group) for group in groups]

    @staticmethod
    def _extract_whitespace_tables(text: str) -> List[str]:
        """Recover simple column-aligned tables from text-only PDF extraction."""
        candidates: List[tuple[int, List[str]]] = []
        for line_number, raw_line in enumerate(text.splitlines()):
            line = raw_line.strip()
            cells = [cell.strip() for cell in re.split(r"\s{2,}", line) if cell.strip()]
            if len(cells) < 2 or not any(
                re.fullmatch(r"[$€£]?\s*[-(]?\d[\d,.\s%)]*", cell)
                for cell in cells[1:]
            ):
                continue
            candidates.append((line_number, cells))

        groups: List[List[List[str]]] = []
        current: List[List[str]] = []
        previous_line = -2
        for line_number, cells in candidates:
            if line_number - previous_line > 2 and current:
                if len(current) >= 2:
                    groups.append(current)
                current = []
            current.append(cells)
            previous_line = line_number
        if len(current) >= 2:
            groups.append(current)

        blocks: List[str] = []
        for rows in groups:
            column_count = max(len(row) for row in rows)
            if column_count < 2:
                continue
            normalized_rows = [
                row + [""] * (column_count - len(row)) for row in rows
            ]
            header = normalized_rows[0]
            markdown = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join("---" for _ in header) + " |",
            ]
            markdown.extend("| " + " | ".join(row) + " |" for row in normalized_rows[1:])
            blocks.append("\n".join(markdown))
        return blocks

    def parse_raw_pages(
        self, doc_name: str, pages: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Chunk structured in-memory pages through the same PDF round-trip as uploads."""
        clean_name = doc_name[:-4] if doc_name.lower().endswith(".pdf") else doc_name
        pdf_bytes = self.render_pages_to_pdf_bytes(clean_name, pages)
        return self.parse_pdf_bytes(pdf_bytes, doc_name=clean_name)

    @staticmethod
    def _is_section_header(line: str) -> bool:
        return (
            3 <= len(line) <= 55
            and not line.startswith("|")
            and not line.startswith(("•", "-", "*"))
            and bool(SECTION_HEADER_REGEX.fullmatch(line))
        )

    def _split_oversized_block(self, block: str) -> List[str]:
        """Split an oversized block at sentence and word boundaries."""
        if len(block) <= self.chunk_size or ("|" in block and "---" in block):
            return [block]
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n", block)
            if sentence.strip()
        ]
        parts: List[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                parts.append(current)
            if len(sentence) <= self.chunk_size:
                current = sentence
                continue
            words = sentence.split()
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip() if current else word
                if current and len(candidate) > self.chunk_size:
                    parts.append(current)
                    current = word
                else:
                    current = candidate
        if current:
            parts.append(current)
        return parts or [block]

    def _chunk_text_content(
        self,
        text: str,
        doc_name: str,
        page_number: int,
        contains_table: bool,
        *,
        ocr_used: bool = False,
        ocr_engine: str | None = None,
    ) -> List[DocumentChunk]:
        """Split on sections/paragraphs and never cut a word or sentence by index."""
        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
        document_header = f"[DOCUMENT: {doc_name} | PAGE: {page_number}]"
        body = cleaned[len(document_header) :].strip() if cleaned.startswith(document_header) else cleaned
        if len(body) < 5:
            return []

        blocks: List[str] = []
        current_block: List[str] = []

        def flush_block() -> None:
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block.clear()

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                flush_block()
                continue
            if self._is_section_header(line) and current_block:
                flush_block()
            current_block.append(line)
        flush_block()

        raw_chunks: List[str] = []
        current_chunk = ""
        for block in blocks:
            for piece in self._split_oversized_block(block):
                candidate = f"{current_chunk}\n\n{piece}".strip() if current_chunk else piece
                if current_chunk and len(candidate) > self.chunk_size:
                    raw_chunks.append(current_chunk)
                    current_chunk = piece
                else:
                    current_chunk = candidate
        if current_chunk:
            raw_chunks.append(current_chunk)

        result: List[DocumentChunk] = []
        for chunk_index, chunk_body in enumerate(raw_chunks):
            full_text = f"{document_header}\n{chunk_body.strip()}"
            chunk_contains_table = contains_table and "|" in chunk_body and "---" in chunk_body
            chunk_id = hashlib.md5(
                f"{doc_name}_p{page_number}_c{chunk_index}_{full_text}".encode("utf-8")
            ).hexdigest()[:12]
            result.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_name=doc_name,
                    page_number=page_number,
                    text=full_text,
                    contains_table=chunk_contains_table,
                     metadata={
                        "source": f"{doc_name}.pdf",
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "has_financial_table": chunk_contains_table,
                        "ocr_used": ocr_used,
                        "ocr_engine": ocr_engine,
                    },
                )
            )
        return result

    @staticmethod
    def _markdown_cells(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    @staticmethod
    def render_pages_to_pdf_bytes(
        doc_name: str,
        pages: List[Dict[str, Any]],
        *,
        max_pages: int = DEFAULT_MAX_RENDERED_PAGES,
    ) -> bytes:
        """Render tables and prose with physical-page pagination and no clipping."""
        if len(pages) > max_pages:
            raise PDFRenderLimitError("Document page limit exceeded.")
        document = fitz.open()
        manifest: Dict[str, List[str]] = {}
        physical_pages = 0
        prose_font_size = 9.5
        prose_line_height = 16.0
        body_top = 66.0
        body_bottom = 750.0
        prose_chars_per_line = 88

        def new_page(source_page: int) -> tuple[fitz.Page, float]:
            nonlocal physical_pages
            physical_pages += 1
            if physical_pages > max_pages:
                raise PDFRenderLimitError("Rendered PDF page limit exceeded.")
            page = document.new_page(width=612, height=792)
            page.draw_rect(
                fitz.Rect(36, 26, 576, 48),
                color=(0.78, 0.82, 0.88),
                fill=(0.95, 0.97, 0.99),
                width=0.7,
            )
            page.insert_text(
                (44, 41),
                f"{doc_name} - Document Page {source_page} (SOURCE PAGE: {source_page})",
                fontsize=8.5,
                fontname="helv",
                color=(0.28, 0.33, 0.41),
            )
            return page, body_top

        try:
            for item in pages:
                source_page = int(item.get("page_number", 1))
                text = str(item.get("text", "")).strip()
                page, y_cursor = new_page(source_page)
                prose_lines: List[str] = []
                table_rows: List[List[str]] = []
                raw_table_lines: List[str] = []

                def ensure_space(height: float) -> None:
                    nonlocal page, y_cursor
                    if y_cursor + height <= body_bottom:
                        return
                    page, y_cursor = new_page(source_page)

                def write_prose_line(line: str) -> None:
                    nonlocal y_cursor
                    if not line:
                        y_cursor += 4.0
                        return
                    wrapped_lines = max(
                        1, math.ceil(len(line) / prose_chars_per_line)
                    )
                    height = wrapped_lines * prose_line_height
                    ensure_space(height)
                    result = page.insert_textbox(
                        fitz.Rect(40, y_cursor, 572, y_cursor + height),
                        line,
                        fontsize=prose_font_size,
                        fontname="helv",
                        color=(0.06, 0.09, 0.16),
                    )
                    if result is not None and result < 0:
                        raise PDFRenderError(
                            "Structured prose could not be rendered without clipping."
                        )
                    y_cursor += height + 2.0

                def flush_prose() -> None:
                    nonlocal y_cursor
                    if not prose_lines:
                        return
                    pending = list(prose_lines)
                    prose_lines.clear()
                    for line in pending:
                        write_prose_line(line)
                    y_cursor += 6.0

                def flush_table() -> None:
                    nonlocal y_cursor
                    if len(table_rows) < 2:
                        prose_lines.extend("  ".join(row) for row in table_rows)
                        table_rows.clear()
                        raw_table_lines.clear()
                        flush_prose()
                        return

                    raw_manifest_table = "\n".join(raw_table_lines)
                    raw_table_lines.clear()
                    column_count = max(len(row) for row in table_rows)
                    column_width = 532.0 / max(column_count, 1)
                    for row_index, row in enumerate(table_rows):
                        cell_font_size = 8.2
                        chars_per_cell = max(
                            8, int(column_width / (cell_font_size * 0.52))
                        )
                        row_height = max(
                            24.0,
                            max(
                                math.ceil(
                                    len(row[index]) / chars_per_cell
                                )
                                * 10.0
                                for index in range(column_count)
                                if index < len(row)
                            )
                            if any(row)
                            else 24.0,
                        )
                        ensure_space(row_height)
                        if row_index == 0:
                            manifest.setdefault(
                                f"{source_page}:{physical_pages - 1}", []
                            ).append(raw_manifest_table)
                        for column_index in range(column_count):
                            cell_text = row[column_index] if column_index < len(row) else ""
                            x0 = 40.0 + column_index * column_width
                            x1 = x0 + column_width
                            cell_rect = fitz.Rect(
                                x0, y_cursor, x1, y_cursor + row_height
                            )
                            page.draw_rect(
                                cell_rect,
                                color=(0.68, 0.73, 0.80),
                                fill=(0.90, 0.94, 0.98),
                                width=0.7,
                            )
                            result = page.insert_textbox(
                                fitz.Rect(
                                    x0 + 5,
                                    y_cursor + 4,
                                    x1 - 4,
                                    y_cursor + row_height - 3,
                                ),
                                cell_text,
                                fontsize=cell_font_size,
                                fontname="helv",
                                color=(0.06, 0.09, 0.16),
                            )
                            if result is not None and result < 0:
                                raise PDFRenderError(
                                    "Structured table cell could not be rendered."
                                )
                        y_cursor += row_height
                    table_rows.clear()
                    y_cursor = min(body_bottom, y_cursor + 12.0)

                for raw_line in text.splitlines():
                    line = raw_line.strip()
                    if line.startswith("|") and line.endswith("|"):
                        raw_table_lines.append(line)
                        if TABLE_DIVIDER_REGEX.fullmatch(line):
                            continue
                        flush_prose()
                        table_rows.append(MultimodalFinancialParser._markdown_cells(line))
                    else:
                        if table_rows:
                            flush_table()
                        if line:
                            prose_lines.append(line)
                        else:
                            prose_lines.append("")

                if table_rows:
                    flush_table()
                flush_prose()

            if manifest:
                document.set_metadata(
                    {
                        "keywords": TABLE_MANIFEST_PREFIX
                        + json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
                    }
                )
            return document.tobytes(garbage=4, deflate=True)
        finally:
            document.close()


def render_pages_to_pdf_bytes(
    doc_name: str,
    pages: List[Dict[str, Any]],
    *,
    max_pages: int = DEFAULT_MAX_RENDERED_PAGES,
) -> bytes:
    """Backward-compatible module-level rendering helper."""
    return MultimodalFinancialParser.render_pages_to_pdf_bytes(
        doc_name, pages, max_pages=max_pages
    )
