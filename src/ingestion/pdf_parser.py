"""PDF parsing, section-aware chunking, and deterministic PDF rendering."""

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import fitz

logger = logging.getLogger(__name__)

SECTION_HEADER_REGEX = re.compile(
    r"^(?:PROFESSIONAL\s+SUMMARY|SUMMARY|EXPERIENCE|WORK\s+EXPERIENCE|"
    r"TECHNICAL\s+SKILLS|SKILLS|PROJECTS|EDUCATION|CERTIFICATIONS|AWARDS|"
    r"LANGUAGES|PUBLICATIONS|OBJECTIVE|PROFILE|ITEM\s+\d+[A-Z]?\.?|NOTE\s+\d+|"
    r"PART\s+[IVX]+|[A-Z][A-Z0-9\s,&\-/]{3,45})$"
)
TABLE_DIVIDER_REGEX = re.compile(r"^\|[\s\-:|]+\|$")


@dataclass
class DocumentChunk:
    """A layout- and section-aware document chunk with an exact page citation."""

    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    contains_table: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultimodalFinancialParser:
    """Parse filings, CVs, academic documents, trials, and technical PDFs."""

    def __init__(self, chunk_size: int = 1400, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, pdf_path: Path, max_pages: int | None = None) -> List[DocumentChunk]:
        """Parse a PDF from disk in strict reading order."""
        return self.parse_pdf_bytes(
            pdf_path.read_bytes(),
            doc_name=pdf_path.stem,
            max_pages=max_pages,
        )

    def parse_pdf_bytes(
        self, pdf_bytes: bytes, doc_name: str, max_pages: int | None = None
    ) -> List[DocumentChunk]:
        """Parse a PDF byte stream and recover native tables when available."""
        clean_name = doc_name[:-4] if doc_name.lower().endswith(".pdf") else doc_name
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            total_pages = len(document) if max_pages is None else min(len(document), max_pages)
            chunks: List[DocumentChunk] = []
            for page_index in range(total_pages):
                chunks.extend(self._parse_page(document[page_index], clean_name, page_index))
            return chunks
        finally:
            document.close()

    def _parse_page(
        self, page: fitz.Page, doc_name: str, physical_page_index: int
    ) -> List[DocumentChunk]:
        raw_text = page.get_text("text", sort=True).strip()
        page_match = re.search(r"SOURCE PAGE:\s*(\d+)", raw_text, flags=re.IGNORECASE)
        page_number = int(page_match.group(1)) if page_match else physical_page_index + 1
        raw_text = re.sub(
            r"OFFICIAL FILING:.*?SOURCE PAGE:\s*\d+\s*",
            "",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        table_markdown_blocks: List[str] = []
        try:
            table_finder = page.find_tables()
            for table in table_finder.tables:
                markdown = table.to_markdown()
                if markdown and len(markdown.strip()) > 20:
                    table_markdown_blocks.append(markdown.strip())
        except Exception as exc:
            logger.debug("Native table detection unavailable: %s", exc)

        has_inline_markdown_table = "|" in raw_text and "---" in raw_text
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
        )

    def parse_raw_pages(
        self, doc_name: str, pages: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Chunk structured in-memory pages without losing section boundaries."""
        chunks: List[DocumentChunk] = []
        for item in pages:
            page_number = int(item.get("page_number", 1))
            text = str(item.get("text", "")).strip()
            contains_table = "|" in text and "---" in text
            formatted = f"[DOCUMENT: {doc_name} | PAGE: {page_number}]\n{text}"
            chunks.extend(
                self._chunk_text_content(
                    text=formatted,
                    doc_name=doc_name,
                    page_number=page_number,
                    contains_table=contains_table,
                )
            )
        return chunks

    @staticmethod
    def _is_section_header(line: str) -> bool:
        return (
            3 <= len(line) <= 55
            and not line.startswith("|")
            and not line.startswith(("•", "-", "*"))
            and bool(SECTION_HEADER_REGEX.fullmatch(line))
        )

    def _split_long_block(self, block: str) -> List[str]:
        """Split an oversized block at sentences, then words as a last resort."""
        if len(block) <= self.chunk_size:
            return [block]

        units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+|\n", block) if unit.strip()]
        if not units:
            return [block]

        pieces: List[str] = []
        current = ""
        for unit in units:
            candidate = f"{current} {unit}".strip() if current else unit
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                pieces.append(current)
            if len(unit) <= self.chunk_size:
                current = unit
                continue

            words: List[str] = []
            word_length = 0
            for word in unit.split():
                next_length = word_length + len(word) + (1 if words else 0)
                if words and next_length > self.chunk_size:
                    pieces.append(" ".join(words))
                    words = []
                    word_length = 0
                words.append(word)
                word_length += len(word) + (1 if len(words) > 1 else 0)
            current = " ".join(words)

        if current:
            pieces.append(current)
        return pieces

    def _chunk_text_content(
        self,
        text: str,
        doc_name: str,
        page_number: int,
        contains_table: bool,
    ) -> List[DocumentChunk]:
        """Split on sections/paragraphs and never cut a word or sentence by index."""
        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(cleaned) < 15:
            return []

        document_header = f"[DOCUMENT: {doc_name} | PAGE: {page_number}]"
        body = cleaned[len(document_header) :].strip() if cleaned.startswith(document_header) else cleaned
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
            for piece in self._split_long_block(block):
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
                f"{doc_name}_p{page_number}_c{chunk_index}_{full_text[:100]}".encode("utf-8")
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
                    },
                )
            )
        return result

    @staticmethod
    def _markdown_cells(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    @staticmethod
    def render_pages_to_pdf_bytes(doc_name: str, pages: List[Dict[str, Any]]) -> bytes:
        """Render pages with vector-drawn, shaded, bordered tables."""
        document = fitz.open()
        try:
            for item in pages:
                page_number = int(item.get("page_number", 1))
                text = str(item.get("text", "")).strip()
                page = document.new_page(width=612, height=792)
                page.draw_rect(
                    fitz.Rect(36, 26, 576, 48),
                    color=(0.78, 0.82, 0.88),
                    fill=(0.95, 0.97, 0.99),
                    width=0.7,
                )
                page.insert_text(
                    (44, 41),
                    f"{doc_name} - Document Page {page_number} (SOURCE PAGE: {page_number})",
                    fontsize=8.5,
                    fontname="helv",
                    color=(0.28, 0.33, 0.41),
                )

                y_cursor = 66.0
                prose_lines: List[str] = []
                table_rows: List[List[str]] = []

                def flush_prose() -> None:
                    nonlocal y_cursor
                    if not prose_lines:
                        return
                    joined = "\n".join(prose_lines)
                    prose_lines.clear()
                    estimated_lines = sum(max(1, len(line) // 88 + 1) for line in joined.splitlines())
                    box_height = min(660.0, max(28.0, estimated_lines * 14.5))
                    page.insert_textbox(
                        fitz.Rect(40, y_cursor, 572, min(750.0, y_cursor + box_height)),
                        joined,
                        fontsize=9.5,
                        fontname="helv",
                        color=(0.06, 0.09, 0.16),
                    )
                    y_cursor = min(750.0, y_cursor + box_height + 8.0)

                def flush_table() -> None:
                    nonlocal y_cursor
                    if len(table_rows) < 2:
                        prose_lines.extend("  ".join(row) for row in table_rows)
                        table_rows.clear()
                        flush_prose()
                        return

                    column_count = max(len(row) for row in table_rows)
                    table_width = 532.0
                    column_width = table_width / max(column_count, 1)
                    row_height = 24.0
                    for row_index, row in enumerate(table_rows):
                        if y_cursor + row_height > 750.0:
                            break
                        for column_index in range(column_count):
                            cell_text = row[column_index] if column_index < len(row) else ""
                            x0 = 40.0 + column_index * column_width
                            x1 = x0 + column_width
                            cell_rect = fitz.Rect(x0, y_cursor, x1, y_cursor + row_height)
                            page.draw_rect(
                                cell_rect,
                                color=(0.68, 0.73, 0.80),
                                fill=(0.90, 0.94, 0.98) if row_index == 0 else (1, 1, 1),
                                width=0.7,
                            )
                            page.insert_textbox(
                                fitz.Rect(x0 + 5, y_cursor + 4, x1 - 4, y_cursor + row_height - 3),
                                cell_text,
                                fontsize=8.2 if row_index else 8.5,
                                fontname="hebo" if row_index == 0 else "helv",
                                color=(0.06, 0.09, 0.16),
                            )
                        y_cursor += row_height
                    table_rows.clear()
                    y_cursor = min(750.0, y_cursor + 12.0)

                for raw_line in text.splitlines():
                    line = raw_line.strip()
                    if line.startswith("|") and line.endswith("|"):
                        if TABLE_DIVIDER_REGEX.fullmatch(line):
                            continue
                        flush_prose()
                        table_rows.append(MultimodalFinancialParser._markdown_cells(line))
                    else:
                        if table_rows:
                            flush_table()
                        if line:
                            prose_lines.append(line)

                if table_rows:
                    flush_table()
                flush_prose()

            return document.tobytes(garbage=4, deflate=True)
        finally:
            document.close()


def render_pages_to_pdf_bytes(doc_name: str, pages: List[Dict[str, Any]]) -> bytes:
    """Backward-compatible module-level rendering helper."""
    return MultimodalFinancialParser.render_pages_to_pdf_bytes(doc_name, pages)
