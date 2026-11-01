"""PDF parsing, section-aware chunking, and deterministic PDF rendering."""

import hashlib
import json
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
TABLE_MANIFEST_PREFIX = "VECTRACITE_MANIFEST:"


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
    """
    Parse filings, CVs, academic documents, trials, and technical PDFs.

    The parser treats a PDF as untrusted input: malformed streams, password
    protection, and image-only pages receive explicit user-facing errors.
    Generated PDFs carry a compact table manifest in metadata so a render and
    reload cannot silently lose table cells.
    """

    def __init__(self, chunk_size: int = 1400, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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
        self, pdf_bytes: bytes, doc_name: str, max_pages: int | None = None
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

            for page_index in range(total_pages):
                page = document[page_index]
                total_images_seen += len(page.get_images())
                try:
                    chunks.extend(
                        self._parse_page(
                            page=page,
                            doc_name=clean_name,
                            physical_page_index=page_index,
                            manifest_tables=manifest_tables,
                        )
                    )
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError(f"Invalid or corrupted PDF file: {exc}") from exc
        finally:
            document.close()

        if not chunks:
            if total_images_seen > 0:
                raise ValueError(
                    "Scanned or image-only PDF detected (no extractable text layer found). "
                    "Please run OCR on the PDF before uploading."
                )
            raise ValueError("No readable text or tables found in PDF.")
        return chunks

    @staticmethod
    def _load_manifest(document: fitz.Document) -> Dict[int, List[str]]:
        raw_keywords = (document.metadata or {}).get("keywords", "") or ""
        if not raw_keywords.startswith(TABLE_MANIFEST_PREFIX):
            return {}
        try:
            decoded = json.loads(raw_keywords[len(TABLE_MANIFEST_PREFIX) :])
            return {int(page): list(blocks) for page, blocks in decoded.items()}
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring malformed VectraCite table manifest")
            return {}

    def _parse_page(
        self,
        page: fitz.Page,
        doc_name: str,
        physical_page_index: int,
        manifest_tables: Dict[int, List[str]] | None = None,
    ) -> List[DocumentChunk]:
        raw_text = page.get_text("text", sort=True).strip()
        page_match = re.search(r"SOURCE PAGE:\s*(\d+)", raw_text, flags=re.IGNORECASE)
        page_number = int(page_match.group(1)) if page_match else physical_page_index + 1
        raw_text = re.sub(
            r"^.*?\(SOURCE PAGE:\s*\d+\)\s*",
            "",
            raw_text,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        raw_text = re.sub(
            r"OFFICIAL FILING:.*?SOURCE PAGE:\s*\d+\s*",
            "",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        table_markdown_blocks = self._extract_table_blocks(
            page=page,
            raw_text=raw_text,
            manifest_tables=(manifest_tables or {}).get(page_number, []),
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
            if len(cells) < 2 or not any(re.search(r"\d", cell) for cell in cells[1:]):
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
        """Render vector tables and store their exact Markdown grids in PDF metadata."""
        document = fitz.open()
        manifest: Dict[str, List[str]] = {}
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
                raw_table_lines: List[str] = []

                def flush_prose() -> None:
                    nonlocal y_cursor
                    if not prose_lines:
                        return
                    joined = "\n".join(prose_lines)
                    prose_lines.clear()
                    estimated_lines = sum(
                        max(1, len(line) // 88 + 1) for line in joined.splitlines()
                    )
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
                        raw_table_lines.clear()
                        flush_prose()
                        return

                    manifest.setdefault(str(page_number), []).append(
                        "\n".join(raw_table_lines)
                    )
                    raw_table_lines.clear()
                    column_count = max(len(row) for row in table_rows)
                    column_width = 532.0 / max(column_count, 1)
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


def render_pages_to_pdf_bytes(doc_name: str, pages: List[Dict[str, Any]]) -> bytes:
    """Backward-compatible module-level rendering helper."""
    return MultimodalFinancialParser.render_pages_to_pdf_bytes(doc_name, pages)
