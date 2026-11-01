import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a layout-aware document chunk with exact page citation and table metadata."""

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
    Extracts narrative text and structured Markdown financial tables from SEC PDFs
    while preserving page-level citation integrity and table boundaries.
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, pdf_path: Path, max_pages: int | None = None) -> List[DocumentChunk]:
        """Parses a PDF file and extracts table-enriched chunks with exact page numbers."""
        doc_name = pdf_path.stem
        chunks: List[DocumentChunk] = []

        doc = fitz.open(pdf_path)
        total_pages = len(doc) if max_pages is None else min(len(doc), max_pages)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1

            table_markdown_blocks: List[str] = []
            try:
                tables = page.find_tables()
                for tbl in tables:
                    md_table = tbl.to_markdown()
                    if md_table and len(md_table.strip()) > 25:
                        table_markdown_blocks.append(md_table.strip())
            except Exception:
                pass

            raw_text = page.get_text("text").strip()
            has_table = len(table_markdown_blocks) > 0

            if has_table:
                combined_content = (
                    f"[DOCUMENT: {doc_name} | PAGE: {page_num}]\n"
                    f"{raw_text}\n\n"
                    f"--- EXTRACTED FINANCIAL TABLES (PAGE {page_num}) ---\n"
                    + "\n\n".join(table_markdown_blocks)
                )
            else:
                combined_content = f"[DOCUMENT: {doc_name} | PAGE: {page_num}]\n{raw_text}"

            chunks.extend(
                self._chunk_text_content(
                    text=combined_content,
                    doc_name=doc_name,
                    page_number=page_num,
                    contains_table=has_table,
                )
            )

        doc.close()
        return chunks

    def parse_pdf_bytes(
        self, pdf_bytes: bytes, doc_name: str, max_pages: int | None = None
    ) -> List[DocumentChunk]:
        """Parses an uploaded PDF byte stream in memory and extracts table-enriched chunks."""
        clean_name = doc_name[:-4] if doc_name.lower().endswith(".pdf") else doc_name
        chunks: List[DocumentChunk] = []

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            total_pages = len(doc) if max_pages is None else min(len(doc), max_pages)

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_num = page_idx + 1

                table_markdown_blocks: List[str] = []
                try:
                    tables = page.find_tables()
                    for tbl in tables:
                        md_table = tbl.to_markdown()
                        if md_table and len(md_table.strip()) > 25:
                            table_markdown_blocks.append(md_table.strip())
                except Exception:
                    pass

                raw_text = page.get_text("text").strip()
                has_table = len(table_markdown_blocks) > 0

                if has_table:
                    combined_content = (
                        f"[DOCUMENT: {clean_name} | PAGE: {page_num}]\n"
                        f"{raw_text}\n\n"
                        f"--- EXTRACTED TABLES (PAGE {page_num}) ---\n"
                        + "\n\n".join(table_markdown_blocks)
                    )
                else:
                    combined_content = f"[DOCUMENT: {clean_name} | PAGE: {page_num}]\n{raw_text}"

                chunks.extend(
                    self._chunk_text_content(
                        text=combined_content,
                        doc_name=clean_name,
                        page_number=page_num,
                        contains_table=has_table,
                    )
                )
        finally:
            doc.close()

        return chunks

    def parse_raw_pages(
        self, doc_name: str, pages: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Directly chunks structured in-memory page payloads (used for fast ingestion & tests)."""
        chunks: List[DocumentChunk] = []
        for item in pages:
            page_num = int(item.get("page_number", 1))
            text = str(item.get("text", "")).strip()
            has_table = "|" in text and "---" in text
            formatted = f"[DOCUMENT: {doc_name} | PAGE: {page_num}]\n{text}"
            chunks.extend(
                self._chunk_text_content(
                    text=formatted,
                    doc_name=doc_name,
                    page_number=page_num,
                    contains_table=has_table,
                )
            )
        return chunks

    def _chunk_text_content(
        self,
        text: str,
        doc_name: str,
        page_number: int,
        contains_table: bool,
    ) -> List[DocumentChunk]:
        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(cleaned) < 20:
            return []

        result: List[DocumentChunk] = []
        start = 0
        chunk_index = 0

        while start < len(cleaned):
            end = min(start + self.chunk_size, len(cleaned))
            slice_text = cleaned[start:end]

            chunk_hash = hashlib.md5(
                f"{doc_name}_p{page_number}_c{chunk_index}_{slice_text[:60]}".encode("utf-8")
            ).hexdigest()[:12]

            result.append(
                DocumentChunk(
                    chunk_id=chunk_hash,
                    doc_name=doc_name,
                    page_number=page_number,
                    text=slice_text,
                    contains_table=contains_table,
                    metadata={
                        "source": f"{doc_name}.pdf",
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "has_financial_table": contains_table,
                    },
                )
            )
            if end == len(cleaned):
                break
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        return result
