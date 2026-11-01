"""Local, citation-grounded answer synthesis for indexed PDF chunks."""

import re
from collections import defaultdict
from typing import AsyncGenerator, Dict, Iterable, List

from src.config import settings
from src.ingestion.pdf_parser import SECTION_HEADER_REGEX
from src.retrieval.hybrid_retriever import RetrievedDocument

QUERY_STOPWORDS = {
    "a",
    "about",
    "according",
    "an",
    "and",
    "are",
    "at",
    "by",
    "compared",
    "did",
    "does",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "overview",
    "tell",
    "the",
    "their",
    "this",
    "to",
    "what",
    "was",
    "were",
    "which",
    "with",
}
META_QUESTION_RE = re.compile(
    r"\b(what model|which model|who are you|what are you|how do you work|"
    r"how does (?:this|it) work|what is this|what llm|privacy|is my data private|"
    r"local execution)\b",
    flags=re.IGNORECASE,
)
DOCUMENTS_QUESTION_RE = re.compile(
    r"\b(what documents|which documents|what files|which files).*\b"
    r"(loaded|indexed|available)\b",
    flags=re.IGNORECASE,
)
SECTION_ALIASES: Dict[str, tuple[str, ...]] = {
    "summary": ("professional summary", "executive summary", "summary", "profile", "overview"),
    "skills": ("technical skills", "skills", "core competencies", "technologies"),
    "experience": ("work experience", "experience", "employment"),
    "projects": ("key projects", "projects"),
    "education": ("education", "academic"),
    "certifications": ("certifications", "certificates", "awards"),
}
META_RESPONSE = (
    "VectraCite-AI runs 100% locally on your machine for complete enterprise data privacy:\n"
    "- Stage 1 (Semantic + Lexical Search): BAAI/bge-small-en-v1.5 (local PyTorch embeddings) + BM25Okapi.\n"
    "- Stage 2 (2D Table & Section Reranking): BAAI/bge-reranker-base with Adaptive SCAR Table-AST coordinate verification.\n"
    "- Stage 3 (Grounded Synthesis & Numeric Audit): extracts verified sections, table coordinates, and YoY deltas while checking numbers against source pages.\n"
    "Zero confidential PDF bytes leave localhost."
)


class UnifiedLLMClient:
    """Structured local synthesizer; no document content is sent to a remote API."""

    def __init__(
        self,
        base_url: str = settings.LLM_BASE_URL,
        api_key: str = settings.LLM_API_KEY,
        model_name: str = settings.LLM_MODEL_NAME,
    ) -> None:
        del base_url, api_key
        self.model_name = model_name

    @staticmethod
    def _query_terms(question: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", question.lower())
        return {word for word in words if len(word) > 2 and word not in QUERY_STOPWORDS}

    @staticmethod
    def _out_of_scope_message(question: str) -> str:
        return (
            f"No matching information for '{question}' was found in the selected PDF documents. "
            "Try asking about a topic covered in your active PDFs or upload a relevant PDF file "
            "in the sidebar."
        )

    @staticmethod
    def _clean_lines(chunk: RetrievedDocument) -> List[str]:
        return [
            line.strip()
            for line in chunk.text.splitlines()
            if line.strip() and not line.strip().startswith("[DOCUMENT:")
        ]

    @staticmethod
    def _canonical_section(line: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9]+", " ", line.casefold()).strip()
        if not normalized or not SECTION_HEADER_REGEX.fullmatch(line):
            return None
        for section, aliases in SECTION_ALIASES.items():
            if normalized in aliases:
                return section
        return None

    @classmethod
    def _requested_sections(cls, question: str) -> set[str]:
        folded = question.casefold()
        requested = {
            section
            for section, aliases in SECTION_ALIASES.items()
            if any(alias in folded for alias in aliases)
        }
        return {"summary"} if "overview" in folded and not requested else requested

    @classmethod
    def _ordered_chunks(cls, chunks: Iterable[RetrievedDocument]) -> List[RetrievedDocument]:
        return sorted(
            chunks,
            key=lambda chunk: (
                chunk.doc_name,
                chunk.page_number,
            ),
        )

    @classmethod
    def _extract_sections(
        cls,
        chunks: List[RetrievedDocument],
        requested: set[str],
    ) -> Dict[str, List[tuple[str, int, List[str]]]]:
        sections: Dict[str, List[tuple[str, int, List[str]]]] = defaultdict(list)
        for doc_name in sorted({chunk.doc_name for chunk in chunks}):
            doc_chunks = cls._ordered_chunks(chunk for chunk in chunks if chunk.doc_name == doc_name)
            active_section: str | None = None
            active_page = 1
            active_lines: List[str] = []

            def flush_section() -> None:
                nonlocal active_lines
                if active_section in requested and active_lines:
                    sections[active_section].append((doc_name, active_page, list(active_lines)))
                active_lines = []

            for chunk in doc_chunks:
                for line in cls._clean_lines(chunk):
                    section = cls._canonical_section(line)
                    if section is not None:
                        flush_section()
                        active_section = section
                        active_page = chunk.page_number
                        active_lines = [line]
                    elif active_section in requested:
                        active_lines.append(line)
            flush_section()
        return sections

    @classmethod
    def _format_sections(
        cls,
        question: str,
        chunks: List[RetrievedDocument],
    ) -> str | None:
        requested = cls._requested_sections(question)
        if not requested:
            return None
        sections = cls._extract_sections(chunks, requested)
        if not sections:
            return None

        parts: List[str] = []
        for section in ("summary", "skills", "experience", "projects", "education", "certifications"):
            for doc_name, page_number, lines in sections.get(section, []):
                parts.append(f"[{doc_name}, Page {page_number}]\n" + "\n".join(lines))
        return "\n\n".join(parts)

    @staticmethod
    def _table_answer(question: str, top: RetrievedDocument) -> str | None:
        from src.retrieval.table_ast import TableASTParser

        matched_cells = TableASTParser.match_query_cells(
            question,
            top.text,
            doc_name=top.doc_name,
            page_number=top.page_number,
        )
        if not matched_cells:
            return None
        cell_lines = [
            f"- {cell.row_key} ({cell.col_header}): {cell.value}" for cell in matched_cells[:8]
        ]
        delta_lines = [
            f"- Period change - {delta['row_key']} ({delta['comparison']}): "
            f"{delta['absolute_change']} ({delta['pct_change']})"
            for delta in TableASTParser.compute_row_deltas(matched_cells)[:4]
        ]
        answer = f"Verified from [{top.doc_name}, Page {top.page_number}]:\n" + "\n".join(cell_lines)
        if delta_lines:
            answer += "\n\nCalculated comparisons:\n" + "\n".join(delta_lines)
        return answer

    @classmethod
    def _narrative_lines(
        cls,
        question: str,
        chunks: List[RetrievedDocument],
    ) -> Dict[str, tuple[RetrievedDocument, List[str]]]:
        terms = cls._query_terms(question)
        grouped: Dict[str, tuple[RetrievedDocument, List[str]]] = {}
        for chunk in cls._ordered_chunks(chunks):
            matching: List[str] = []
            for raw_line in re.split(r"(?<=[.!?])\s+|\n", chunk.text):
                line = " ".join(raw_line.split()).strip()
                if not line or line.startswith(("[", "---", "|")):
                    continue
                line_terms = set(re.findall(r"[a-z0-9]+", line.lower()))
                if terms & line_terms:
                    matching.append(line)
            if matching and chunk.doc_name not in grouped:
                grouped[chunk.doc_name] = (chunk, matching[:3])
        return grouped

    @classmethod
    def _general_answer(cls, question: str, chunks: List[RetrievedDocument]) -> str:
        grouped = cls._narrative_lines(question, chunks)
        if len(grouped) > 1:
            parts = []
            for chunk, lines in grouped.values():
                parts.append(
                    f"- [{chunk.doc_name}, Page {chunk.page_number}]\n  " + "\n  ".join(lines[:2])
                )
            return "Findings by document:\n" + "\n".join(parts)
        if grouped:
            chunk, lines = next(iter(grouped.values()))
            return f"According to [{chunk.doc_name}, Page {chunk.page_number}]:\n" + "\n".join(lines[:6])

        top = chunks[0]
        top_score = float(top.rerank_score if top.rerank_score is not None else top.rrf_score)
        if top_score >= 0.30:
            excerpt = " ".join(top.text.split())[:350]
            return f"Based on [{top.doc_name}, Page {top.page_number}], {excerpt}"
        return ""

    async def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> str:
        """Generate a structured, citation-grounded answer entirely on localhost."""
        if DOCUMENTS_QUESTION_RE.search(question):
            document_names = sorted(set(indexed_docs or [chunk.doc_name for chunk in chunks]))
            if document_names:
                return (
                    "Indexed PDFs: "
                    + ", ".join(f"{name}.pdf" for name in document_names)
                    + ". Select a document in the sidebar or ask a question across all PDFs."
                )
            return "No PDFs are indexed yet. Upload a PDF or restore the demo shelf from the left sidebar."

        if META_QUESTION_RE.search(question):
            return META_RESPONSE
        if not chunks:
            return self._out_of_scope_message(question)

        section_answer = self._format_sections(question, chunks)
        if section_answer:
            return section_answer

        table_answer = self._table_answer(question, chunks[0])
        if table_answer:
            return table_answer

        narrative_answer = self._general_answer(question, chunks)
        return narrative_answer or self._out_of_scope_message(question)

    async def stream_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the same local answer path as server-sent events."""
        answer = await self.generate_answer(question, chunks, indexed_docs=indexed_docs)
        words = answer.split(" ")
        for index, word in enumerate(words):
            suffix = " " if index < len(words) - 1 else ""
            yield f"data: {word}{suffix}\n\n"
        yield "data: [DONE]\n\n"
