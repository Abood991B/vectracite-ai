"""Deterministic, citation-grounded answer synthesis.

This is application policy: it decides which evidence supports a question, how tables and sections are
formatted, and when an answer is out of scope. It runs entirely locally and depends only on domain
types and pure parsing helpers — no HTTP client, no settings, no provider.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List

from src.domain.documents import RetrievedDocument
from src.domain.sections import SECTION_HEADER_REGEX
from src.retrieval.table_ast import TableASTParser


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
QUERY_TERM_SYNONYMS: Dict[str, set[str]] = {
    "cashflow": {"cash", "flow"},
    "freecashflow": {"free", "cash", "flow"},
    "fcf": {"free", "cash", "flow"},
    "capex": {"capital", "expenditure"},
    "ppne": {"property", "plant", "equipment"},
    "operatingmargin": {"operating", "margin"},
    "netincome": {"net", "income"},
    "netsales": {"net", "sales"},
    "totalrevenue": {"total", "revenue"},
}
SECTION_ALIASES: Dict[str, tuple[str, ...]] = {
    "summary": ("professional summary", "executive summary", "summary", "profile", "overview"),
    "skills": ("technical skills", "skills", "core competencies", "technologies"),
    "experience": ("work experience", "experience", "employment"),
    "projects": ("key projects", "projects"),
    "education": ("education", "academic"),
    "certifications": ("certifications", "certificates", "awards"),
}
TABLE_INTENT_RE = re.compile(
    r"\b(table|row|column|revenue|sales|income|profit|margin|capex|"
    r"capital expenditure|expense|cash flow|cashflow|assets?|liabilit\w*|"
    r"ratio|earnings|dividend|debt|turnover|growth|percent|percentage|"
    r"fy\s?\d{2,4}|\d{4}|%)\b",
    flags=re.IGNORECASE,
)
META_QUESTION_RE = re.compile(
    r"\b(what model|which model|who are you|what are you|how do you work|"
    r"how does (?:this|it) work|what llm|privacy|is my data private|"
    r"local execution)\b",
    flags=re.IGNORECASE,
)
DOCUMENTS_QUESTION_RE = re.compile(
    r"\b(what documents|which documents|what files|which files).*\b"
    r"(loaded|indexed|available)\b",
    flags=re.IGNORECASE,
)
META_RESPONSE = (
    "VectraCite-AI retrieves document passages using dense and lexical search, "
    "reranks them, and produces a cited answer with numeric checks. "
    "The default synthesizer is local. When a remote LLM or OCR service is enabled, "
    "document content is sent to that configured service."
)


class LocalSynthesizer:
    """Build the deterministic answer a local deployment serves without a provider."""

    def answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> str:
        """Select evidence, then prefer sections, then tables, then narrative."""
        answer_chunks = self._select_answer_chunks(question, chunks, indexed_docs)
        section_answer = self._format_sections(question, answer_chunks)
        if section_answer:
            return section_answer

        table_answer = self._table_answer(question, answer_chunks)
        if table_answer:
            return table_answer

        narrative_answer = self._general_answer(question, answer_chunks)
        return narrative_answer or self.out_of_scope(question)

    def out_of_scope(self, question: str) -> str:
        """Return the standard message for a question the evidence cannot answer."""
        return self._out_of_scope_message(question)

    def build_evidence(self, question: str, chunks: List[RetrievedDocument]) -> str:
        """Render the evidence block a remote provider receives."""
        return self._build_evidence(question, chunks)
    @staticmethod
    def _query_terms(question: str) -> set[str]:
        cleaned = re.sub(r"['�?T]s\b", "", question.lower())
        words = re.findall(r"[\w]+", cleaned, flags=re.UNICODE)
        terms: set[str] = set()
        for word in words:
            if len(word) > 2 and word not in QUERY_STOPWORDS:
                terms.add(word)
            parts = re.findall(r"[a-z]+|\d+", word)
            if len(parts) > 1:
                terms.update(
                    part
                    for part in parts
                    if len(part) > 2 and part not in QUERY_STOPWORDS
                )
            terms.update(QUERY_TERM_SYNONYMS.get(word, set()))
        return terms
    @classmethod
    def _is_cross_document_request(
        cls, question: str, indexed_docs: List[str] | None = None
    ) -> bool:
        if not re.search(
            r"\b(compare|comparison|between|across|versus|vs\.?|both|each)\b",
            question,
            flags=re.IGNORECASE,
        ):
            return False

        question_folded = question.casefold()
        question_compact = re.sub(r"[^a-z0-9]", "", question_folded)
        docs = list(indexed_docs or [])
        exact_documents = {
            doc_name
            for doc_name in docs
            if re.search(rf"\b{re.escape(doc_name.casefold())}\b", question_folded)
        }
        explicit_documents = set(exact_documents)
        if not exact_documents:
            for doc_name in docs:
                alias = doc_name.split("_", maxsplit=1)[0].casefold()
                compact = re.sub(r"[^a-z0-9]", "", alias)
                if (len(alias) >= 2 and re.search(rf"\b{re.escape(alias)}\b", question_folded)) or (
                    compact and compact in question_compact
                ):
                    explicit_documents.add(doc_name)
        return len(explicit_documents) >= 2 or bool(
            re.search(r"\b(documents?|files?|reports?|filings?|pdfs?)\b", question, re.I)
        )
    @classmethod
    def _select_answer_chunks(
        cls,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> List[RetrievedDocument]:
        """Keep normal answers on the primary hit; admit only qualified comparisons."""
        if not chunks:
            return []
        primary_doc = chunks[0].doc_name
        selected = [chunk for chunk in chunks if chunk.doc_name == primary_doc]
        if not cls._is_cross_document_request(question, indexed_docs):
            return selected

        query_terms = cls._query_terms(question)
        for chunk in chunks:
            if chunk.doc_name == primary_doc or not query_terms:
                continue
            chunk_terms = set(
                re.findall(r"[\w]+", chunk.text.casefold(), flags=re.UNICODE)
            )
            coverage = len(query_terms & chunk_terms) / len(query_terms)
            score = float(chunk.rerank_score or 0.0)
            if score >= 0.60 and coverage >= 0.50:
                selected.append(chunk)
        return selected
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
    @classmethod
    def _table_answer(
        cls, question: str, chunks: List[RetrievedDocument]
    ) -> str | None:

        if not TABLE_INTENT_RE.search(question):
            return None
        query_terms = cls._query_terms(question)
        ordered = sorted(
            chunks,
            key=lambda chunk: (
                not chunk.contains_table,
                -(float(chunk.rerank_score or chunk.rrf_score or 0.0)),
            ),
        )
        for chunk in ordered:
            matched_cells = TableASTParser.match_query_cells(
                question,
                chunk.text,
                doc_name=chunk.doc_name,
                page_number=chunk.page_number,
            )
            if not matched_cells:
                continue
            entity_terms = set(
                re.findall(r"[a-z0-9]+", chunk.doc_name.split("_", maxsplit=1)[0].lower())
            )
            metric_terms = {
                term
                for term in query_terms - entity_terms
                if term not in QUERY_STOPWORDS and len(term) >= 3
            }
            has_metric_overlap = any(
                metric_terms
                & (
                    set(re.findall(r"[a-z0-9]+", cell.row_key.lower()))
                    | set(re.findall(r"[a-z0-9]+", cell.col_header.lower()))
                )
                for cell in matched_cells
            )
            if metric_terms and not has_metric_overlap:
                continue
            cell_lines = [
                f"- {cell.row_key} ({cell.col_header}): {cell.value}"
                for cell in matched_cells[:8]
            ]
            delta_lines = [
                f"- Period change - {delta['row_key']} ({delta['comparison']}): "
                f"{delta['absolute_change']} ({delta['pct_change']})"
                for delta in TableASTParser.compute_row_deltas(matched_cells)[:4]
            ]
            answer = (
                f"From [{chunk.doc_name}, Page {chunk.page_number}]:\n"
                + "\n".join(cell_lines)
            )
            if delta_lines:
                answer += "\n\nCalculated comparisons:\n" + "\n".join(delta_lines)
            return answer
        return None
    @classmethod
    def _narrative_lines(
        cls,
        question: str,
        chunks: List[RetrievedDocument],
    ) -> Dict[str, tuple[RetrievedDocument, List[str]]]:
        terms = cls._query_terms(question)
        grouped: Dict[str, tuple[RetrievedDocument, List[str]]] = {}
        scores: Dict[str, int] = {}
        for chunk in cls._ordered_chunks(chunks):
            scored_lines: List[tuple[int, int, str]] = []
            for line_index, raw_line in enumerate(
                re.split(r"(?<=[.!?])\s+|\n", chunk.text)
            ):
                line = " ".join(raw_line.split()).strip()
                if not line or line.startswith(("[", "---", "|")):
                    continue
                raw_tokens = re.findall(r"[a-z0-9]+", line.lower())
                if len(raw_tokens) < 3 and not re.search(r"\d", line):
                    continue
                if line.isupper() and not re.search(r"\d", line):
                    continue
                line_terms = set(raw_tokens)
                overlap = len(terms & line_terms)
                if overlap:
                    scored_lines.append((overlap, line_index, line))
            if not scored_lines:
                continue
            selected = sorted(scored_lines, key=lambda item: item[0], reverse=True)[:5]
            selected.sort(key=lambda item: item[1])
            chunk_score = sum(item[0] for item in selected)
            if chunk.doc_name not in grouped or chunk_score > scores[chunk.doc_name]:
                grouped[chunk.doc_name] = (chunk, [item[2] for item in selected])
                scores[chunk.doc_name] = chunk_score
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
    @staticmethod
    def _build_evidence(question: str, chunks: List[RetrievedDocument]) -> str:
        """Build a bounded, citation-labelled evidence packet for a remote model."""
        blocks: List[str] = []
        remaining = 24_000
        for chunk in chunks:
            header = f"[{chunk.doc_name}, Page {chunk.page_number}]"
            body = chunk.text.strip()
            block = f"{header}\n{body}"
            if len(block) > remaining:
                block = block[:remaining]
            if not block:
                continue
            blocks.append(block)
            remaining -= len(block)
            if remaining <= 0:
                break
        return "QUESTION:\n{}\n\nEVIDENCE:\n{}".format(
            question, "\n\n".join(blocks)
        )
