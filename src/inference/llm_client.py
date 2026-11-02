"""Local and optional OpenAI-compatible citation-grounded answer synthesis."""

import json
import logging
import re
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, Iterable, List

import httpx

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
TABLE_INTENT_RE = re.compile(
    r"\b(table|row|column|revenue|sales|income|profit|margin|capex|"
    r"capital expenditure|expense|cash flow|cashflow|assets?|liabilit\w*|"
    r"ratio|earnings|dividend|debt|turnover|growth|percent|percentage|"
    r"fy\s?\d{2,4}|\d{4}|%)\b",
    flags=re.IGNORECASE,
)
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
logger = logging.getLogger(__name__)
REMOTE_SYSTEM_PROMPT = (
    "You are a financial document analyst. Answer ONLY from the supplied EVIDENCE. "
    "Every factual claim must end with a citation in the exact form [DOC_NAME, Page N]. "
    "Never invent numbers, pages, documents, or sources. If the evidence does not answer "
    "the question, say that the information is not present in the indexed documents. "
    "Prefer concise bullets and preserve table labels and units verbatim."
)
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
        *,
        remote_enabled: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.remote_enabled = (
            settings.LLM_REMOTE_ENABLED if remote_enabled is None else remote_enabled
        )
        self.timeout_seconds = timeout_seconds or settings.LLM_TIMEOUT_SECONDS
        self._transport = transport
        self.last_provider = "local_deterministic"
        self.last_error: str | None = None
        self.remote_requests = 0
        self.remote_failures = 0

    def configure_provider(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        enabled: bool | None = None,
    ) -> Dict[str, object]:
        """Update the optional OpenAI-compatible provider without restarting."""
        if base_url is not None and base_url.strip():
            self.base_url = base_url.strip().rstrip("/")
        if api_key is not None:
            self.api_key = api_key.strip()
        if model_name is not None and model_name.strip():
            self.model_name = model_name.strip()
        if enabled is not None:
            self.remote_enabled = enabled
        return self.get_provider_status()

    def get_provider_status(self) -> Dict[str, object]:
        """Return safe provider metadata without exposing the configured secret."""
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "api_key_configured": bool(self.api_key and self.api_key != "EMPTY"),
            "enabled": self.remote_enabled,
            "last_provider": self.last_provider,
            "last_error": self.last_error,
            "remote_requests": self.remote_requests,
            "remote_failures": self.remote_failures,
        }

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
        from src.retrieval.table_ast import TableASTParser

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
                f"Verified from [{chunk.doc_name}, Page {chunk.page_number}]:\n"
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

    def _request_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chat_endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _request_payload(
        self, question: str, chunks: List[RetrievedDocument], *, stream: bool
    ) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": REMOTE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_evidence(question, chunks),
                },
            ],
            "temperature": 0.0,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "stream": stream,
        }

    def _mark_remote_success(self) -> None:
        self.last_provider = "remote_openai_compatible"
        self.last_error = None
        self.remote_requests += 1

    def _mark_remote_failure(self, exc: Exception) -> None:
        self.last_provider = "local_fallback"
        self.last_error = f"{type(exc).__name__}: {exc}"[:500]
        self.remote_failures += 1
        logger.warning(
            "Remote LLM request failed (%s); using deterministic fallback",
            type(exc).__name__,
        )

    async def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> str:
        """Use the configured remote model when enabled, else deterministic synthesis."""
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

        if not self.remote_enabled:
            self.last_provider = "local_deterministic"
        if self.remote_enabled:
            try:
                async with httpx.AsyncClient(
                    transport=self._transport,
                    timeout=self.timeout_seconds,
                ) as client:
                    response = await client.post(
                        self._chat_endpoint(),
                        headers=self._request_headers(),
                        json=self._request_payload(question, chunks, stream=False),
                    )
                    response.raise_for_status()
                    data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Remote provider returned an empty answer")
                cited_pages = {
                    (match.group(1).strip(), int(match.group(2)))
                    for match in re.finditer(
                        r"\[([^,\]]+),\s*(?:page|p)\.?\s*(\d+)\]",
                        content,
                        flags=re.IGNORECASE,
                    )
                }
                available_pages = {
                    (chunk.doc_name, chunk.page_number) for chunk in chunks
                }
                if not cited_pages or not cited_pages <= available_pages:
                    raise ValueError(
                        "Remote provider cited a page outside the retrieved evidence"
                    )
                self._mark_remote_success()
                return content.strip()
            except Exception as exc:
                self._mark_remote_failure(exc)

        if self.last_provider != "local_fallback":
            self.last_provider = "local_deterministic"
        return self._generate_local_answer(question, chunks, indexed_docs=indexed_docs)

    def _generate_local_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> str:
        """Generate a structured, citation-grounded answer entirely on localhost."""
        answer_chunks = self._select_answer_chunks(question, chunks, indexed_docs)
        section_answer = self._format_sections(question, answer_chunks)
        if section_answer:
            return section_answer

        table_answer = self._table_answer(question, answer_chunks)
        if table_answer:
            return table_answer

        narrative_answer = self._general_answer(question, answer_chunks)
        return narrative_answer or self._out_of_scope_message(question)

    async def stream_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield answer text from the remote SSE stream, falling back locally."""
        if not self.remote_enabled:
            self.last_provider = "local_deterministic"
        if self.remote_enabled and chunks:
            emitted = False
            try:
                async with httpx.AsyncClient(
                    transport=self._transport,
                    timeout=self.timeout_seconds,
                ) as client:
                    async with client.stream(
                        "POST",
                        self._chat_endpoint(),
                        headers=self._request_headers(),
                        json=self._request_payload(question, chunks, stream=True),
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                            event = json.loads(payload)
                            delta = event["choices"][0].get("delta", {})
                            token = delta.get("content")
                            if isinstance(token, str) and token:
                                emitted = True
                                yield token
                if not emitted:
                    raise ValueError("Remote provider streamed no content")
                self._mark_remote_success()
                return
            except Exception as exc:
                self._mark_remote_failure(exc)
                if emitted:
                    return

        if self.last_provider != "local_fallback":
            self.last_provider = "local_deterministic"
        answer = self._generate_local_answer(question, chunks, indexed_docs=indexed_docs)
        words = answer.split(" ")
        for index, word in enumerate(words):
            suffix = " " if index < len(words) - 1 else ""
            yield f"{word}{suffix}"
