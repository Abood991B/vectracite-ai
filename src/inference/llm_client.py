import logging
import re
from typing import AsyncGenerator, List

from openai import AsyncOpenAI

from src.config import settings
from src.retrieval.hybrid_retriever import RetrievedDocument

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are VectraCite-AI, an institutional-grade document intelligence assistant.
Answer the user's question strictly using ONLY the provided retrieved document chunks and tables.
Requirements:
1. Cite the exact document name and page number for every figure or claim using [DOC_NAME, Page X].
2. If the retrieved context does not contain sufficient evidence, say so clearly.
3. Never fabricate figures, dates, or percentages.
"""

META_QUESTION_RE = re.compile(
    r"\b(what model|which model|who are you|what are you|how do you work|how does (?:this|it) work|what is this|what llm)\b",
    flags=re.IGNORECASE,
)
DOCUMENTS_QUESTION_RE = re.compile(
    r"\b(what documents|which documents|what files|which files).*\b(loaded|indexed|available)\b",
    flags=re.IGNORECASE,
)
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
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
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
META_RESPONSE = (
    "VectraCite-AI runs a 3-stage pipeline: (1) BAAI/bge-small-en-v1.5 local PyTorch "
    "bi-encoder embeddings + BM25 lexical search, (2) Adaptive SCAR 2D Table-AST "
    "reranking with BAAI/bge-reranker-base, and (3) Grounded Answer Synthesis with "
    "deterministic numeric verification. Ask any question about your indexed PDFs or "
    "upload a new PDF in the left sidebar."
)


class UnifiedLLMClient:
    """Async vLLM client with deterministic, citation-grounded offline behavior."""

    def __init__(
        self,
        base_url: str = settings.LLM_BASE_URL,
        api_key: str = settings.LLM_API_KEY,
        model_name: str = settings.LLM_MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        is_local_unconfigured = "localhost:8000" in base_url and api_key == "EMPTY"
        req_timeout = 0.25 if is_local_unconfigured else 15.0
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=req_timeout,
            max_retries=0 if is_local_unconfigured else 2,
        )

    def _build_context_prompt(self, question: str, chunks: List[RetrievedDocument]) -> str:
        context_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            score = chunk.rerank_score if chunk.rerank_score is not None else chunk.rrf_score
            context_blocks.append(
                f"--- SOURCE {idx}: [{chunk.doc_name}, Page {chunk.page_number}] "
                f"(Table={chunk.contains_table}, Score={score:.4f}) ---\n"
                f"{chunk.text}"
            )
        joined_context = "\n\n".join(context_blocks)
        return (
            f"RETRIEVED CONTEXT:\n{joined_context}\n\n"
            f"QUESTION: {question}\n\n"
            "Provide a concise, citation-grounded answer citing [DOC_NAME, Page X]:"
        )

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

    def _deterministic_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
    ) -> tuple[str, str | None]:
        """Return grounded text and its evidence kind: table, narrative, score, or None."""
        from src.retrieval.table_ast import TableASTParser

        top = chunks[0]
        matched_cells = TableASTParser.match_query_cells(
            question, top.text, doc_name=top.doc_name, page_number=top.page_number
        )
        if matched_cells:
            cell_summary = "; ".join(
                f"{cell.row_key} ({cell.col_header}): {cell.value}" for cell in matched_cells[:8]
            )
            deltas = TableASTParser.compute_row_deltas(matched_cells)
            delta_summary = ""
            if deltas:
                delta_summary = "; " + "; ".join(
                    f"{delta['row_key']} {delta['comparison']}: "
                    f"{delta['absolute_change']} ({delta['pct_change']})"
                    for delta in deltas
                )
            return (
                f"Verified from [{top.doc_name}, Page {top.page_number}]: "
                f"{cell_summary}{delta_summary}.",
                "table",
            )

        query_terms = self._query_terms(question)
        scored_lines: List[tuple[int, RetrievedDocument, str]] = []
        for chunk in chunks:
            for raw_line in re.split(r"(?<=[.!?])\s+|\n", chunk.text):
                line = " ".join(raw_line.split()).strip()
                if not line or line.startswith("[") or line.startswith("---") or line.startswith("|"):
                    continue
                line_terms = set(re.findall(r"[a-z0-9]+", line.lower()))
                overlap = len(query_terms & line_terms)
                if overlap:
                    scored_lines.append((overlap, chunk, line))

        scored_lines.sort(
            key=lambda item: (
                item[0],
                item[1].rerank_score if item[1].rerank_score is not None else item[1].rrf_score,
            ),
            reverse=True,
        )
        if scored_lines:
            selected: List[str] = []
            for _, chunk, line in scored_lines[:2]:
                selected.append(f"[{chunk.doc_name}, Page {chunk.page_number}] {line}")
            return "Based on " + " ".join(selected), "narrative"

        top_score = float(top.rerank_score if top.rerank_score is not None else top.rrf_score)
        if top_score >= 0.30:
            excerpt = " ".join(top.text.split())[:350]
            return f"Based on [{top.doc_name}, Page {top.page_number}], {excerpt}", "score"

        return "", None

    async def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> str:
        """Generate a grounded answer, including universal meta and out-of-scope handling."""
        if DOCUMENTS_QUESTION_RE.search(question):
            document_names = sorted(set(indexed_docs or [chunk.doc_name for chunk in chunks]))
            if document_names:
                return (
                    "Indexed PDFs: "
                    + ", ".join(f"{name}.pdf" for name in document_names)
                    + ". Select a document in the sidebar or ask a question across all PDFs."
                )
            return (
                "No PDFs are indexed yet. Upload a PDF or restore the demo shelf from the "
                "left sidebar."
            )
        if META_QUESTION_RE.search(question):
            return META_RESPONSE

        if not chunks:
            return self._out_of_scope_message(question)

        deterministic_answer, evidence_kind = self._deterministic_answer(question, chunks)
        if evidence_kind is None:
            return self._out_of_scope_message(question)
        if evidence_kind in {"table", "score"}:
            return deterministic_answer

        user_prompt = self._build_context_prompt(question, chunks)
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception as exc:
            logger.info("Using deterministic extractive citation synthesizer (%s)", type(exc).__name__)

        return deterministic_answer

    async def stream_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the same universal answer path as server-sent events."""
        answer = await self.generate_answer(question, chunks, indexed_docs=indexed_docs)
        words = answer.split(" ")
        for i, word in enumerate(words):
            suffix = " " if i < len(words) - 1 else ""
            yield f"data: {word}{suffix}\n\n"
        yield "data: [DONE]\n\n"
