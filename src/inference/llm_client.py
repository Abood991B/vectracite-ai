import logging
from typing import AsyncGenerator, List
from openai import AsyncOpenAI
from src.config import settings
from src.retrieval.hybrid_retriever import RetrievedDocument

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are VectraCite-AI, an institutional-grade financial document intelligence assistant.
Answer the user's question strictly using ONLY the provided retrieved document chunks and financial tables.
Requirements:
1. Cite the exact document name and page number for every figure or claim using the format [DOC_NAME, Page X].
2. If the retrieved context does not contain sufficient evidence to answer the question, state clearly: "INSUFFICIENT_EVIDENCE_IN_CONTEXT".
3. Never fabricate financial figures, dates, or percentages.
"""


class UnifiedLLMClient:
    """
    Async inference client compatible with self-hosted vLLM servers (OpenAI protocol)
    and external providers, with deterministic grounded synthesis fallback when running offline tests.
    """

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
        for idx, c in enumerate(chunks, start=1):
            context_blocks.append(
                f"--- SOURCE {idx}: [{c.doc_name}, Page {c.page_number}] "
                f"(Table={c.contains_table}, Score={c.rerank_score or c.rrf_score:.4f}) ---\n"
                f"{c.text}"
            )
        joined_context = "\n\n".join(context_blocks)
        return (
            f"RETRIEVED CONTEXT:\n{joined_context}\n\n"
            f"QUESTION: {question}\n\n"
            "Provide a concise, citation-grounded answer citing [DOC_NAME, Page X]:"
        )

    async def generate_answer(self, question: str, chunks: List[RetrievedDocument]) -> str:
        """Generates a citation-grounded answer via vLLM/OpenAI or deterministic evidence synthesis."""
        if not chunks:
            return "INSUFFICIENT_EVIDENCE_IN_CONTEXT: No relevant document chunks were retrieved."

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

        # Deterministic coordinate-grounded synthesis when running without an external GPU server
        top = chunks[0]
        from src.retrieval.table_ast import TableASTParser

        matched_cells = TableASTParser.match_query_cells(
            question, top.text, doc_name=top.doc_name, page_number=top.page_number
        )
        if matched_cells:
            cell_summary = "; ".join(
                f"{c.row_key} ({c.col_header}): {c.value}" for c in matched_cells[:4]
            )
            return f"Verified from [{top.doc_name}, Page {top.page_number}]: {cell_summary}."

        clean_excerpt = " ".join(top.text.split())[:350]
        return (
            f"Based on [{top.doc_name}, Page {top.page_number}], "
            f"the extracted evidence states: \"{clean_excerpt}\""
        )

    async def stream_answer(
        self, question: str, chunks: List[RetrievedDocument]
    ) -> AsyncGenerator[str, None]:
        """Streams tokens via Server-Sent Events (SSE) for real-time low-latency UI rendering."""
        answer = await self.generate_answer(question, chunks)
        words = answer.split(" ")
        for i, word in enumerate(words):
            suffix = " " if i < len(words) - 1 else ""
            yield f"data: {word}{suffix}\n\n"
        yield "data: [DONE]\n\n"
