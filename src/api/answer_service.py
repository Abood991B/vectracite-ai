"""Single guarded answer pipeline shared by JSON and SSE routes."""

import re
import time
from typing import Any, Callable, List

from fastapi import HTTPException

from src.cache_and_guardrails.guardrails import EnterpriseGuardrails
from src.cache_and_guardrails.numeric_verifier import DeterministicNumericVerifier
from src.cache_and_guardrails.semantic_cache import SemanticCache
from src.inference.llm_client import UnifiedLLMClient
from src.inference.schemas import (
    CitationItem,
    NumericVerificationSchema,
    QueryRequest,
    QueryResponse,
    RowDeltaSchema,
    TableCellSchema,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.table_ast import TableASTParser


def answer_contributors(answer: str, chunks: List[Any]) -> List[Any]:
    """Keep only evidence pages explicitly cited by the synthesized answer."""
    cited_pages = {
        (match.group(1).strip(), int(match.group(2)))
        for match in re.finditer(
            r"\[([^,\]]+),\s*(?:page|p)\.?\s*(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )
    }
    if not cited_pages:
        return []
    contributors: List[Any] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = (chunk.doc_name, chunk.page_number)
        if key in cited_pages and chunk.chunk_id not in seen:
            contributors.append(chunk)
            seen.add(chunk.chunk_id)
    return contributors


class AnswerService:
    """Produces the fully guarded response consumed by both query transports."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        semantic_cache: SemanticCache,
        guardrails: EnterpriseGuardrails,
        numeric_verifier: DeterministicNumericVerifier,
        llm_client: UnifiedLLMClient,
        ensure_indexed: Callable[[], None],
        normalize_scope: Callable[[str | None], str | None],
        infer_scope: Callable[[str], str | None],
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.semantic_cache = semantic_cache
        self.guardrails = guardrails
        self.numeric_verifier = numeric_verifier
        self.llm_client = llm_client
        self.ensure_indexed = ensure_indexed
        self.normalize_scope = normalize_scope
        self.infer_scope = infer_scope

    async def execute(self, payload: QueryRequest) -> QueryResponse:
        """Run input guards, cache lookup, retrieval, synthesis, and verification once."""
        start_time = time.perf_counter()

        input_check = self.guardrails.validate_input(payload.question)
        if not input_check.passed:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Query blocked by Enterprise AI Guardrails.",
                    "triggered_rules": input_check.triggered_rules,
                },
            )

        self.ensure_indexed()
        scope = self.normalize_scope(payload.doc_name) or self.infer_scope(
            input_check.sanitized_text
        )
        cache_key = (
            f"{scope or 'ALL'}::top_k={payload.top_k}::{input_check.sanitized_text}"
        )
        if payload.use_cache:
            cached = self.semantic_cache.lookup(cache_key)
            if cached is not None:
                cached["latency_ms"] = round(
                    (time.perf_counter() - start_time) * 1000.0, 2
                )
                return QueryResponse(**cached)

        hybrid_candidates = self.retriever.retrieve_hybrid(
            query=input_check.sanitized_text,
            top_k=max(payload.top_k * 2, 6),
            doc_name=scope,
        )
        reranked_chunks = self.reranker.rerank(
            query=input_check.sanitized_text,
            candidates=hybrid_candidates,
            top_n=payload.top_k,
        )

        indexed_docs = sorted({chunk.doc_name for chunk in self.retriever.chunks_store})
        raw_answer = await self.llm_client.generate_answer(
            input_check.sanitized_text,
            reranked_chunks,
            indexed_docs=indexed_docs,
        )
        output_check = self.guardrails.sanitize_output(raw_answer)
        answer_chunks = answer_contributors(
            output_check.sanitized_text, reranked_chunks
        )

        matched_cells = []
        computed_deltas = []
        if answer_chunks:
            top_chunk = answer_chunks[0]
            raw_cells = TableASTParser.match_query_cells(
                input_check.sanitized_text,
                top_chunk.text,
                doc_name=top_chunk.doc_name,
                page_number=top_chunk.page_number,
            )
            matched_cells = [
                TableCellSchema(**cell.to_dict()).model_dump() for cell in raw_cells
            ]
            computed_deltas = [
                RowDeltaSchema(**delta).model_dump()
                for delta in TableASTParser.compute_row_deltas(raw_cells)
            ]

        verification = self.numeric_verifier.verify(
            output_check.sanitized_text, answer_chunks
        )

        citations: List[CitationItem] = [
            CitationItem(
                chunk_id=chunk.chunk_id,
                doc_name=chunk.doc_name,
                page_number=chunk.page_number,
                contains_table=chunk.contains_table,
                rerank_score=round(float(chunk.rerank_score or chunk.rrf_score), 4),
                excerpt=self.guardrails.sanitize_output(chunk.text[:280]).sanitized_text,
            )
            for chunk in answer_chunks
        ]

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        response_data = {
            "question": input_check.sanitized_text,
            "answer": output_check.sanitized_text,
            "matched_cells": matched_cells,
            "computed_deltas": computed_deltas,
            "citations": [citation.model_dump() for citation in citations],
            "verification": NumericVerificationSchema(
                **verification.to_dict()
            ).model_dump(),
            "cache_hit": False,
            "cache_similarity": None,
            "latency_ms": latency_ms,
            "guardrail_flags": output_check.triggered_rules,
            "metadata": {
                "retrieval_strategy": "hybrid_rrf_plus_table_schema_cross_encoder",
                "candidates_evaluated": len(hybrid_candidates),
                "document_scope": scope or "ALL",
                "answer_provider": self.llm_client.last_provider,
                "provider_error": self.llm_client.last_error,
            },
        }

        if payload.use_cache and reranked_chunks:
            self.semantic_cache.store(cache_key, response_data)

        return QueryResponse(**response_data)
