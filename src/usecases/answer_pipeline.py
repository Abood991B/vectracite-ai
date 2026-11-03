"""Guarded answer pipeline: the application policy shared by JSON and SSE delivery.

Framework-free by construction. It knows the ports it needs, the domain types it moves, and the
response DTO it returns — nothing about FastAPI, Qdrant, Redis, or httpx. A refusal is a domain event
(`QueryBlockedError`); the HTTP adapter decides that means 400.
"""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Callable, List

from src.cache_and_guardrails.guardrails import EnterpriseGuardrails
from src.cache_and_guardrails.numeric_verifier import DeterministicNumericVerifier
from src.inference.schemas import (
    CitationItem,
    NumericVerificationSchema,
    QueryRequest,
    QueryResponse,
    RowDeltaSchema,
    TableCellSchema,
)
from src.retrieval.table_ast import TableASTParser
from src.usecases.ports import (
    AnswerCachePort,
    QueryBlockedError,
    RerankerPort,
    RetrieverPort,
    SynthesizerPort,
)
from src.usecases.synthesis import (
    TABLE_INTENT_RE,
    expand_query,
    is_comparison_question,
    mentioned_documents,
    query_terms,
)
from src.usecases.query_normalization import correct_query_terms
from src.domain.citations import parse_citations
from src.domain.documents import RetrievedDocument


def answer_contributors(answer: str, chunks: List[Any]) -> List[Any]:
    """Keep only evidence pages explicitly cited by the synthesized answer."""
    cited_pages = parse_citations(answer)
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


class AnswerPipeline:
    """Produces the fully guarded response consumed by both query transports."""

    def __init__(
        self,
        *,
        retriever: RetrieverPort,
        reranker: RerankerPort,
        semantic_cache: AnswerCachePort,
        guardrails: EnterpriseGuardrails,
        numeric_verifier: DeterministicNumericVerifier,
        llm_client: SynthesizerPort,
        ensure_indexed: Callable[[], None],
        normalize_scope: Callable[[str | None], str | None],
        infer_scope: Callable[[str, List[str]], str | None],
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
            raise QueryBlockedError(input_check.triggered_rules)

        self.ensure_indexed()
        indexed_docs = self.retriever.indexed_document_names()
        scope = self.normalize_scope(payload.doc_name) or self.infer_scope(
            input_check.sanitized_text, indexed_docs
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

        vocabulary = self.retriever.corpus_vocabulary()
        normalized_question = correct_query_terms(
            input_check.sanitized_text, vocabulary
        )
        retrieval_query = expand_query(normalized_question)
        hybrid_candidates, reranked_chunks = self._select_evidence(
            question=normalized_question,
            retrieval_query=retrieval_query,
            payload=payload,
            scope=scope,
            indexed_docs=indexed_docs,
        )

        raw_answer = await self.llm_client.generate_answer(
            normalized_question,
            reranked_chunks,
            indexed_docs=indexed_docs,
        )
        response = self._build_response(
            question=input_check.sanitized_text,
            answer=raw_answer,
            reranked_chunks=reranked_chunks,
            scope=scope,
            candidates_evaluated=len(hybrid_candidates),
            start_time=start_time,
        )
        if payload.use_cache and reranked_chunks:
            self.semantic_cache.store(cache_key, response.model_dump())
        return response

    def _select_evidence(
        self,
        *,
        question: str,
        retrieval_query: str,
        payload: QueryRequest,
        scope: str | None,
        indexed_docs: List[str],
    ) -> tuple[List[RetrievedDocument], List[RetrievedDocument]]:
        """Retrieve, rerank, then add each document's matching tables for table questions."""
        comparison_mode = is_comparison_question(question) and scope is None
        if comparison_mode:
            targets = mentioned_documents(question, indexed_docs) or indexed_docs
            by_doc: dict[str, List[RetrievedDocument]] = {}
            for doc_name in targets:
                hits = self.retriever.retrieve_hybrid(
                    query=retrieval_query,
                    top_k=max(payload.top_k * 2, 8),
                    doc_name=doc_name,
                )
                if hits:
                    by_doc[doc_name] = hits
            hybrid_candidates = [chunk for hits in by_doc.values() for chunk in hits]
            reranked_chunks: List[RetrievedDocument] = []
            for hits in by_doc.values():
                reranked_chunks.extend(
                    self.reranker.rerank(
                        query=retrieval_query,
                        candidates=hits,
                        top_n=6,
                    )
                )
            table_docs = list(by_doc)
        else:
            hybrid_candidates = self.retriever.retrieve_hybrid(
                query=retrieval_query,
                top_k=max(payload.top_k * 2, 6),
                doc_name=scope,
            )
            reranked_chunks = self.reranker.rerank(
                query=retrieval_query,
                candidates=hybrid_candidates,
                top_n=payload.top_k,
            )
            if scope:
                reranked_chunks = [
                    chunk for chunk in reranked_chunks if chunk.doc_name == scope
                ]
                table_docs = [scope]
            else:
                table_docs = mentioned_documents(question, indexed_docs) or sorted(
                    {chunk.doc_name for chunk in hybrid_candidates}
                )
        return hybrid_candidates, self._add_table_candidates(
            question, table_docs, reranked_chunks
        )

    def _add_table_candidates(
        self,
        question: str,
        doc_names: List[str],
        chunks: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """Append matching table chunks so an income statement can outrank retrieved narrative."""
        if not TABLE_INTENT_RE.search(question):
            return chunks
        terms = query_terms(question)
        seen = {(chunk.doc_name, chunk.chunk_id) for chunk in chunks}
        augmented = list(chunks)
        for doc_name in doc_names:
            for chunk in self.retriever.table_candidates(doc_name, terms):
                key = (chunk.doc_name, chunk.chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                augmented.append(
                    RetrievedDocument(
                        chunk_id=chunk.chunk_id,
                        doc_name=chunk.doc_name,
                        page_number=chunk.page_number,
                        text=chunk.text,
                        contains_table=chunk.contains_table,
                        dense_score=0.0,
                        bm25_score=0.0,
                        rrf_score=0.0,
                    )
                )
        return augmented

    def _build_response(
        self,
        *,
        question: str,
        answer: str,
        reranked_chunks: List[Any],
        scope: str | None,
        candidates_evaluated: int,
        start_time: float,
        cache_hit: bool = False,
        cache_similarity: float | None = None,
        already_sanitized: bool = False,
        guardrail_flags: List[str] | None = None,
    ) -> QueryResponse:
        if already_sanitized:
            sanitized_answer = answer
            flags = list(guardrail_flags or [])
        else:
            output_check = self.guardrails.sanitize_output(answer)
            sanitized_answer = output_check.sanitized_text
            flags = output_check.triggered_rules
        answer_chunks = answer_contributors(sanitized_answer, reranked_chunks)

        matched_cells = []
        computed_deltas = []
        if answer_chunks:
            top_chunk = answer_chunks[0]
            raw_cells = TableASTParser.match_query_cells(
                question,
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

        verification = self.numeric_verifier.verify(sanitized_answer, answer_chunks)
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
        return QueryResponse(
            question=question,
            answer=sanitized_answer,
            matched_cells=matched_cells,
            computed_deltas=computed_deltas,
            citations=citations,
            verification=NumericVerificationSchema(**verification.to_dict()),
            cache_hit=cache_hit,
            cache_similarity=cache_similarity,
            latency_ms=latency_ms,
            guardrail_flags=flags,
            metadata={
                "retrieval_strategy": "hybrid_rrf_plus_table_schema_cross_encoder",
                "candidates_evaluated": candidates_evaluated,
                "document_scope": scope or "ALL",
                "answer_provider": self.llm_client.last_provider,
                "provider_error": self.llm_client.last_error,
            },
        )

    def validate(self, payload: QueryRequest) -> str:
        """Validate input before a streaming response is started."""
        input_check = self.guardrails.validate_input(payload.question)
        if not input_check.passed:
            raise QueryBlockedError(input_check.triggered_rules)
        return input_check.sanitized_text

    async def stream(
        self, payload: QueryRequest
    ) -> AsyncGenerator[str | QueryResponse, None]:
        """Stream redacted answer tokens, then emit the final verified response."""
        started_at = time.perf_counter()
        question = self.validate(payload)
        self.ensure_indexed()
        indexed_docs = self.retriever.indexed_document_names()
        scope = self.normalize_scope(payload.doc_name) or self.infer_scope(
            question, indexed_docs
        )
        cache_key = f"{scope or 'ALL'}::top_k={payload.top_k}::{question}"
        if payload.use_cache:
            cached = self.semantic_cache.lookup(cache_key)
            if cached is not None:
                cached["latency_ms"] = round(
                    (time.perf_counter() - started_at) * 1000.0, 2
                )
                response = QueryResponse(**cached)
                for word in response.answer.split(" "):
                    yield word + " "
                cached["latency_ms"] = round(
                    (time.perf_counter() - started_at) * 1000.0, 2
                )
                yield QueryResponse(**cached)
                return

        vocabulary = self.retriever.corpus_vocabulary()
        normalized_question = correct_query_terms(question, vocabulary)
        retrieval_query = expand_query(normalized_question)
        hybrid_candidates, reranked_chunks = self._select_evidence(
            question=normalized_question,
            retrieval_query=retrieval_query,
            payload=payload,
            scope=scope,
            indexed_docs=indexed_docs,
        )
        redactor = self.guardrails.streaming_redactor()
        async for token in self.llm_client.stream_answer(
            normalized_question,
            reranked_chunks,
            indexed_docs=indexed_docs,
        ):
            safe_text = redactor.feed(token)
            if safe_text:
                yield safe_text
        safe_tail = redactor.finish()
        if safe_tail:
            yield safe_tail

        response = self._build_response(
            question=question,
            answer=redactor.sanitized_text,
            reranked_chunks=reranked_chunks,
            scope=scope,
            candidates_evaluated=len(hybrid_candidates),
            start_time=started_at,
            already_sanitized=True,
            guardrail_flags=redactor.triggered_rules,
        )
        if payload.use_cache and reranked_chunks:
            self.semantic_cache.store(cache_key, response.model_dump())
        yield response
