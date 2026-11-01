from contextlib import asynccontextmanager
import base64
import time
from typing import Any, AsyncIterator, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from src.api.ui import WORKBENCH_HTML
from evaluation.run_ablation_benchmark import build_seed_chunks
from src.cache_and_guardrails.guardrails import EnterpriseGuardrails
from src.cache_and_guardrails.numeric_verifier import DeterministicNumericVerifier
from src.cache_and_guardrails.semantic_cache import SemanticCache
from src.config import settings
from src.inference.llm_client import UnifiedLLMClient
from src.inference.schemas import (
    CitationItem,
    IngestRequest,
    NumericVerificationSchema,
    PdfUploadRequest,
    RowDeltaSchema,
    TableCellSchema,
    QueryRequest,
    QueryResponse,
)
from src.ingestion.pdf_parser import MultimodalFinancialParser
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.table_ast import TableASTParser

parser = MultimodalFinancialParser(chunk_size=1200, chunk_overlap=200)
retriever = HybridRetriever(
    in_memory=settings.QDRANT_IN_MEMORY,
    use_lightweight_embedder=settings.USE_LIGHTWEIGHT_MODELS,
)
reranker = CrossEncoderReranker(
    use_lightweight_scorer=settings.USE_LIGHTWEIGHT_MODELS,
)
semantic_cache = SemanticCache()
guardrails = EnterpriseGuardrails()
numeric_verifier = DeterministicNumericVerifier()
llm_client = UnifiedLLMClient()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Pre-indexes multi-industry seed corpus and warms up BGE + Cross-Encoder models on startup."""
    if not retriever.chunks_store:
        seed_chunks = build_seed_chunks(parser)
        retriever.index_chunks(seed_chunks)
    warmup = retriever.retrieve_hybrid("warmup", top_k=1)
    reranker.rerank("warmup", warmup, top_n=1)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    description=(
        "Universal Multimodal Document Intelligence, Hybrid RAG (Qdrant + BM25 RRF), "
        "Table-Schema Cross-Encoder Reranking, Deterministic Numeric Verification & vLLM Engine."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/", response_class=HTMLResponse)
async def interactive_dashboard() -> HTMLResponse:
    """Serves the clean VectraCite-AI engineering workbench."""
    return HTMLResponse(content=WORKBENCH_HTML)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Liveness and readiness probe for Kubernetes & Cloud Run."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "indexed_chunks": len(retriever.chunks_store),
        "indexed_documents": len(retriever.list_documents()),
        "cache": semantic_cache.stats(),
    }


@app.get("/documents")
async def list_indexed_documents() -> Dict[str, Any]:
    """Lists all currently indexed documents, chunk counts, and table counts."""
    if not retriever.chunks_store:
        retriever.index_chunks(build_seed_chunks(parser))
    docs = retriever.list_documents()
    return {"total_documents": len(docs), "documents": docs}


@app.get("/documents/{doc_name}")
async def inspect_document(doc_name: str) -> Dict[str, Any]:
    """Returns all indexed pages, chunks, and extracted Markdown tables for `doc_name`."""
    if not retriever.chunks_store:
        retriever.index_chunks(build_seed_chunks(parser))
    chunks = retriever.get_document_chunks(doc_name)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
    return {
        "doc_name": doc_name,
        "total_chunks": len(chunks),
        "table_chunks": sum(1 for c in chunks if c["contains_table"]),
        "chunks": chunks,
    }


@app.post("/ingest/pdf")
async def ingest_pdf_file(payload: PdfUploadRequest) -> Dict[str, Any]:
    """Decodes a base64-encoded PDF file, extracts text and tables, and indexes it."""
    try:
        raw_bytes = base64.b64decode(payload.pdf_base64, validate=True)
        clean_name = (
            payload.doc_name[:-4]
            if payload.doc_name.lower().endswith(".pdf")
            else payload.doc_name
        )
        retriever.delete_document(clean_name)
        chunks = parser.parse_pdf_bytes(raw_bytes, doc_name=clean_name)
        if not chunks:
            raise ValueError("No readable text or tables found in PDF.")
        indexed_count = retriever.index_chunks(chunks)
        return {
            "status": "indexed_pdf",
            "doc_name": clean_name,
            "chunks_created": indexed_count,
            "table_chunks": sum(1 for c in chunks if c.contains_table),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc


@app.post("/ingest")
async def ingest_document(payload: IngestRequest) -> Dict[str, Any]:
    """Appends new document pages and Markdown tables into Qdrant + BM25."""
    raw_pages = [{"page_number": p.page_number, "text": p.text} for p in payload.pages]
    chunks = parser.parse_raw_pages(doc_name=payload.doc_name, pages=raw_pages)
    indexed_count = retriever.index_chunks(chunks)
    return {
        "status": "indexed",
        "doc_name": payload.doc_name,
        "chunks_created": indexed_count,
        "table_chunks": sum(1 for c in chunks if c.contains_table),
    }


@app.put("/documents/{doc_name}")
async def update_or_replace_document(doc_name: str, payload: IngestRequest) -> Dict[str, Any]:
    """Atomically replaces or updates an existing document in Qdrant + BM25."""
    removed = retriever.delete_document(doc_name)
    raw_pages = [{"page_number": p.page_number, "text": p.text} for p in payload.pages]
    chunks = parser.parse_raw_pages(doc_name=doc_name, pages=raw_pages)
    indexed_count = retriever.index_chunks(chunks)
    return {
        "status": "updated",
        "doc_name": doc_name,
        "previous_chunks_removed": removed,
        "new_chunks_indexed": indexed_count,
    }


@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str) -> Dict[str, Any]:
    """Deletes a document and all of its chunks from Qdrant + BM25."""
    removed = retriever.delete_document(doc_name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
    return {"status": "deleted", "doc_name": doc_name, "chunks_removed": removed}


@app.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    """
    End-to-end RAG execution pipeline:
      1. Input Prompt-Injection Guardrail Check
      2. Redis Semantic Cache Lookup (<5ms fast path)
      3. Qdrant Dense + BM25 Sparse Hybrid Retrieval (Reciprocal Rank Fusion)
      4. Table-Schema Calibrated Cross-Encoder Reranking (BAAI/bge-reranker-base)
      5. Citation-Grounded LLM Synthesis + Deterministic Numeric Verification + PII Redaction
    """
    start_time = time.perf_counter()

    input_check = guardrails.validate_input(payload.question)
    if not input_check.passed:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Query blocked by Enterprise AI Guardrails.",
                "triggered_rules": input_check.triggered_rules,
            },
        )

    if payload.use_cache:
        cached = semantic_cache.lookup(input_check.sanitized_text)
        if cached is not None:
            cached["latency_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            return QueryResponse(**cached)

    if not retriever.chunks_store:
        retriever.index_chunks(build_seed_chunks(parser))

    hybrid_candidates = retriever.retrieve_hybrid(
        query=input_check.sanitized_text, top_k=max(payload.top_k * 2, 6)
    )

    reranked_chunks = reranker.rerank(
        query=input_check.sanitized_text,
        candidates=hybrid_candidates,
        top_n=payload.top_k,
    )

    matched_cells = []
    computed_deltas = []
    if reranked_chunks:
        top_c = reranked_chunks[0]
        raw_cells = TableASTParser.match_query_cells(
            input_check.sanitized_text,
            top_c.text,
            doc_name=top_c.doc_name,
            page_number=top_c.page_number,
        )
        matched_cells = [TableCellSchema(**cell.to_dict()).model_dump() for cell in raw_cells]
        computed_deltas = [
            RowDeltaSchema(**delta).model_dump()
            for delta in TableASTParser.compute_row_deltas(raw_cells)
        ]

    raw_answer = await llm_client.generate_answer(input_check.sanitized_text, reranked_chunks)
    output_check = guardrails.sanitize_output(raw_answer)
    verification = numeric_verifier.verify(output_check.sanitized_text, reranked_chunks)

    citations: List[CitationItem] = [
        CitationItem(
            chunk_id=c.chunk_id,
            doc_name=c.doc_name,
            page_number=c.page_number,
            contains_table=c.contains_table,
            rerank_score=round(float(c.rerank_score or c.rrf_score), 4),
            excerpt=c.text[:280],
        )
        for c in reranked_chunks
    ]

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    response_data = {
        "question": input_check.sanitized_text,
        "answer": output_check.sanitized_text,
        "matched_cells": matched_cells,
        "computed_deltas": computed_deltas,
        "citations": [c.model_dump() for c in citations],
        "verification": NumericVerificationSchema(**verification.to_dict()).model_dump(),
        "cache_hit": False,
        "cache_similarity": None,
        "latency_ms": latency_ms,
        "guardrail_flags": output_check.triggered_rules,
        "metadata": {
            "retrieval_strategy": "hybrid_rrf_plus_table_schema_cross_encoder",
            "candidates_evaluated": len(hybrid_candidates),
        },
    }

    if payload.use_cache and reranked_chunks:
        semantic_cache.store(input_check.sanitized_text, response_data)

    return QueryResponse(**response_data)


@app.post("/query/stream")
async def stream_query_documents(payload: QueryRequest) -> StreamingResponse:
    """Streams citation-grounded tokens via Server-Sent Events (SSE)."""
    input_check = guardrails.validate_input(payload.question)
    if not input_check.passed:
        raise HTTPException(status_code=400, detail=input_check.triggered_rules)

    if not retriever.chunks_store:
        retriever.index_chunks(build_seed_chunks(parser))

    candidates = retriever.retrieve_hybrid(input_check.sanitized_text, top_k=payload.top_k * 2)
    reranked = reranker.rerank(input_check.sanitized_text, candidates, top_n=payload.top_k)
    return StreamingResponse(
        llm_client.stream_answer(input_check.sanitized_text, reranked),
        media_type="text/event-stream",
    )


@app.get("/metrics")
async def service_metrics() -> Dict[str, Any]:
    """Exposes operational telemetry for monitoring dashboards."""
    return {
        "indexed_chunks": len(retriever.chunks_store),
        "indexed_documents": len(retriever.list_documents()),
        "semantic_cache": semantic_cache.stats(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "llm_model": settings.LLM_MODEL_NAME,
    }
