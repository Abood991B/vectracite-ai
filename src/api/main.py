from contextlib import asynccontextmanager
import base64
import json
import time
from typing import Any, AsyncIterator, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from src.api.ui import WORKBENCH_HTML
from evaluation.run_ablation_benchmark import (
    build_dynamic_eval_queries,
    load_corpus_chunks_from_pdfs,
    run_evaluation_suite,
)
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
from src.ingestion.pdf_parser import MultimodalFinancialParser, render_pages_to_pdf_bytes
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.table_ast import TableASTParser

parser = MultimodalFinancialParser(chunk_size=1200, chunk_overlap=200)
retriever = HybridRetriever(
    in_memory=settings.QDRANT_IN_MEMORY,
)
reranker = CrossEncoderReranker()
semantic_cache = SemanticCache()
guardrails = EnterpriseGuardrails()
numeric_verifier = DeterministicNumericVerifier()
llm_client = UnifiedLLMClient()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Index the on-disk PDF corpus and warm up both production BGE models."""
    if not retriever.chunks_store:
        corpus_chunks = load_corpus_chunks_from_pdfs(parser)
        if corpus_chunks:
            retriever.index_chunks(corpus_chunks)
    warmup = retriever.retrieve_hybrid("warmup", top_k=1)
    reranker.rerank("warmup", warmup, top_n=1)
    yield


def _clean_doc_name(doc_name: str) -> str:
    """Normalize a document identifier before using it as a corpus filename."""
    clean_name = doc_name.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    if clean_name.lower().endswith(".pdf"):
        clean_name = clean_name[:-4]
    if not clean_name:
        raise HTTPException(status_code=400, detail="Document name cannot be empty.")
    return clean_name


def _persist_pdf(doc_name: str, pdf_bytes: bytes) -> None:
    """Persist a valid PDF in the same directory used by startup and evaluation."""
    settings.CORPUS_PDF_DIR.mkdir(parents=True, exist_ok=True)
    (settings.CORPUS_PDF_DIR / f"{doc_name}.pdf").write_bytes(pdf_bytes)


def _ensure_indexed() -> None:
    """Load the real corpus lazily for clients that do not trigger lifespan hooks."""
    if not retriever.chunks_store:
        corpus_chunks = load_corpus_chunks_from_pdfs(parser)
        if corpus_chunks:
            retriever.index_chunks(corpus_chunks)


def _benchmark_definitions() -> List[Dict[str, Any]]:
    queries_path = settings.DATA_DIR / "benchmark_queries.json"
    if not queries_path.exists():
        return []
    return json.loads(queries_path.read_text(encoding="utf-8"))


def _build_scenarios() -> List[Dict[str, Any]]:
    """Build scenario cards from benchmark JSON plus live table schemas."""
    definitions = _benchmark_definitions()
    indexed_docs = {document["doc_name"] for document in retriever.list_documents()}
    scenarios: List[Dict[str, Any]] = []
    covered_docs: set[str] = set()

    for definition in definitions:
        doc_name = str(definition["doc_name"])
        scenarios.append(
            {
                "id": f"benchmark:{doc_name}:{definition['page_number']}",
                "doc_name": doc_name,
                "page_number": int(definition["page_number"]),
                "question": str(definition["question"]),
                "source": "benchmark",
            }
        )
        covered_docs.add(doc_name)

    live_cases = build_dynamic_eval_queries(retriever.chunks_store)
    for case in live_cases:
        doc_name = str(case["doc_name"])
        if doc_name in covered_docs:
            continue
        scenarios.append(
            {
                "id": f"live:{doc_name}:{case['page_number']}",
                "doc_name": doc_name,
                "page_number": int(case["page_number"]),
                "question": str(case["question"]),
                "source": "live_table_schema",
            }
        )
        covered_docs.add(doc_name)

    for doc_name in sorted(indexed_docs - covered_docs):
        chunks = retriever.get_document_chunks(doc_name)
        table_chunk = next((chunk for chunk in chunks if chunk["contains_table"]), None)
        if table_chunk:
            cells = TableASTParser.extract_cells(
                table_chunk["text"], doc_name, int(table_chunk["page_number"])
            )
            if cells:
                first_cell = cells[0]
                question = (
                    f"In {doc_name}, what was {first_cell.row_key} in "
                    f"{first_cell.col_header}?"
                )
            else:
                question = f"What does the indexed table in {doc_name} show?"
            page_number = int(table_chunk["page_number"])
        else:
            question = f"Summarize the indexed evidence in {doc_name}."
            page_number = int(chunks[0]["page_number"]) if chunks else 1

        scenarios.append(
            {
                "id": f"indexed:{doc_name}:{page_number}",
                "doc_name": doc_name,
                "page_number": page_number,
                "question": question,
                "source": "indexed_document",
            }
        )

    return scenarios


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
    _ensure_indexed()
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
    _ensure_indexed()
    docs = retriever.list_documents()
    return {"total_documents": len(docs), "documents": docs}


@app.get("/documents/{doc_name}")
async def inspect_document(doc_name: str) -> Dict[str, Any]:
    """Returns all indexed pages, chunks, and extracted Markdown tables for `doc_name`."""
    _ensure_indexed()
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
        clean_name = _clean_doc_name(payload.doc_name)
        retriever.delete_document(clean_name)
        chunks = parser.parse_pdf_bytes(raw_bytes, doc_name=clean_name)
        if not chunks:
            raise ValueError("No readable text or tables found in PDF.")
        _persist_pdf(clean_name, raw_bytes)
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
    """Persist and index structured pages as a real PDF document."""
    clean_name = _clean_doc_name(payload.doc_name)
    raw_pages = [{"page_number": p.page_number, "text": p.text} for p in payload.pages]
    chunks = parser.parse_raw_pages(doc_name=clean_name, pages=raw_pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text or tables found in document.")
    retriever.delete_document(clean_name)
    _persist_pdf(clean_name, render_pages_to_pdf_bytes(clean_name, raw_pages))
    indexed_count = retriever.index_chunks(chunks)
    return {
        "status": "indexed",
        "doc_name": clean_name,
        "chunks_created": indexed_count,
        "table_chunks": sum(1 for c in chunks if c.contains_table),
    }


@app.put("/documents/{doc_name}")
async def update_or_replace_document(doc_name: str, payload: IngestRequest) -> Dict[str, Any]:
    """Atomically replaces an indexed document and its persisted PDF."""
    clean_name = _clean_doc_name(doc_name)
    removed = retriever.delete_document(clean_name)
    raw_pages = [{"page_number": p.page_number, "text": p.text} for p in payload.pages]
    chunks = parser.parse_raw_pages(doc_name=clean_name, pages=raw_pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text or tables found in document.")
    _persist_pdf(clean_name, render_pages_to_pdf_bytes(clean_name, raw_pages))
    indexed_count = retriever.index_chunks(chunks)
    return {
        "status": "updated",
        "doc_name": clean_name,
        "previous_chunks_removed": removed,
        "new_chunks_indexed": indexed_count,
    }


@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str) -> Dict[str, Any]:
    """Deletes a document and all of its chunks from Qdrant + BM25."""
    clean_name = _clean_doc_name(doc_name)
    removed = retriever.delete_document(clean_name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{clean_name}' not found.")

    baseline_docs = {str(item["doc_name"]) for item in _benchmark_definitions()}
    if clean_name not in baseline_docs:
        corpus_path = settings.CORPUS_PDF_DIR / f"{clean_name}.pdf"
        if corpus_path.exists():
            corpus_path.unlink()
    return {"status": "deleted", "doc_name": clean_name, "chunks_removed": removed}


@app.get("/scenarios")
async def list_query_scenarios() -> Dict[str, Any]:
    """Return benchmark and live schema scenarios for every indexed document."""
    _ensure_indexed()
    scenarios = _build_scenarios()
    return {"total_scenarios": len(scenarios), "scenarios": scenarios}


@app.get("/evaluate/latest")
async def latest_evaluation() -> Dict[str, Any]:
    """Return the latest persisted evaluation, creating it when absent."""
    _ensure_indexed()
    report_path = settings.DATA_DIR / "latest_evaluation_report.json"
    if not report_path.exists():
        return run_evaluation_suite(retriever.chunks_store)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_docs = {row["doc_name"] for row in report.get("per_query_breakdown", [])}
    live_docs = {case["doc_name"] for case in build_dynamic_eval_queries(retriever.chunks_store)}
    if report_docs != live_docs:
        return run_evaluation_suite(retriever.chunks_store)
    return report


@app.post("/evaluate")
async def evaluate_current_index() -> Dict[str, Any]:
    """Run a live ablation across all currently indexed documents."""
    _ensure_indexed()
    return run_evaluation_suite(retriever.chunks_store)


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

    _ensure_indexed()

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

    _ensure_indexed()

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
