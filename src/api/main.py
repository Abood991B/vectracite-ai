from contextlib import asynccontextmanager
import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import logging
import threading
from typing import Any, AsyncIterator, Callable, Dict, List, TypeVar
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from src.api.answer_service import AnswerService
from src.api.distributed_lock import distributed_mutation_lock
from src.api.index_state import bump_index_generation, read_index_generation
from src.api.middleware import BodySizeLimitMiddleware
from src.api.security import require_admin_access, require_read_access
from src.api.ui import WORKBENCH_HTML
from evaluation.run_ablation_benchmark import load_corpus_chunks_from_pdfs
from src.adapters.evaluation_runner import build_live_cases, compute_signature, run_suite
from src.adapters.page_images import InvalidPdf, PageImageRenderer, PageOutOfRange, count_pdf_pages
from src.adapters.uploads import PdfFileStore
from src.usecases.documents import commit_document, load_persisted_chunks
from src.usecases.scenarios import build_scenarios
from src.usecases.scoping import (
    InvalidDocumentNameError,
    clean_document_name,
    infer_scope_from_question,
    normalize_scope,
)
from src.cache_and_guardrails.guardrails import EnterpriseGuardrails
from src.cache_and_guardrails.numeric_verifier import DeterministicNumericVerifier
from src.cache_and_guardrails.semantic_cache import SemanticCache
from src.config import settings
from src.inference.llm_client import UnifiedLLMClient
from src.inference.schemas import (
    IngestRequest,
    LLMConfigRequest,
    PdfUploadRequest,
    QueryRequest,
    QueryResponse,
)
from src.ingestion.ocr import build_ocr_backend
from src.ingestion.pdf_parser import (
    MultimodalFinancialParser,
    PDFRenderError,
    PDFRenderLimitError,
    render_pages_to_pdf_bytes,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

parser = MultimodalFinancialParser(
    chunk_size=settings.PARSER_CHUNK_SIZE,
    chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
    ocr_backend=build_ocr_backend(),
    ocr_dpi=settings.OCR_DPI,
    ocr_min_chars=settings.OCR_MIN_CHARS,
    ocr_max_pages=settings.OCR_MAX_PAGES,
)
upload_store = PdfFileStore(settings.USER_UPLOADS_DIR, settings.CORPUS_PDF_DIR)
retriever = HybridRetriever(
    in_memory=settings.QDRANT_IN_MEMORY,
    use_lightweight_embedder=settings.USE_LIGHTWEIGHT_MODELS,
)
reranker = CrossEncoderReranker(
    use_lightweight_scorer=settings.USE_LIGHTWEIGHT_MODELS,
    cross_encoder_gate=settings.RERANKER_GATE,
    confidence_threshold=settings.RERANKER_CONFIDENCE_THRESHOLD,
)
semantic_cache = SemanticCache()
guardrails = EnterpriseGuardrails()
numeric_verifier = DeterministicNumericVerifier()
llm_client = UnifiedLLMClient()
DEMO_DOCUMENT_NAMES = frozenset(
    {
        "AMD_2022_10K",
        "MSFT_2023_10K",
        "JPM_2022_10K",
        "ONCOLOGY_TRIAL_2025",
    }
)


def demo_document_names() -> frozenset[str]:
    """Return the demo shelf from the corpus manifest, with a safe fallback."""
    manifest_path = settings.CORPUS_PDF_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            documents = manifest.get("documents") or {}
            if documents:
                return frozenset(str(name) for name in documents)
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring malformed corpus manifest %s", manifest_path)
    stems = frozenset(path.stem for path in settings.CORPUS_PDF_DIR.glob("*.pdf"))
    return stems or DEMO_DOCUMENT_NAMES


T = TypeVar("T")
logger = logging.getLogger(__name__)
_mutation_lock = asyncio.Lock()
_ingestion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-ingestion")
_ingestion_slot = threading.BoundedSemaphore(1)
page_images = PageImageRenderer()


@asynccontextmanager
async def _mutation_guard() -> AsyncIterator[None]:
    """Serialize a mutation against local handlers, then against other replicas.

    Local requests queue on the asyncio lock before asking Redis for the lease, so a waiting
    handler never holds the cross-process lease. The lease is re-checked after acquisition
    because a holder that lost it mid-write must fail instead of writing through.
    """
    async with _mutation_lock:
        async with distributed_mutation_lock() as lease:
            lease.require_lease()
            yield


def _cors_origins() -> list[str]:
    """Return the explicit browser origins permitted to call the API."""
    return [origin.strip() for origin in settings.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]


async def _run_ingestion_step(operation: Callable[..., T], *args: Any) -> T:
    """Stop timed-out parsing between pages and retain the slot until it stops."""
    if not _ingestion_slot.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another document is being processed.")
    cancelled = threading.Event()
    try:
        future = _ingestion_executor.submit(operation, *args, cancel_event=cancelled)
    except BaseException:
        _ingestion_slot.release()
        raise
    future.add_done_callback(lambda _future: _ingestion_slot.release())
    wrapped = asyncio.wrap_future(future)
    wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    try:
        return await asyncio.wait_for(
            asyncio.shield(wrapped), timeout=settings.INGEST_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        cancelled.set()
        raise HTTPException(status_code=408, detail="Document processing timed out.") from exc
    except asyncio.CancelledError:
        cancelled.set()
        raise


def _validate_structured_pages(pages: list[Any]) -> None:
    if len(pages) > settings.MAX_STRUCTURED_PAGES:
        raise HTTPException(status_code=413, detail="Document page limit exceeded.")
    if sum(len(page.text) for page in pages) > settings.MAX_STRUCTURED_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="Document text limit exceeded.")


def _render_and_parse_document(
    clean_name: str, raw_pages: List[Dict[str, Any]], *, cancel_event: threading.Event
) -> tuple[bytes, List[Any]]:
    """Render once, then parse the exact bytes that will be persisted."""
    pdf_bytes = render_pages_to_pdf_bytes(
        clean_name, raw_pages, max_pages=settings.MAX_RENDERED_PDF_PAGES
    )
    return pdf_bytes, parser.parse_pdf_bytes(pdf_bytes, clean_name, cancel_event=cancel_event)


def _parse_uploaded_pdf(
    raw_bytes: bytes, clean_name: str, *, cancel_event: threading.Event,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    coverage: list[dict[str, Any]] = []
    chunks = parser.parse_pdf_bytes(
        raw_bytes, clean_name, coverage=coverage, cancel_event=cancel_event, progress=progress
    )
    return chunks, coverage


def _upload_result(clean_name: str, chunks: list[Any], coverage: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "indexed_pdf",
        "doc_name": clean_name,
        "chunks_created": len(chunks),
        "table_chunks": sum(1 for c in chunks if c.contains_table),
        "coverage": {
            "pages": coverage,
            "pages_without_text": [p["page"] for p in coverage if p["status"] in {"image_without_text", "no_text"}],
            "pages_with_unrecognized_images": [p["page"] for p in coverage if p["contains_images"] and p["status"] == "native_text"],
        },
    }


def _decode_pdf_upload(payload: PdfUploadRequest) -> tuple[bytes, str]:
    maximum_encoded_size = ((settings.MAX_PDF_BYTES + 2) // 3) * 4
    if len(payload.pdf_base64) > maximum_encoded_size:
        raise HTTPException(status_code=413, detail="PDF size limit exceeded.")
    try:
        raw_bytes = base64.b64decode(payload.pdf_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Failed to decode PDF.") from exc
    _validate_pdf_limits(raw_bytes)
    return raw_bytes, _clean_doc_name(payload.doc_name)


def _validate_pdf_limits(raw_bytes: bytes) -> None:
    if len(raw_bytes) > settings.MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF size limit exceeded.")
    try:
        page_count = count_pdf_pages(raw_bytes)
    except InvalidPdf as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if page_count > settings.MAX_PDF_PAGES:
        raise HTTPException(status_code=413, detail="PDF page limit exceeded.")


def _load_persisted_chunks(include_defaults: bool | None = None) -> List[Any]:
    """Load defaults plus user PDFs, letting a user copy override a default by name."""
    return load_persisted_chunks(
        parser=parser,
        corpus_loader=load_corpus_chunks_from_pdfs,
        uploads=upload_store,
        include_defaults=include_defaults,
    )


async def _periodic_index_sync() -> None:
    """Refresh the local BM25/catalog view only when the shared index changes."""
    interval = max(1.0, settings.INDEX_SYNC_SECONDS)
    last_generation = await asyncio.to_thread(read_index_generation)
    while True:
        await asyncio.sleep(interval)
        try:
            generation = await asyncio.to_thread(read_index_generation)
            if generation is not None:
                if last_generation is None:
                    last_generation = generation
                    continue
                if generation == last_generation:
                    continue
                await asyncio.to_thread(retriever.sync_from_qdrant, force=True)
                last_generation = generation
                continue
            remote_count = await asyncio.to_thread(retriever.count_points)
            if remote_count != len(retriever.chunks_store):
                await asyncio.to_thread(retriever.sync_from_qdrant, force=True)
        except Exception:
            logger.exception("Periodic Qdrant catalog sync failed")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Validate security, hydrate shared state, and warm up the retrieval stack."""
    settings.validate_auth_configuration()
    _ensure_indexed()
    warmup = retriever.retrieve_hybrid("warmup", top_k=1)
    reranker.rerank("warmup", warmup, top_n=1)
    sync_task = asyncio.create_task(_periodic_index_sync())
    try:
        yield
    finally:
        sync_task.cancel()
        try:
            await sync_task
        except asyncio.CancelledError:
            pass
        await llm_client.aclose()


def _clean_doc_name(doc_name: str) -> str:
    """Map a domain filename refusal to the HTTP status the API promises."""
    try:
        return clean_document_name(doc_name)
    except InvalidDocumentNameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _normalize_scope(doc_name: str | None) -> str | None:
    """Map a domain scope refusal to the HTTP status the API promises."""
    try:
        return normalize_scope(doc_name)
    except InvalidDocumentNameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _ensure_indexed() -> None:
    """Load or hydrate a usable catalog for clients that do not trigger lifespan."""
    if retriever.chunks_store:
        return
    if retriever.backend != "memory" or retriever.shared_client:
        try:
            if retriever.count_points() > 0:
                retriever.hydrate_from_qdrant()
                if retriever.chunks_store:
                    return
        except Exception:
            logger.exception("Unable to hydrate the shared Qdrant catalog")
    corpus_chunks = _load_persisted_chunks()
    if corpus_chunks:
        retriever.index_chunks(
            corpus_chunks,
            use_embedding_cache=True,
            write_embedding_cache=True,
        )


def _benchmark_definitions() -> List[Dict[str, Any]]:
    queries_path = settings.DATA_DIR / "benchmark_queries.json"
    if not queries_path.exists():
        return []
    return json.loads(queries_path.read_text(encoding="utf-8"))


def _build_scenarios() -> List[Dict[str, Any]]:
    """Build scenario cards from benchmark JSON plus live table schemas."""
    return build_scenarios(
        definitions=_benchmark_definitions(),
        retriever=retriever,
        live_cases=build_live_cases(retriever.chunks_store),
    )


def _commit_document(clean_name: str, chunks: List[Any], pdf_bytes: bytes) -> int:
    """Persist bytes and swap the indexed version, restoring both on failure."""
    return commit_document(
        doc_name=clean_name,
        chunks=chunks,
        pdf_bytes=pdf_bytes,
        catalog=retriever,
        uploads=upload_store,
        invalidate_cache=semantic_cache.clear,
        notify_index_changed=bump_index_generation,
    )


answer_service = AnswerService(
    retriever=retriever,
    reranker=reranker,
    semantic_cache=semantic_cache,
    guardrails=guardrails,
    numeric_verifier=numeric_verifier,
    llm_client=llm_client,
    ensure_indexed=_ensure_indexed,
    normalize_scope=_normalize_scope,
    infer_scope=infer_scope_from_question,
)


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
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)
app.add_middleware(BodySizeLimitMiddleware)




@app.get("/", response_class=HTMLResponse)
async def interactive_dashboard() -> HTMLResponse:
    """Serves the clean VectraCite-AI engineering workbench."""
    return HTMLResponse(content=WORKBENCH_HTML)


@app.get("/assets/logo.svg")
async def logo_asset() -> FileResponse:
    """Serve the official VectraCite-AI vector mark."""
    logo_path = settings.BASE_DIR / "assets" / "logo.svg"
    if not logo_path.exists():
        raise HTTPException(status_code=404, detail="Logo asset not found.")
    return FileResponse(logo_path, media_type="image/svg+xml")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_asset() -> FileResponse:
    """Serve the same vector mark for browsers that request a favicon.ico."""
    logo_path = settings.BASE_DIR / "assets" / "logo.svg"
    if not logo_path.exists():
        raise HTTPException(status_code=404, detail="Logo asset not found.")
    return FileResponse(logo_path, media_type="image/svg+xml")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Process liveness probe; dependency readiness lives at /ready."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "indexed_chunks": len(retriever.chunks_store),
        "indexed_documents": len(retriever.list_documents()),
        "cache": semantic_cache.stats(),
    }


@app.get("/ready")
async def readiness_check() -> Response:
    """Readiness probe for required dependencies, including an empty catalog."""
    _ensure_indexed()
    health = retriever.health()
    semantic_cache.check_health()
    cache = semantic_cache.stats()
    failures: List[str] = []
    if health["degraded"]:
        failures.append("vector_backend_degraded")
    if settings.REQUIRE_REDIS and cache["backend"] != "redis":
        failures.append("redis_unavailable")
    payload = {
        "status": "ready" if not failures else "not_ready",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "failures": failures,
        "indexed_chunks": len(retriever.chunks_store),
        "indexed_documents": len(retriever.list_documents()),
        "retriever": health,
        "cache": cache,
    }
    return Response(
        content=json.dumps(payload),
        media_type="application/json",
        status_code=200 if not failures else 503,
    )


@app.get("/config/llm", dependencies=[Depends(require_read_access)])
async def get_llm_config() -> Dict[str, Any]:
    """Return safe runtime provider metadata without exposing the API key."""
    return llm_client.get_provider_status()


@app.post("/config/llm", dependencies=[Depends(require_admin_access)])
async def configure_llm(payload: LLMConfigRequest) -> Dict[str, Any]:
    """Update the optional OpenAI-compatible provider without restarting."""
    return llm_client.configure_provider(
        base_url=payload.base_url,
        api_key=payload.api_key,
        model_name=payload.model_name,
        enabled=payload.enabled,
    )


@app.post("/config/llm/test", dependencies=[Depends(require_admin_access)])
async def test_llm_connection() -> Dict[str, Any]:
    """Probe the configured provider and report reachability without running a completion."""
    return await llm_client.test_connection()


@app.get("/documents", dependencies=[Depends(require_read_access)])
async def list_indexed_documents() -> Dict[str, Any]:
    """Lists all currently indexed documents, chunk counts, and table counts."""
    _ensure_indexed()
    docs = retriever.list_documents()
    for document in docs:
        document["storage_tier"] = (
            "user_upload"
            if (settings.USER_UPLOADS_DIR / f"{document['doc_name']}.pdf").exists()
            else "demo_corpus"
        )
    return {"total_documents": len(docs), "documents": docs}


@app.get("/documents/{doc_name}/pdf", dependencies=[Depends(require_read_access)])
async def document_pdf(doc_name: str) -> FileResponse:
    """Return the original PDF for inline browser preview or download."""
    clean_name = _clean_doc_name(doc_name)
    pdf_path = upload_store.find(clean_name)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail=f"Document '{clean_name}' PDF not found.")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
        content_disposition_type="inline",
    )


@app.get(
    "/documents/{doc_name}/pages/{page_number}/image",
    dependencies=[Depends(require_read_access)],
)
async def document_page_image(
    doc_name: str,
    page_number: int,
    request: Request,
    highlight: str = "",
) -> Response:
    """Render one source page as a high-DPI PNG with query-term highlights."""
    clean_name = _clean_doc_name(doc_name)
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be positive.")
    pdf_path = upload_store.find(clean_name)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail=f"Document '{clean_name}' PDF not found.")

    try:
        rendered = await asyncio.to_thread(
            page_images.render, pdf_path, clean_name, page_number, highlight
        )
    except PageOutOfRange as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    headers = {"ETag": rendered.etag, "Cache-Control": "private, max-age=3600"}
    if request.headers.get("if-none-match") == rendered.etag:
        return Response(status_code=304, headers=headers)
    return Response(content=rendered.png, media_type="image/png", headers=headers)


@app.post("/documents/clear-demos", dependencies=[Depends(require_admin_access)])
async def clear_demo_documents() -> Dict[str, Any]:
    """Remove only demo documents from the active index and preserve user uploads."""
    async with _mutation_guard():
        user_document_names = {
            path.stem for path in settings.USER_UPLOADS_DIR.glob("*.pdf")
        }
        removed_chunks = 0
        for doc_name in demo_document_names() - user_document_names:
            removed_chunks += retriever.delete_document(doc_name)
        semantic_cache.clear()
        upload_store.mark_defaults_cleared()
        await asyncio.to_thread(bump_index_generation)
    return {
        "status": "cleared_demos",
        "documents": len(retriever.list_documents()),
        "chunks_removed": removed_chunks,
        "uploads_removed": 0,
        "uploads_preserved": len(user_document_names),
    }


@app.post("/documents/clear-all", dependencies=[Depends(require_admin_access)])
async def clear_all_documents() -> Dict[str, Any]:
    """Remove every indexed document, uploaded PDF, and cached answer."""
    async with _mutation_guard():
        upload_paths = sorted(settings.USER_UPLOADS_DIR.glob("*.pdf"))
        retriever.clear()
        semantic_cache.clear()
        removed_files = 0
        for path in upload_paths:
            try:
                path.unlink(missing_ok=True)
                removed_files += 1
            except OSError:
                logger.exception("Unable to remove uploaded PDF %s", path.name)
        upload_store.mark_defaults_cleared()
        await asyncio.to_thread(bump_index_generation)
    return {
        "status": "cleared_all",
        "documents": len(retriever.list_documents()),
        "uploads_removed": removed_files,
    }


@app.post("/documents/restore-defaults", dependencies=[Depends(require_admin_access)])
async def restore_default_documents() -> Dict[str, Any]:
    """Restore demos while preserving and re-indexing every user upload."""
    async with _mutation_guard():
        persisted_chunks = _load_persisted_chunks(include_defaults=True)
        retriever.clear()
        semantic_cache.clear()
        indexed_count = (
            retriever.index_chunks(
                persisted_chunks,
                use_embedding_cache=True,
                write_embedding_cache=True,
            )
            if persisted_chunks
            else 0
        )
        upload_store.clear_defaults_marker()
        await asyncio.to_thread(bump_index_generation)
    return {
        "status": "restored_defaults",
        "documents": len(retriever.list_documents()),
        "chunks_indexed": indexed_count,
    }


@app.get("/documents/{doc_name}", dependencies=[Depends(require_read_access)])
async def inspect_document(
    doc_name: str,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    """Returns paginated indexed pages, chunks, and extracted Markdown tables."""
    _ensure_indexed()
    chunks = retriever.get_document_chunks(doc_name)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
    page = chunks[offset : offset + limit]
    return {
        "doc_name": doc_name,
        "total_chunks": len(chunks),
        "table_chunks": sum(1 for c in chunks if c["contains_table"]),
        "offset": offset,
        "limit": limit,
        "returned_chunks": len(page),
        "truncated": offset + len(page) < len(chunks),
        "chunks": page,
    }


@app.post("/ingest/pdf", dependencies=[Depends(require_admin_access)])
async def ingest_pdf_file(payload: PdfUploadRequest) -> Dict[str, Any]:
    """Decode, validate, persist, and atomically index an uploaded PDF."""
    raw_bytes, clean_name = _decode_pdf_upload(payload)
    try:
        chunks, coverage = await _run_ingestion_step(
            _parse_uploaded_pdf, raw_bytes, clean_name
        )
        if not chunks:
            raise ValueError("No readable text or tables found in PDF.")
        async with _mutation_guard():
            await asyncio.to_thread(
                _commit_document, clean_name, chunks, raw_bytes
            )
        return _upload_result(clean_name, chunks, coverage)
    except HTTPException:
        raise
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc


@app.post("/ingest/pdf/stream", dependencies=[Depends(require_admin_access)])
async def ingest_pdf_stream(payload: PdfUploadRequest) -> StreamingResponse:
    """Stream page progress; disconnecting cancels parsing before indexing begins."""
    raw_bytes, clean_name = _decode_pdf_upload(payload)

    async def events() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        indexing = False

        def progress(page: int, total: int) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, {
                    "stage": "parsing", "page": page, "total_pages": total,
                })
            except RuntimeError:
                pass

        async def process() -> None:
            nonlocal indexing
            try:
                chunks, coverage = await _run_ingestion_step(
                    partial(_parse_uploaded_pdf, progress=progress), raw_bytes, clean_name
                )
                if not chunks:
                    raise ValueError("No readable text or tables found in PDF.")
                await queue.put({"stage": "indexing"})
                indexing = True
                async with _mutation_guard():
                    await asyncio.to_thread(_commit_document, clean_name, chunks, raw_bytes)
                await queue.put({"stage": "complete", "result": _upload_result(clean_name, chunks, coverage)})
            except asyncio.CancelledError:
                raise
            except HTTPException as exc:
                await queue.put({"stage": "error", "detail": str(exc.detail), "status": exc.status_code})
            except ValueError as exc:
                await queue.put({"stage": "error", "detail": str(exc), "status": 400})
            except Exception:
                logger.exception("Streaming document ingestion failed")
                await queue.put({"stage": "error", "detail": "Document processing failed.", "status": 500})
            finally:
                await queue.put(None)

        task = asyncio.create_task(process())
        try:
            yield ": processing\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": processing\n\n"
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            if not indexing and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ingest", dependencies=[Depends(require_admin_access)])
async def ingest_document(payload: IngestRequest) -> Dict[str, Any]:
    """Persist and atomically index structured pages as a real PDF document."""
    clean_name = _clean_doc_name(payload.doc_name)
    _validate_structured_pages(payload.pages)
    raw_pages = [{"page_number": p.page_number, "text": p.text} for p in payload.pages]
    try:
        pdf_bytes, chunks = await _run_ingestion_step(
            _render_and_parse_document, clean_name, raw_pages
        )
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text or tables found in document.",
            )
        async with _mutation_guard():
            await asyncio.to_thread(
                _commit_document, clean_name, chunks, pdf_bytes
            )
    except PDFRenderLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except PDFRenderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "status": "indexed",
        "doc_name": clean_name,
        "chunks_created": len(chunks),
        "table_chunks": sum(1 for c in chunks if c.contains_table),
    }


@app.put("/documents/{doc_name}", dependencies=[Depends(require_admin_access)])
async def update_or_replace_document(doc_name: str, payload: IngestRequest) -> Dict[str, Any]:
    """Atomically replace an indexed document and its persisted PDF."""
    clean_name = _clean_doc_name(doc_name)
    _validate_structured_pages(payload.pages)
    raw_pages = [{"page_number": p.page_number, "text": p.text} for p in payload.pages]
    try:
        pdf_bytes, chunks = await _run_ingestion_step(
            _render_and_parse_document, clean_name, raw_pages
        )
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text or tables found in document.",
            )
        async with _mutation_guard():
            removed = await asyncio.to_thread(
                _commit_document, clean_name, chunks, pdf_bytes
            )
    except PDFRenderLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except PDFRenderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "status": "updated",
        "doc_name": clean_name,
        "previous_chunks_removed": removed,
        "new_chunks_indexed": len(chunks),
    }


@app.delete("/documents/{doc_name}", dependencies=[Depends(require_admin_access)])
async def delete_document(doc_name: str) -> Dict[str, Any]:
    """Delete a document and all of its chunks from Qdrant and BM25."""
    clean_name = _clean_doc_name(doc_name)
    async with _mutation_guard():
        removed = retriever.delete_document(clean_name)
        if removed == 0:
            raise HTTPException(
                status_code=404, detail=f"Document '{clean_name}' not found."
            )
        upload_store.delete(clean_name, is_demo=clean_name in demo_document_names())
        semantic_cache.clear()
        await asyncio.to_thread(bump_index_generation)
    return {
        "status": "deleted",
        "doc_name": clean_name,
        "chunks_removed": removed,
        "documents_remaining": len(retriever.list_documents()),
    }


@app.get("/scenarios", dependencies=[Depends(require_read_access)])
async def list_query_scenarios() -> Dict[str, Any]:
    """Return benchmark and live schema scenarios for every indexed document."""
    _ensure_indexed()
    scenarios = _build_scenarios()
    return {"total_scenarios": len(scenarios), "scenarios": scenarios}


@app.get("/evaluate/latest", dependencies=[Depends(require_read_access)])
async def latest_evaluation() -> Dict[str, Any]:
    """Return the persisted evaluation only when it matches the current index."""
    _ensure_indexed()
    report_path = settings.DATA_DIR / "latest_evaluation_report.json"
    if not report_path.exists():
        return run_suite(retriever.chunks_store)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    current_signature = compute_signature(retriever.chunks_store)
    if report.get("evaluation_signature") != current_signature:
        return run_suite(retriever.chunks_store)
    return report


@app.post("/evaluate", dependencies=[Depends(require_admin_access)])
async def evaluate_current_index() -> Dict[str, Any]:
    """Run a live ablation across all currently indexed documents."""
    _ensure_indexed()
    return run_suite(retriever.chunks_store)


async def _execute_query(payload: QueryRequest) -> QueryResponse:
    """Execute the shared guarded RAG pipeline."""
    return await answer_service.execute(payload)


@app.post(
    "/query", response_model=QueryResponse, dependencies=[Depends(require_read_access)]
)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    """Execute the shared guarded RAG pipeline and return its structured result."""
    return await _execute_query(payload)


@app.post("/query/stream", dependencies=[Depends(require_read_access)])
async def stream_query_documents(payload: QueryRequest) -> StreamingResponse:
    """Stream redacted answer tokens, then the verified citations and metadata."""
    answer_service.validate(payload)

    async def event_stream() -> AsyncIterator[str]:
        async for item in answer_service.stream(payload):
            if isinstance(item, QueryResponse):
                serialized = item.model_dump()
                verification_event = {
                    "question": serialized["question"],
                    "answer": serialized["answer"],
                    "matched_cells": serialized["matched_cells"],
                    "computed_deltas": serialized["computed_deltas"],
                    "verification": serialized["verification"],
                    "citations": serialized["citations"],
                    "guardrail_flags": serialized["guardrail_flags"],
                    "cache_hit": serialized["cache_hit"],
                    "cache_similarity": serialized["cache_similarity"],
                    "latency_ms": serialized["latency_ms"],
                    "metadata": serialized["metadata"],
                }
                yield (
                    "event: verification\ndata: "
                    + json.dumps(verification_event)
                    + "\n\n"
                )
                yield "event: done\ndata: [DONE]\n\n"
                return
            yield f"event: token\ndata: {json.dumps({'text': item})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/metrics", dependencies=[Depends(require_read_access)])
async def service_metrics() -> Dict[str, Any]:
    """Exposes operational telemetry for monitoring dashboards."""
    provider_status = llm_client.get_provider_status()
    return {
        "indexed_chunks": len(retriever.chunks_store),
        "indexed_documents": len(retriever.list_documents()),
        "semantic_cache": semantic_cache.stats(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "llm_model": provider_status["model_name"],
        "llm_base_url": provider_status["base_url"],
        "llm_api_key_configured": provider_status["api_key_configured"],
        "llm_remote_enabled": provider_status["enabled"],
        "llm_last_provider": provider_status["last_provider"],
        "llm_last_error": provider_status["last_error"],
        "retriever": retriever.health(),
    }
