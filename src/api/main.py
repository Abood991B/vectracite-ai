from contextlib import asynccontextmanager
import asyncio
import base64
import binascii
from collections import OrderedDict
import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, TypeVar
import pymupdf as fitz
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from src.api.answer_service import AnswerService
from src.api.distributed_lock import distributed_mutation_lock
from src.api.index_state import bump_index_generation, read_index_generation
from src.api.middleware import BodySizeLimitMiddleware
from src.api.security import require_admin_access, require_read_access
from src.api.ui import WORKBENCH_HTML
from evaluation.run_ablation_benchmark import (
    build_dynamic_eval_queries,
    compute_evaluation_signature,
    load_corpus_chunks_from_pdfs,
    run_evaluation_suite,
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
from src.retrieval.table_ast import TableASTParser

parser = MultimodalFinancialParser(
    chunk_size=settings.PARSER_CHUNK_SIZE,
    chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
    ocr_backend=build_ocr_backend(),
)
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
_DEFAULTS_CLEARED = False
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
_PAGE_IMAGE_CACHE: "OrderedDict[str, tuple[str, bytes]]" = OrderedDict()
_PAGE_IMAGE_CACHE_LIMIT = 128


def _cors_origins() -> list[str]:
    """Return the explicit browser origins permitted to call the API."""
    return [origin.strip() for origin in settings.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]


async def _run_ingestion_step(operation: Callable[..., T], *args: Any) -> T:
    """Bound CPU-heavy PDF work without blocking the event loop indefinitely."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(operation, *args), timeout=settings.INGEST_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail="Document processing timed out.") from exc


def _validate_structured_pages(pages: list[Any]) -> None:
    if len(pages) > settings.MAX_STRUCTURED_PAGES:
        raise HTTPException(status_code=413, detail="Document page limit exceeded.")
    if sum(len(page.text) for page in pages) > settings.MAX_STRUCTURED_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="Document text limit exceeded.")


def _render_and_parse_document(
    clean_name: str, raw_pages: List[Dict[str, Any]]
) -> tuple[bytes, List[Any]]:
    """Render once, then parse the exact bytes that will be persisted."""
    pdf_bytes = render_pages_to_pdf_bytes(clean_name, raw_pages)
    return pdf_bytes, parser.parse_pdf_bytes(pdf_bytes, clean_name)


def _validate_pdf_limits(raw_bytes: bytes) -> None:
    if len(raw_bytes) > settings.MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF size limit exceeded.")
    try:
        with fitz.open(stream=raw_bytes, filetype="pdf") as document:
            if document.page_count > settings.MAX_PDF_PAGES:
                raise HTTPException(status_code=413, detail="PDF page limit exceeded.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded content is not a valid PDF.") from exc


def _load_persisted_chunks(include_defaults: bool | None = None) -> List[Any]:
    """Load defaults plus user PDFs, letting a user copy override a default by name."""
    load_defaults = (
        (not _DEFAULTS_CLEARED) if include_defaults is None else include_defaults
    )
    baseline_chunks = load_corpus_chunks_from_pdfs(parser) if load_defaults else []
    upload_paths = sorted(settings.USER_UPLOADS_DIR.glob("*.pdf"))
    uploaded_names = {path.stem for path in upload_paths}
    chunks = [chunk for chunk in baseline_chunks if chunk.doc_name not in uploaded_names]
    for pdf_path in upload_paths:
        chunks.extend(parser.parse_pdf(pdf_path))
    return chunks


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
    """Normalize a document identifier before using it as a corpus filename."""
    clean_name = doc_name.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    if clean_name.lower().endswith(".pdf"):
        clean_name = clean_name[:-4]
    if not clean_name or clean_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Document name cannot be empty.")
    if any(character in '<>:"/\\|?*' for character in clean_name):
        raise HTTPException(status_code=400, detail="Document name contains invalid filename characters.")
    return clean_name


def _normalize_scope(doc_name: str | None) -> str | None:
    """Normalize the optional query scope while treating ALL as global search."""
    if not doc_name or not doc_name.strip() or doc_name.strip().upper() == "ALL":
        return None
    return _clean_doc_name(doc_name)


def _infer_scope_from_question(question: str) -> str | None:
    """Honor an explicit document reference only when it is unambiguous."""
    question_folded = question.casefold()
    question_compact = re.sub(r"[^a-z0-9]", "", question_folded)
    indexed_names = sorted(
        {chunk.doc_name for chunk in retriever.chunks_store},
        key=len,
        reverse=True,
    )
    if len(indexed_names) <= 1:
        return indexed_names[0] if indexed_names else None

    exact_matches = [
        doc_name
        for doc_name in indexed_names
        if re.search(rf"\b{re.escape(doc_name.casefold())}\b", question_folded)
    ]
    if exact_matches:
        return exact_matches[0] if len(exact_matches) == 1 else None

    matches: List[str] = []
    for doc_name in indexed_names:
        aliases = {
            doc_name.split("_", maxsplit=1)[0].casefold(),
        }
        compact_aliases = {
            re.sub(r"[^a-z0-9]", "", alias) for alias in aliases if len(alias) >= 2
        }
        if any(
            len(alias) >= 2
            and re.search(rf"\b{re.escape(alias)}\b", question_folded)
            for alias in aliases
        ) or any(alias and alias in question_compact for alias in compact_aliases):
            matches.append(doc_name)

    if len(matches) > 1:
        years = {
            year if len(year) == 4 else f"20{year}"
            for year in re.findall(r"\b(?:fy\s?)?((?:19|20)\d{2}|\d{2})\b", question, re.I)
        }
        year_matches = [
            doc_name
            for doc_name in matches
            if any(year in doc_name for year in years)
        ]
        if len(year_matches) == 1:
            matches = year_matches

    if len(matches) != 1:
        return None
    if re.search(
        r"\b(compare|comparison|between|across|versus|vs\.?|both|each)\b",
        question,
        flags=re.IGNORECASE,
    ):
        return None
    return matches[0]


def _persist_pdf(doc_name: str, pdf_bytes: bytes) -> Path:
    """Atomically persist a user PDF separately from the checked-in default corpus."""
    settings.USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    target = settings.USER_UPLOADS_DIR / f"{doc_name}.pdf"
    temp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_bytes(pdf_bytes)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return target


def _remove_user_pdf(doc_name: str) -> None:
    """Remove a user PDF and any legacy custom copy from the corpus directory."""
    user_path = settings.USER_UPLOADS_DIR / f"{doc_name}.pdf"
    if user_path.exists():
        user_path.unlink()

    legacy_path = settings.CORPUS_PDF_DIR / f"{doc_name}.pdf"
    if doc_name not in demo_document_names() and legacy_path.exists():
        legacy_path.unlink()


def _find_pdf_path(doc_name: str) -> Path | None:
    """Find a stored user upload first, then a checked-in demo PDF."""
    clean_name = _clean_doc_name(doc_name)
    for root in (settings.USER_UPLOADS_DIR, settings.CORPUS_PDF_DIR):
        pdf_path = root / f"{clean_name}.pdf"
        if pdf_path.exists():
            return pdf_path
    return None


def _ensure_indexed() -> None:
    """Load or hydrate a usable catalog for clients that do not trigger lifespan."""
    if retriever.chunks_store or _DEFAULTS_CLEARED:
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
    definitions = _benchmark_definitions()
    indexed_docs = {document["doc_name"] for document in retriever.list_documents()}
    scenarios: List[Dict[str, Any]] = []
    covered_docs: set[str] = set()

    for definition in definitions:
        doc_name = str(definition["doc_name"])
        if doc_name not in indexed_docs:
            continue
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
        if doc_name not in indexed_docs or doc_name in covered_docs:
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


def _commit_document(clean_name: str, chunks: List[Any], pdf_bytes: bytes) -> int:
    """Persist bytes and swap the indexed version, restoring both on failure."""
    old_user_path = settings.USER_UPLOADS_DIR / f"{clean_name}.pdf"
    old_user_bytes = old_user_path.read_bytes() if old_user_path.exists() else None
    snapshot = retriever.snapshot_document(clean_name)
    try:
        _persist_pdf(clean_name, pdf_bytes)
        removed = retriever.replace_document(clean_name, chunks)
    except Exception:
        try:
            if old_user_bytes is not None:
                _persist_pdf(clean_name, old_user_bytes)
            elif old_user_path.exists():
                old_user_path.unlink(missing_ok=True)
            if snapshot:
                retriever.restore_document(clean_name, snapshot)
        except Exception:
            logger.exception("Unable to roll back failed document commit")
        raise
    semantic_cache.clear()
    bump_index_generation()
    return removed


answer_service = AnswerService(
    retriever=retriever,
    reranker=reranker,
    semantic_cache=semantic_cache,
    guardrails=guardrails,
    numeric_verifier=numeric_verifier,
    llm_client=llm_client,
    ensure_indexed=_ensure_indexed,
    normalize_scope=_normalize_scope,
    infer_scope=_infer_scope_from_question,
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
    """Readiness probe that fails when the index or required dependencies are unusable."""
    _ensure_indexed()
    health = retriever.health()
    semantic_cache.check_health()
    cache = semantic_cache.stats()
    failures: List[str] = []
    if health["degraded"]:
        failures.append("vector_backend_degraded")
    if not retriever.chunks_store:
        failures.append("index_empty")
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
    pdf_path = _find_pdf_path(clean_name)
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
    pdf_path = _find_pdf_path(clean_name)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail=f"Document '{clean_name}' PDF not found.")

    stat = pdf_path.stat()
    cache_key = (
        f"{clean_name}:{page_number}:{highlight.strip().casefold()}:"
        f"{stat.st_mtime_ns}:{stat.st_size}"
    )
    cached = _PAGE_IMAGE_CACHE.get(cache_key)
    if cached is not None:
        etag, png_bytes = cached
        if request.headers.get("if-none-match") == etag:
            return Response(
                status_code=304,
                headers={"ETag": etag, "Cache-Control": "private, max-age=3600"},
            )
        _PAGE_IMAGE_CACHE.move_to_end(cache_key)
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"ETag": etag, "Cache-Control": "private, max-age=3600"},
        )

    document = fitz.open(pdf_path)
    try:
        generated = bool(
            len(document)
            and re.search(
                r"SOURCE PAGE:\s*\d+",
                document[0].get_text("text", sort=True),
                flags=re.IGNORECASE,
            )
        )
        if generated:
            selected_page = None
            for page in document:
                source_text = page.get_text("text", sort=True)
                source_match = re.search(
                    r"SOURCE PAGE:\s*(\d+)", source_text, flags=re.IGNORECASE
                )
                if source_match and int(source_match.group(1)) == page_number:
                    selected_page = page
                    break
            if selected_page is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Page {page_number} is not present in '{clean_name}'.",
                )
        else:
            if page_number > len(document):
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Page {page_number} is outside '{clean_name}' "
                        f"({len(document)} pages)."
                    ),
                )
            selected_page = document[page_number - 1]

        stopwords = {
            "about",
            "according",
            "and",
            "compared",
            "from",
            "into",
            "that",
            "the",
            "their",
            "what",
            "were",
            "with",
        }
        terms = []
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9%$.-]{2,}", highlight):
            normalized = term.casefold()
            if normalized not in stopwords and normalized not in terms:
                terms.append(normalized)
        highlight_count = 0
        for term in terms:
            for rect in selected_page.search_for(term):
                selected_page.add_highlight_annot(rect).update()
                highlight_count += 1
                if highlight_count >= 8:
                    break
            if highlight_count >= 8:
                break

        png_bytes = selected_page.get_pixmap(dpi=150, alpha=False).tobytes("png")
        etag = '"' + hashlib.sha256(png_bytes).hexdigest()[:32] + '"'
        _PAGE_IMAGE_CACHE[cache_key] = (etag, png_bytes)
        _PAGE_IMAGE_CACHE.move_to_end(cache_key)
        while len(_PAGE_IMAGE_CACHE) > _PAGE_IMAGE_CACHE_LIMIT:
            _PAGE_IMAGE_CACHE.popitem(last=False)
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"ETag": etag, "Cache-Control": "private, max-age=3600"},
        )
    finally:
        document.close()


@app.post("/documents/clear-demos", dependencies=[Depends(require_admin_access)])
async def clear_demo_documents() -> Dict[str, Any]:
    """Remove only demo documents from the active index and preserve user uploads."""
    global _DEFAULTS_CLEARED
    async with _mutation_lock:
        async with distributed_mutation_lock():
            user_document_names = {
                path.stem for path in settings.USER_UPLOADS_DIR.glob("*.pdf")
            }
            _DEFAULTS_CLEARED = True
            removed_chunks = 0
            for doc_name in demo_document_names() - user_document_names:
                removed_chunks += retriever.delete_document(doc_name)
            semantic_cache.clear()
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
    global _DEFAULTS_CLEARED
    async with _mutation_lock:
        async with distributed_mutation_lock():
            upload_paths = sorted(settings.USER_UPLOADS_DIR.glob("*.pdf"))
            _DEFAULTS_CLEARED = True
            retriever.clear()
            semantic_cache.clear()
            removed_files = 0
            for path in upload_paths:
                try:
                    path.unlink(missing_ok=True)
                    removed_files += 1
                except OSError:
                    logger.exception("Unable to remove uploaded PDF %s", path.name)
            await asyncio.to_thread(bump_index_generation)
    return {
        "status": "cleared_all",
        "documents": len(retriever.list_documents()),
        "uploads_removed": removed_files,
    }


@app.post("/documents/restore-defaults", dependencies=[Depends(require_admin_access)])
async def restore_default_documents() -> Dict[str, Any]:
    """Restore demos while preserving and re-indexing every user upload."""
    global _DEFAULTS_CLEARED
    async with _mutation_lock:
        async with distributed_mutation_lock():
            persisted_chunks = _load_persisted_chunks(include_defaults=True)
            _DEFAULTS_CLEARED = False
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
    maximum_encoded_size = ((settings.MAX_PDF_BYTES + 2) // 3) * 4
    if len(payload.pdf_base64) > maximum_encoded_size:
        raise HTTPException(status_code=413, detail="PDF size limit exceeded.")
    try:
        raw_bytes = base64.b64decode(payload.pdf_base64, validate=True)
        _validate_pdf_limits(raw_bytes)
        clean_name = _clean_doc_name(payload.doc_name)
        chunks = await _run_ingestion_step(
            parser.parse_pdf_bytes, raw_bytes, clean_name
        )
        if not chunks:
            raise ValueError("No readable text or tables found in PDF.")
        async with _mutation_lock:
            async with distributed_mutation_lock():
                await asyncio.to_thread(
                    _commit_document, clean_name, chunks, raw_bytes
                )
        return {
            "status": "indexed_pdf",
            "doc_name": clean_name,
            "chunks_created": len(chunks),
            "table_chunks": sum(1 for c in chunks if c.contains_table),
        }
    except HTTPException:
        raise
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc


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
        async with _mutation_lock:
            async with distributed_mutation_lock():
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
        async with _mutation_lock:
            async with distributed_mutation_lock():
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
    async with _mutation_lock:
        async with distributed_mutation_lock():
            removed = retriever.delete_document(clean_name)
            if removed == 0:
                raise HTTPException(
                    status_code=404, detail=f"Document '{clean_name}' not found."
                )
            _remove_user_pdf(clean_name)
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
        return run_evaluation_suite(retriever.chunks_store)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    current_signature = compute_evaluation_signature(retriever.chunks_store)
    if report.get("evaluation_signature") != current_signature:
        return run_evaluation_suite(retriever.chunks_store)
    return report


@app.post("/evaluate", dependencies=[Depends(require_admin_access)])
async def evaluate_current_index() -> Dict[str, Any]:
    """Run a live ablation across all currently indexed documents."""
    _ensure_indexed()
    return run_evaluation_suite(retriever.chunks_store)


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
