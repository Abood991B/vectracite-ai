"""Content-addressed cache for parsed corpus chunks.

PDF parsing is the expensive part of startup for real filings. The cache is
keyed by every PDF's SHA256 plus the parser's chunking configuration and parser
cache version, so a changed PDF or parser never serves stale chunks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from src.config import settings
from src.domain.documents import DocumentChunk
from src.ingestion.pdf_parser import (
    PARSER_CACHE_VERSION,
    MultimodalFinancialParser,
)

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parser_signature(parser: MultimodalFinancialParser) -> Dict[str, object]:
    signature = {
        "chunk_size": parser.chunk_size,
        "chunk_overlap": parser.chunk_overlap,
        "parser_cache_version": PARSER_CACHE_VERSION,
        "ocr_engine": getattr(parser.ocr_backend, "name", None),
        "ocr_dpi": parser.ocr_dpi,
        "ocr_min_chars": parser.ocr_min_chars,
    }
    if parser.ocr_backend is not None:
        signature["ocr_max_pages"] = parser.ocr_max_pages
        signature["ocr_completeness_version"] = 1
    return signature


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _load_cache(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable corpus cache %s: %s", path, exc)
        return None


def _fingerprints(pdf_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    fingerprints: Dict[str, Dict[str, Any]] = {}
    for path in pdf_paths:
        data = path.read_bytes()
        fingerprints[path.stem] = {
            "filename": path.name,
            "bytes": len(data),
            "sha256": _sha256(data),
        }
    return fingerprints


def _cache_is_valid(
    cache: Dict[str, Any] | None,
    parser_signature: Dict[str, object],
    fingerprints: Dict[str, Dict[str, Any]],
) -> bool:
    if not cache:
        return False
    if cache.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    if cache.get("parser") != parser_signature:
        return False
    cached_documents = cache.get("documents") or {}
    if set(cached_documents) != set(fingerprints):
        return False
    for doc_name, fingerprint in fingerprints.items():
        cached = cached_documents.get(doc_name) or {}
        if cached.get("sha256") != fingerprint["sha256"]:
            return False
        if cached.get("bytes") != fingerprint["bytes"]:
            return False
        if not isinstance(cached.get("chunks"), list):
            return False
    return True


def _chunks_from_cache(cache: Dict[str, Any]) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    documents = cache.get("documents") or {}
    for doc_name in sorted(documents):
        for item in documents[doc_name].get("chunks") or []:
            if item.get("doc_name") != doc_name:
                raise ValueError(f"corpus cache has a mismatched chunk for {doc_name}")
            chunks.append(
                DocumentChunk(
                    chunk_id=str(item["chunk_id"]),
                    doc_name=str(item["doc_name"]),
                    page_number=int(item["page_number"]),
                    text=str(item["text"]),
                    contains_table=bool(item["contains_table"]),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
    return chunks


def cache_path_for_corpus(corpus_dir: Path) -> Path:
    """Return the cache file that belongs to a corpus directory.

    The default corpus uses the configured index path; any other corpus keeps its cache beside the
    directory itself, so the file can be checked in and reused across runs. This is the single owner
    of that decision — callers must not build the filename themselves.
    """
    if corpus_dir.resolve() == settings.CORPUS_PDF_DIR.resolve():
        return settings.CORPUS_INDEX_PATH
    return corpus_dir.parent / f"{corpus_dir.name}_index.json"


def load_corpus_chunks(
    corpus_dir: Path | None = None,
    parser: MultimodalFinancialParser | None = None,
    *,
    cache_path: Path | None = None,
    force: bool = False,
    write: bool = True,
    skip_errors: bool = False,
) -> List[DocumentChunk]:
    """Return corpus chunks, using a content-addressed cache when it is valid."""
    active_dir = corpus_dir or settings.CORPUS_PDF_DIR
    active_parser = parser or MultimodalFinancialParser(
        chunk_size=settings.PARSER_CHUNK_SIZE,
        chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
    )
    active_cache_path = cache_path or settings.CORPUS_INDEX_PATH
    pdf_paths = sorted(active_dir.glob("*.pdf"))
    if not pdf_paths:
        return []

    fingerprints = _fingerprints(pdf_paths)
    parser_signature = _parser_signature(active_parser)
    started_at = time.perf_counter()

    if not force:
        cache = _load_cache(active_cache_path)
        if _cache_is_valid(cache, parser_signature, fingerprints):
            try:
                chunks = _chunks_from_cache(cache)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Corpus cache is malformed, reparsing: %s", exc)
            else:
                logger.info(
                    "Corpus cache hit: %d PDFs, %d chunks in %.3fs",
                    len(pdf_paths),
                    len(chunks),
                    time.perf_counter() - started_at,
                )
                return chunks

    documents: Dict[str, Any] = {}
    all_chunks: List[DocumentChunk] = []
    for path in pdf_paths:
        document_started_at = time.perf_counter()
        data = path.read_bytes()
        try:
            chunks = active_parser.parse_pdf_bytes(data, doc_name=path.stem)
        except Exception:
            if not skip_errors:
                raise
            logger.exception("Skipping unparseable corpus PDF %s", path.name)
            fingerprints.pop(path.stem, None)
            continue
        documents[path.stem] = {
            "filename": path.name,
            "bytes": len(data),
            "sha256": _sha256(data),
            "chunks": [chunk.to_dict() for chunk in chunks],
        }
        all_chunks.extend(chunks)
        page_count = max((chunk.page_number for chunk in chunks), default=0)
        logger.info(
            "Parsed %s: %d pages, %d chunks in %.2fs",
            path.name,
            page_count,
            len(chunks),
            time.perf_counter() - document_started_at,
        )

    if write:
        _atomic_write(
            active_cache_path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "parser": parser_signature,
                "source_dir": str(active_dir),
                "documents": documents,
            },
        )
        logger.info(
            "Wrote corpus cache %s (%d chunks) in %.2fs",
            active_cache_path,
            len(all_chunks),
            time.perf_counter() - started_at,
        )
    return all_chunks
