"""Content-addressed cache for corpus embeddings.

Sentence-transformer inference is CPU-bound and takes minutes for a real filing
corpus. The cache stores one vector per chunk ID and is only valid for the exact
model and chunk dimensions that produced it. Lookups are per chunk, so a cache
built from the checked-in filings still serves every corpus chunk after a user
upload joins the index; only the new chunks are embedded.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np

from src.config import settings
from src.ingestion.pdf_parser import DocumentChunk

logger = logging.getLogger(__name__)


def _read_cache(
    model_name: str,
    cache_path: Path | None,
) -> tuple[List[str], np.ndarray] | None:
    """The cached chunk ids and vectors, or ``None`` when the file is unusable."""
    path = cache_path or settings.EMBEDDING_CACHE_PATH
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            version = int(payload["version"][0])
            cached_model = str(payload["model"][0])
            chunk_ids = [str(value) for value in payload["chunk_ids"].tolist()]
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Ignoring unreadable embedding cache %s: %s", path, exc)
        return None

    if version != settings.EMBEDDING_CACHE_VERSION:
        return None
    if cached_model != model_name:
        return None
    if len(chunk_ids) != len(vectors):
        return None
    if vectors.ndim != 2 or vectors.shape[1] != settings.EMBEDDING_DIM:
        return None
    return chunk_ids, vectors


def load_vector_map(
    chunk_ids: Iterable[str],
    model_name: str = settings.EMBEDDING_MODEL,
    cache_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Cached vectors for whichever requested chunks the cache holds."""
    payload = _read_cache(model_name, cache_path)
    if payload is None:
        return {}
    cached_ids, vectors = payload
    index_by_id = {chunk_id: index for index, chunk_id in enumerate(cached_ids)}
    return {
        chunk_id: vectors[index_by_id[chunk_id]]
        for chunk_id in chunk_ids
        if chunk_id in index_by_id
    }


def load_embeddings(
    chunks: List[DocumentChunk],
    model_name: str = settings.EMBEDDING_MODEL,
    cache_path: Path | None = None,
) -> np.ndarray | None:
    """Return cached vectors in chunk order, or ``None`` when any chunk is missing.

    Extra entries left by a user upload do not invalidate the cache: the corpus chunks still
    resolve, which is what keeps a restart after an upload from re-embedding the filings.
    """
    cached = load_vector_map(
        (chunk.chunk_id for chunk in chunks), model_name, cache_path
    )
    if any(chunk.chunk_id not in cached for chunk in chunks):
        return None
    return np.stack([cached[chunk.chunk_id] for chunk in chunks])


def save_embeddings(
    chunks: List[DocumentChunk],
    vectors: np.ndarray,
    model_name: str = settings.EMBEDDING_MODEL,
    cache_path: Path | None = None,
) -> None:
    """Persist vectors only when their shape matches the chunk list exactly."""
    path = cache_path or settings.EMBEDDING_CACHE_PATH
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(chunks), settings.EMBEDDING_DIM):
        raise ValueError(
            f"embedding cache shape {matrix.shape} does not match "
            f"{len(chunks)} chunks x {settings.EMBEDDING_DIM}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temp_path, "wb") as handle:
            np.savez_compressed(
                handle,
                version=np.asarray([settings.EMBEDDING_CACHE_VERSION], dtype=np.int64),
                model=np.asarray([model_name]),
                chunk_ids=np.asarray([chunk.chunk_id for chunk in chunks]),
                vectors=matrix,
            )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    logger.info("Wrote embedding cache %s (%d vectors)", path, len(chunks))
