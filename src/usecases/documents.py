"""Document lifecycle use cases: load what is persisted, commit a new version atomically.

Orchestration only — the catalog, the file store, cache invalidation, and index notification are
supplied by the caller, so the policy can run against fakes without a database or a filesystem.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Protocol

from src.domain.documents import DocumentChunk

logger = logging.getLogger(__name__)


class DocumentCatalogPort(Protocol):
    """The indexed catalog operations a commit needs."""

    def snapshot_document(self, doc_name: str) -> List[DocumentChunk]: ...

    def replace_document(self, doc_name: str, chunks: List[DocumentChunk]) -> int: ...

    def restore_document(self, doc_name: str, chunks: List[DocumentChunk]) -> None: ...


class UploadStorePort(Protocol):
    """Persistence operations a commit needs."""

    def read(self, doc_name: str) -> bytes | None: ...

    def write(self, doc_name: str, pdf_bytes: bytes) -> Any: ...

    def delete(self, doc_name: str, *, is_demo: bool) -> None: ...

    def list_upload_paths(self) -> List[Any]: ...

    def defaults_cleared(self) -> bool: ...


def load_persisted_chunks(
    *,
    parser: Any,
    corpus_loader: Callable[[Any], List[DocumentChunk]],
    uploads: UploadStorePort,
    include_defaults: bool | None = None,
) -> List[DocumentChunk]:
    """Load defaults plus user PDFs, letting a user copy override a default by name."""
    load_defaults = (not uploads.defaults_cleared()) if include_defaults is None else include_defaults
    baseline_chunks = corpus_loader(parser) if load_defaults else []
    upload_paths = uploads.list_upload_paths()
    uploaded_names = {path.stem for path in upload_paths}
    chunks = [chunk for chunk in baseline_chunks if chunk.doc_name not in uploaded_names]
    for pdf_path in upload_paths:
        chunks.extend(parser.parse_pdf(pdf_path))
    return chunks


def commit_document(
    *,
    doc_name: str,
    chunks: List[DocumentChunk],
    pdf_bytes: bytes,
    catalog: DocumentCatalogPort,
    uploads: UploadStorePort,
    invalidate_cache: Callable[[], None],
    notify_index_changed: Callable[[], None],
) -> int:
    """Persist bytes and swap the indexed version, restoring both on failure."""
    previous_bytes = uploads.read(doc_name)
    snapshot = catalog.snapshot_document(doc_name)
    try:
        uploads.write(doc_name, pdf_bytes)
        removed = catalog.replace_document(doc_name, chunks)
    except Exception:
        try:
            if previous_bytes is not None:
                uploads.write(doc_name, previous_bytes)
            else:
                uploads.delete(doc_name, is_demo=False)
            if snapshot:
                catalog.restore_document(doc_name, snapshot)
        except Exception:
            logger.exception("Unable to roll back failed document commit")
        raise
    invalidate_cache()
    notify_index_changed()
    return removed
