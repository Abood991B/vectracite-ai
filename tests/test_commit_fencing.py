"""Pin the commit fence and the duplicate-version reconcile.

A replica that loses its Redis lease can resume after a newer replica has written. The fence
makes its destructive steps conditional on a shared commit epoch, and the reconcile makes
hydration converge on the newest epoch. Together they replace "best effort" with "a stale writer
cannot destroy a newer version, and a restart leaves one version per document".
"""

import logging

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.domain.documents import DocumentChunk
from src.retrieval.hybrid_retriever import HybridRetriever

DOC = "FENCING_DOC"


@pytest.fixture
def retriever() -> HybridRetriever:
    client = QdrantClient(":memory:")
    instance = HybridRetriever(
        collection_name="fencing_test",
        in_memory=True,
        use_lightweight_embedder=True,
        client=client,
    )
    yield instance
    client.close()


def _chunks(marker: str, count: int = 2) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"{DOC}-{marker}-{index}",
            doc_name=DOC,
            page_number=index + 1,
            text=f"{marker} segment {index} reports revenue of {1000 + index}.",
            contains_table=False,
            metadata={"chunk_index": index},
        )
        for index in range(count)
    ]


def _write_version(
    retriever: HybridRetriever,
    marker: str,
    *,
    version: str,
    epoch: int | None,
) -> None:
    """Write one version without the stale delete, like a crash between the two steps."""
    chunks = _chunks(marker)
    vectors = retriever._encode_texts([chunk.text for chunk in chunks])
    points = retriever._build_points(
        chunks, vectors, version=version, epoch=epoch
    )
    retriever._upsert_points(points)


def _payloads(retriever: HybridRetriever) -> list[dict]:
    records, _ = retriever.qdrant.scroll(
        collection_name=retriever.collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_name", match=models.MatchValue(value=DOC)
                )
            ]
        ),
        limit=1000,
        with_payload=True,
    )
    return [record.payload or {} for record in records]


def test_stale_replace_cannot_delete_a_newer_commit(retriever) -> None:
    retriever.replace_document(DOC, _chunks("NEW"), epoch=6)
    retriever.replace_document(DOC, _chunks("OLD"), epoch=5)

    payloads = _payloads(retriever)
    assert any("NEW" in payload["text"] for payload in payloads), (
        "the stale writer deleted the newer commit"
    )
    assert len({payload["version"] for payload in payloads}) == 2

    retriever.hydrate_from_qdrant()
    assert retriever.chunks_store
    assert all("NEW" in chunk.text for chunk in retriever.chunks_store)
    assert all("NEW" in payload["text"] for payload in _payloads(retriever))


def test_stale_delete_cannot_remove_a_newer_commit(retriever) -> None:
    retriever.replace_document(DOC, _chunks("NEW"), epoch=6)

    assert retriever.delete_document(DOC, epoch=5) == 0
    assert retriever.has_document(DOC)
    assert retriever.get_document_chunks(DOC)
    assert all("NEW" in payload["text"] for payload in _payloads(retriever))


def test_a_current_delete_removes_the_document(retriever) -> None:
    retriever.replace_document(DOC, _chunks("NEW"), epoch=6)

    assert retriever.delete_document(DOC, epoch=6) == 2
    assert not retriever.has_document(DOC)
    assert _payloads(retriever) == []


def test_hydration_repairs_a_crash_between_upsert_and_stale_delete(retriever) -> None:
    _write_version(retriever, "OLD", version="v-old", epoch=5)
    _write_version(retriever, "NEW", version="v-new", epoch=6)
    assert len({payload["version"] for payload in _payloads(retriever)}) == 2

    assert retriever.hydrate_from_qdrant() == 2
    assert all("NEW" in chunk.text for chunk in retriever.chunks_store)
    payloads = _payloads(retriever)
    assert {payload["version"] for payload in payloads} == {"v-new"}


def test_hydration_prefers_a_versioned_commit_over_bulk_index_points(retriever) -> None:
    retriever.index_chunks(_chunks("BULK"), use_embedding_cache=False)
    _write_version(retriever, "NEW", version="v-new", epoch=6)
    assert len({payload.get("version") for payload in _payloads(retriever)}) == 2

    retriever.hydrate_from_qdrant()
    assert all("NEW" in chunk.text for chunk in retriever.chunks_store)
    assert all("NEW" in payload["text"] for payload in _payloads(retriever))


def test_hydration_is_idempotent_after_a_repair(retriever) -> None:
    _write_version(retriever, "OLD", version="v-old", epoch=5)
    _write_version(retriever, "NEW", version="v-new", epoch=6)

    assert retriever.hydrate_from_qdrant() == 2
    assert retriever.hydrate_from_qdrant() == 2
    assert len(_payloads(retriever)) == 2


def test_hydration_keeps_tied_legacy_versions_and_warns(retriever, caplog) -> None:
    """Pre-upgrade duplicates have no epoch; keep them rather than guess an order."""
    _write_version(retriever, "OLD", version="v-old", epoch=None)
    _write_version(retriever, "NEW", version="v-new", epoch=None)

    with caplog.at_level(logging.WARNING):
        retriever.hydrate_from_qdrant()

    assert len(_payloads(retriever)) == 4
    assert len({payload["version"] for payload in _payloads(retriever)}) == 2
    assert "without an ordering epoch" in caplog.text
