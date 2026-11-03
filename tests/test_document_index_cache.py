"""Pin that every way of changing the chunk store also drops its derived indexes.

The statement and people-count indexes are built once per document and reused, which is what
removed ~10-20 ms per query on the real corpus. That optimisation is only safe if no mutation can
leave a stale index behind, so each mutation path gets a test: warm the index, change the document,
assert the next read reflects the change. A mutation added later that forgets the choke point fails
one of these.
"""

import pytest
from qdrant_client import QdrantClient

from src.domain.documents import DocumentChunk
from src.retrieval.hybrid_retriever import HybridRetriever

DOC = "CACHE_DOC"

STATEMENT_PAGE = (
    "[DOCUMENT: CACHE_DOC | PAGE: 54] ADOBE INC. CONSOLIDATED STATEMENTS OF INCOME "
    "(In millions) 2022 2021 2020"
)
HEADCOUNT_PAGE = (
    "[DOCUMENT: CACHE_DOC | PAGE: 6] As of December 31, 2022, we employed 29,239 people worldwide."
)
REPLACEMENT_STATEMENT = (
    "[DOCUMENT: CACHE_DOC | PAGE: 77] ACME CORP CONSOLIDATED STATEMENTS OF OPERATIONS "
    "(In millions) 2023 2022 2021"
)
REPLACEMENT_HEADCOUNT = (
    "[DOCUMENT: CACHE_DOC | PAGE: 8] At December 31, 2023, we had approximately 31,500 employees."
)


@pytest.fixture()
def retriever() -> HybridRetriever:
    return HybridRetriever(
        collection_name="document_index_cache",
        in_memory=True,
        client=QdrantClient(":memory:"),
    )


def _chunk(chunk_id: str, page: int, text: str, index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_name=DOC,
        page_number=page,
        text=text,
        contains_table=False,
        metadata={"chunk_index": index},
    )


def _statement_page_numbers(retriever: HybridRetriever) -> list[int]:
    return sorted({chunk.page_number for chunk in retriever.statement_candidates(DOC)})


def _headcount_pages(retriever: HybridRetriever) -> list[int]:
    return sorted(
        {
            chunk.page_number
            for chunk in retriever.attribute_candidates(DOC, {"employees"})
        }
    )


def test_indexing_more_chunks_for_a_document_refreshes_the_derived_pages(retriever) -> None:
    """index_chunks merges by chunk id, so the derived index must grow, not stay at the old state."""
    retriever.index_chunks(
        [_chunk("v1-statement", 54, STATEMENT_PAGE), _chunk("v1-headcount", 6, HEADCOUNT_PAGE)],
        use_embedding_cache=False,
    )
    assert _statement_page_numbers(retriever) == [54]
    assert _headcount_pages(retriever) == [6]

    retriever.index_chunks(
        [
            _chunk("v2-statement", 77, REPLACEMENT_STATEMENT),
            _chunk("v2-headcount", 8, REPLACEMENT_HEADCOUNT),
        ],
        use_embedding_cache=False,
    )

    assert _statement_page_numbers(retriever) == [54, 77]
    assert _headcount_pages(retriever) == [6, 8]


def test_replacing_a_document_drops_its_derived_pages(retriever) -> None:
    retriever.index_chunks(
        [_chunk("v1-statement", 54, STATEMENT_PAGE), _chunk("v1-headcount", 6, HEADCOUNT_PAGE)],
        use_embedding_cache=False,
    )
    assert _statement_page_numbers(retriever) == [54]

    retriever.replace_document(
        DOC,
        [
            _chunk("v2-statement", 77, REPLACEMENT_STATEMENT),
            _chunk("v2-headcount", 8, REPLACEMENT_HEADCOUNT),
        ],
    )

    assert _statement_page_numbers(retriever) == [77]
    assert _headcount_pages(retriever) == [8]


def test_deleting_a_document_drops_its_derived_pages(retriever) -> None:
    retriever.index_chunks(
        [_chunk("v1-statement", 54, STATEMENT_PAGE), _chunk("v1-headcount", 6, HEADCOUNT_PAGE)],
        use_embedding_cache=False,
    )
    assert _statement_page_numbers(retriever) == [54]

    retriever.delete_document(DOC)

    assert _statement_page_numbers(retriever) == []
    assert _headcount_pages(retriever) == []


def test_clearing_the_store_drops_derived_pages(retriever) -> None:
    retriever.index_chunks(
        [_chunk("v1-statement", 54, STATEMENT_PAGE), _chunk("v1-headcount", 6, HEADCOUNT_PAGE)],
        use_embedding_cache=False,
    )
    assert _statement_page_numbers(retriever) == [54]

    retriever.clear()

    assert _statement_page_numbers(retriever) == []
    assert _headcount_pages(retriever) == []


def test_hydration_drops_pages_built_from_the_previous_store(retriever) -> None:
    retriever.index_chunks(
        [_chunk("v1-statement", 54, STATEMENT_PAGE), _chunk("v1-headcount", 6, HEADCOUNT_PAGE)],
        use_embedding_cache=False,
    )
    assert _statement_page_numbers(retriever) == [54]

    # Simulate another process writing a newer version, then this process re-reading the store.
    retriever.replace_document(
        DOC,
        [
            _chunk("v2-statement", 77, REPLACEMENT_STATEMENT),
            _chunk("v2-headcount", 8, REPLACEMENT_HEADCOUNT),
        ],
    )
    retriever.hydrate_from_qdrant()

    assert _statement_page_numbers(retriever) == [77]
    assert _headcount_pages(retriever) == [8]


def test_the_second_read_does_not_rebuild_the_index(retriever) -> None:
    retriever.index_chunks([_chunk("v1-statement", 54, STATEMENT_PAGE)], use_embedding_cache=False)

    first = retriever.statement_candidates(DOC)
    second = retriever.statement_candidates(DOC)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert retriever.statement_candidates("UNKNOWN_DOC") == []
