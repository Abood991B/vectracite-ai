"""Pin entity recognition to the corpus itself, not to a table of company names in code.

The old implementation held a literal map of five document entities, so recognising a new company
meant editing ``src/``. Entity terms now come from the document name, and the entities a query
names come from the documents that are actually indexed.
"""

from qdrant_client import QdrantClient

from src.domain.documents import DocumentChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.table_ast import (
    TableASTParser,
    document_entity_terms,
    named_entities,
)

AMD_TABLE = (
    "EXTRACTED TABLES (PAGE 54)\n"
    "| (Millions) | Col2 |\n"
    "| --- | --- |\n"
    "| AMD net revenue | 23,601 |"
)
ADOBE_TABLE = (
    "EXTRACTED TABLES (PAGE 65)\n"
    "| (Millions) | Col2 |\n"
    "| --- | --- |\n"
    "| Adobe net revenue | 17,606 |"
)


def test_entity_terms_come_from_the_document_name() -> None:
    assert document_entity_terms("AMD_2022_10K") == {"amd"}
    assert document_entity_terms("3M_2018_10K") == {"3m"}
    assert document_entity_terms("ACTIVISIONBLIZZARD_2019_10K") == {"activisionblizzard"}
    assert document_entity_terms("") == set()


def test_named_entities_only_include_documents_the_corpus_holds() -> None:
    doc_names = ["AMD_2022_10K", "ADOBE_2022_10K"]

    assert named_entities("What was AMD's net revenue?", doc_names) == {"amd"}
    assert named_entities("Compare AMD and Adobe revenue", doc_names) == {"amd", "adobe"}
    assert named_entities("What was Microsoft's revenue?", doc_names) == set()


def test_a_table_that_does_not_name_the_queried_entity_does_not_align() -> None:
    """The query names AMD, so an Adobe table must not score, however many metric words it shares."""
    query = "What was AMD's net revenue in 2022?"
    entities = named_entities(query, ["AMD_2022_10K", "ADOBE_2022_10K"])

    assert TableASTParser.compute_ast_alignment_score(query, ADOBE_TABLE, entities) == 0.0
    assert TableASTParser.compute_ast_alignment_score(query, AMD_TABLE, entities) > 0.0


def test_no_queried_entity_leaves_alignment_unchanged() -> None:
    query = "What was net revenue in 2022?"

    assert TableASTParser.compute_ast_alignment_score(
        query, ADOBE_TABLE, set()
    ) == TableASTParser.compute_ast_alignment_score(query, ADOBE_TABLE, None)


def test_query_entities_are_discovered_from_the_indexed_documents() -> None:
    retriever = HybridRetriever(
        collection_name="entity_terms",
        in_memory=True,
        client=QdrantClient(":memory:"),
    )
    retriever.index_chunks(
        [
            DocumentChunk(
                chunk_id="amd-page",
                doc_name="AMD_2022_10K",
                page_number=1,
                text="AMD results",
                contains_table=False,
                metadata={"chunk_index": 0},
            ),
            DocumentChunk(
                chunk_id="acme-page",
                doc_name="ACME_2024_10K",
                page_number=1,
                text="Acme results",
                contains_table=False,
                metadata={"chunk_index": 0},
            ),
        ],
        use_embedding_cache=False,
    )

    assert retriever._query_entities("What was AMD's revenue?") == {"amd"}
    assert retriever._query_entities("What was Acme's revenue?") == {"acme"}
    assert retriever._query_entities("What was revenue?") == set()
