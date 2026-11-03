"""Pin headcount ("vague attribute") questions end to end.

"How many employees does AMD have?" used to be answered from a compensation or stock-plan page:
those pages repeat the attribute word, retrieval ranked them first, and narrative synthesis scored
lines by term overlap alone. The fix has four parts, each pinned here: a people-count rule, an
attribute scan over the document's own chunks, count-aware line scoring, and a query normalizer that
stops rewriting "count" into "court".
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from src.domain.attributes import (
    asks_for_people_count,
    is_count_question,
    person_count_value,
)
from src.domain.documents import DocumentChunk, RetrievedDocument
from src.retrieval.hybrid_retriever import HybridRetriever
from src.usecases.query_normalization import correct_query_terms
from src.usecases.synthesis import LocalSynthesizer, wants_table_evidence

HEADCOUNT_LINE = (
    "At December 31, 2022, the Company employed approximately 92,000 people "
    "(full-time equivalents), with approximately 37,000 employed in the United States."
)
STOCK_PLAN_LINE = (
    "The Company also sponsors employee savings plans for employees under Section 401(k); "
    "eligible employees receive an employer contribution of 3 percent of compensation."
)


@pytest.mark.parametrize(
    "line",
    [
        "The Company employed 93,516 people (full-time equivalents) in 2018.",
        "we employed 29,239 people, of which 51% were in the United States",
        "we had approximately 25,000 employees in our global workforce.",
        "we had approximately 9,200 total full-time and part-time employees.",
    ],
)
def test_person_count_value_reads_the_stated_headcount(line: str) -> None:
    assert person_count_value(line) is not None


@pytest.mark.parametrize(
    "line",
    [
        "Gloria Chen 58 Chief People Officer and Executive Vice President, Employee Experience",
        "Health savings accounts increase 3 percent per year for employees who retired early.",
        "Employees hired on or after January 1, 2009, receive a cash match of 100%.",
    ],
)
def test_person_count_value_ignores_ages_plan_rates_and_plan_years(line: str) -> None:
    assert person_count_value(line) is None


def test_count_questions_are_recognised_but_financial_counts_stay_table_questions() -> None:
    assert is_count_question("How many employees does AMD have?")
    assert asks_for_people_count("What is 3M's employee count in FY2022?")
    assert asks_for_people_count("How many people work at Adobe?")

    # A bare filing year must not turn a headcount question into a table question.
    assert wants_table_evidence("What is 3M's employee count in FY2022?") is False
    # A count that names a statement metric still belongs to table synthesis.
    assert wants_table_evidence("How many shares were repurchased in 2022?") is True
    assert asks_for_people_count("What were 3M's net sales in 2018?") is False


def test_narrative_synthesis_prefers_the_sentence_that_states_the_count() -> None:
    """The stock-plan page repeats "employees" more often; the count sentence must still win."""
    stock_plan = RetrievedDocument(
        chunk_id="plan",
        doc_name="3M_2022_10K",
        page_number=78,
        text=STOCK_PLAN_LINE + " " + STOCK_PLAN_LINE,
        contains_table=False,
        dense_score=0.9,
        bm25_score=0.9,
        rrf_score=0.9,
        rerank_score=0.9,
    )
    human_capital = RetrievedDocument(
        chunk_id="human-capital",
        doc_name="3M_2022_10K",
        page_number=6,
        text=HEADCOUNT_LINE,
        contains_table=False,
        dense_score=0.1,
        bm25_score=0.1,
        rrf_score=0.1,
        rerank_score=0.1,
    )

    answer = LocalSynthesizer().answer(
        "What is 3M's employee count in FY2022?", [stock_plan, human_capital], ["3M_2022_10K"]
    )

    assert "Page 6" in answer
    assert "92,000" in answer
    assert "Page 78" not in answer


def test_narrative_synthesis_reads_a_sentence_behind_the_document_header() -> None:
    """The "[DOCUMENT: ...]" header must not swallow the first sentence of a chunk."""
    chunk = RetrievedDocument(
        chunk_id="header",
        doc_name="ADOBE_2022_10K",
        page_number=16,
        text=f"[DOCUMENT: ADOBE_2022_10K | PAGE: 16] {HEADCOUNT_LINE}",
        contains_table=False,
        dense_score=0.5,
        bm25_score=0.5,
        rrf_score=0.5,
        rerank_score=0.5,
    )

    assert "92,000" in LocalSynthesizer().answer("How many employees?", [chunk])


def test_attribute_scan_finds_the_count_page_of_that_document() -> None:
    client = QdrantClient(":memory:")
    retriever = HybridRetriever(
        collection_name="attribute_test",
        in_memory=True,
        use_lightweight_embedder=True,
        client=client,
    )
    try:
        retriever.index_chunks(
            [
                DocumentChunk(
                    chunk_id="amd-plan",
                    doc_name="AMD_2022_10K",
                    page_number=15,
                    text=STOCK_PLAN_LINE,
                    contains_table=False,
                    metadata={"chunk_index": 0},
                ),
                DocumentChunk(
                    chunk_id="amd-count",
                    doc_name="AMD_2022_10K",
                    page_number=14,
                    text="As of December 31, 2022, we had approximately 25,000 employees.",
                    contains_table=False,
                    metadata={"chunk_index": 1},
                ),
                DocumentChunk(
                    chunk_id="amd-sales",
                    doc_name="AMD_2022_10K",
                    page_number=48,
                    text="| Total net revenue | 23,601 |",
                    contains_table=True,
                    metadata={"chunk_index": 2},
                ),
            ],
            use_embedding_cache=False,
        )

        assert [
            chunk.chunk_id
            for chunk in retriever.attribute_candidates("AMD_2022_10K", {"employees"})
        ] == ["amd-count"]
        assert retriever.attribute_candidates("OTHER_DOC", {"employees"}) == []
    finally:
        client.close()


def test_the_normalizer_keeps_a_corpus_fragment_instead_of_correcting_it_to_an_unrelated_word() -> None:
    """The corpus builds "account" from the same letters, so "count" is a word, not a typo."""
    vocabulary = {"account": 400, "accounts": 120, "court": 245, "county": 44, "revenue": 40}

    assert (
        correct_query_terms("What is 3M's employee count?", vocabulary)
        == "What is 3M's employee count?"
    )
    # Genuine typos are still corrected.
    assert correct_query_terms("what was the reveune?", vocabulary) == "what was the revenue?"
