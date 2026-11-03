"""Pin the comparison/table evidence selection.

Retrieval's reranked top-k can rank narrative segment sales above an income statement, and a
comparison question used to fall back to narrative when the table was outside the top 6. These
tests pin the pieces that fix that: year-aware document targets, the document's own table
candidates, and best-metric table selection.

The labelled set in ``data/retrieval_quality_cases.json`` scores the whole path end to end. The
tests at the bottom pin the four table-selection rules one at a time, because each one alone is
enough to send a comparison answer to a page that does not state the figure: a period column
header counted as a metric, a segment row beating the consolidated total, a plain-text statement
the table parser cannot see, and evidence order deciding a tie.
"""

import pytest
from qdrant_client import QdrantClient

from src.domain.documents import DocumentChunk, RetrievedDocument
from src.retrieval.hybrid_retriever import HybridRetriever
from src.usecases.synthesis import LocalSynthesizer, mentioned_documents

DOC = "COMPARE_DOC"
INCOME_TABLE = (
    "EXTRACTED TABLES (PAGE 14)\n"
    "| (Dollars in millions) | 2018 | 2017 |\n"
    "| --- | --- | --- |\n"
    "| Net sales | 32,765 | 32,009 |\n"
    "| Operating income | 6,000 | 5,500 |"
)
SEGMENT_TABLE = (
    "EXTRACTED TABLES (PAGE 33)\n"
    "| Segment | 2018 |\n"
    "| --- | --- |\n"
    "| Industrial sales | 12,300 |"
)
OPEX_TABLE = (
    "EXTRACTED TABLES (PAGE 43)\n"
    "| Expense | 2022 |\n"
    "| --- | --- |\n"
    "| Sales and marketing | 4,968 |\n"
    "| Research and development | 3,000 |"
)
INCOME_STATEMENT_TABLE = (
    "EXTRACTED TABLES (PAGE 45)\n"
    "| (In millions) | 2022 |\n"
    "| --- | --- |\n"
    "| Total revenue | 17,606 |\n"
    "| Cost of revenue | 2,000 |\n"
    "| Gross profit | 15,606 |\n"
    "| Net income | 4,756 |"
)


@pytest.fixture
def retriever() -> HybridRetriever:
    client = QdrantClient(":memory:")
    instance = HybridRetriever(
        collection_name="comparison_test",
        in_memory=True,
        use_lightweight_embedder=True,
        client=client,
    )
    yield instance
    client.close()


def _evidence(
    page: int,
    text: str,
    *,
    doc_name: str = "3M_2018_10K",
    score: float = 0.5,
    table: bool = True,
) -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id=f"{doc_name}-{page}",
        doc_name=doc_name,
        page_number=page,
        text=text,
        contains_table=table,
        dense_score=score,
        bm25_score=score,
        rrf_score=score,
        rerank_score=score,
    )


def test_mentioned_documents_prefers_the_question_year() -> None:
    docs = ["3M_2018_10K", "3M_2022_10K", "AMD_2022_10K"]

    assert mentioned_documents(
        "Compare net sales for 3M and AMD in 2022.", docs
    ) == ["3M_2022_10K", "AMD_2022_10K"]
    assert mentioned_documents("What were 3M's net sales in 2018?", docs) == [
        "3M_2018_10K"
    ]
    assert mentioned_documents("Compare 3M and AMD net sales.", docs) == docs
    # AMD has no 2018 filing, so it stays in the comparison instead of silently disappearing.
    assert mentioned_documents("Compare 3M and AMD in 2018.", docs) == [
        "3M_2018_10K",
        "AMD_2022_10K",
    ]


def test_table_answer_prefers_the_income_statement_over_a_higher_ranked_segment_table() -> None:
    segments = _evidence(33, SEGMENT_TABLE, score=0.9)
    income = _evidence(14, INCOME_TABLE, score=0.1)

    answer = LocalSynthesizer._table_answer(
        "What were 3M's net sales in 2018?", [segments, income]
    )

    assert answer is not None
    assert "Page 14" in answer
    assert "32,765" in answer
    assert "Page 33" not in answer


def test_table_answer_prefers_statement_rows_over_a_higher_ranked_expense_table() -> None:
    opex = _evidence(43, OPEX_TABLE, doc_name="ADOBE_2022_10K", score=0.9)
    income = _evidence(
        45, INCOME_STATEMENT_TABLE, doc_name="ADOBE_2022_10K", score=0.1
    )

    answer = LocalSynthesizer._table_answer(
        "Compare net sales for Adobe and AMD in 2022.", [opex, income]
    )

    assert answer is not None
    assert "Page 45" in answer
    assert "17,606" in answer
    assert "Page 43" not in answer


def test_a_net_income_table_does_not_satisfy_a_net_sales_question() -> None:
    income = _evidence(
        28,
        "EXTRACTED TABLES (PAGE 28)\n| (Millions) | 2022 |\n| --- | --- |\n| Net income | 2,311 |",
        doc_name="3M_2022_10K",
        score=0.9,
    )
    segments = _evidence(
        25,
        "EXTRACTED TABLES (PAGE 25)\n| Segment | 2022 |\n| --- | --- |\n| Net sales (millions) | 34,229 |",
        doc_name="3M_2022_10K",
        score=0.1,
    )

    answer = LocalSynthesizer._table_answer(
        "What were 3M's net sales in 2022?", [income, segments]
    )

    assert answer is not None
    assert "Page 25" in answer
    assert "Page 28" not in answer


def test_table_candidates_filters_to_matching_tables_of_that_document(retriever) -> None:
    retriever.index_chunks(
        [
            DocumentChunk(
                chunk_id="cmp-net",
                doc_name=DOC,
                page_number=14,
                text=INCOME_TABLE,
                contains_table=True,
                metadata={"chunk_index": 0},
            ),
            DocumentChunk(
                chunk_id="cmp-inventory",
                doc_name=DOC,
                page_number=40,
                text="| Metric | 2018 |\n| --- | --- |\n| Inventory | 100 |",
                contains_table=True,
                metadata={"chunk_index": 1},
            ),
            DocumentChunk(
                chunk_id="cmp-narrative",
                doc_name=DOC,
                page_number=33,
                text="Narrative discussion of net sales.",
                contains_table=False,
                metadata={"chunk_index": 2},
            ),
        ],
        use_embedding_cache=False,
    )

    assert [
        chunk.chunk_id for chunk in retriever.table_candidates(DOC, {"net", "sales"})
    ] == ["cmp-net"]
    assert retriever.table_candidates("OTHER_DOC", {"net", "sales"}) == []


def test_a_period_column_header_does_not_promote_a_segment_table() -> None:
    """AMD 2022: the segment table's "2022" column tied it to a "net sales in 2022" question."""
    segment = _evidence(
        33,
        "EXTRACTED TABLES (PAGE 33)\n"
        "| Segment | 2022 |\n"
        "| --- | --- |\n"
        "| Net sales | 12,300 |",
        doc_name="AMD_2022_10K",
        score=0.9,
    )
    statement = _evidence(
        48,
        "EXTRACTED TABLES (PAGE 48)\n"
        "| (Millions) | Col2 |\n"
        "| --- | --- |\n"
        "| Total net revenue | 23,601 |",
        doc_name="AMD_2022_10K",
        score=0.1,
    )

    answer = LocalSynthesizer._table_answer(
        "Compare net sales for 3M and AMD in 2022.", [segment, statement]
    )

    assert answer is not None
    assert "Page 48" in answer
    assert "23,601" in answer
    assert "Page 33" not in answer


def test_a_consolidated_total_row_beats_a_segment_row_with_the_same_terms() -> None:
    """AMD 2022: "Net revenue" is a note row; "Total net revenue" is the reported figure."""
    note = _evidence(
        73,
        "EXTRACTED TABLES (PAGE 73)\n"
        "| Disaggregation | Col3 |\n"
        "| --- | --- |\n"
        "| Net revenue | 24,117 |",
        doc_name="AMD_2022_10K",
        score=0.9,
    )
    statement = _evidence(
        48,
        "EXTRACTED TABLES (PAGE 48)\n"
        "| (Millions) | Col3 |\n"
        "| --- | --- |\n"
        "| Total net revenue | 23,601 |",
        doc_name="AMD_2022_10K",
        score=0.1,
    )

    answer = LocalSynthesizer._table_answer(
        "Compare net sales for 3M and AMD in 2022.", [note, statement]
    )

    assert answer is not None
    assert "Page 48" in answer
    assert "23,601" in answer
    assert "Page 73" not in answer


def test_a_plain_text_statement_answers_when_the_table_lost_the_row_key() -> None:
    """3M 2022: "Net sales" and 34,229 are parsed into the header row, so no cell row names it."""
    segment = _evidence(
        25,
        "Net sales (millions) $ 18,400 $ 9,901 $ 5,928",
        doc_name="3M_2022_10K",
        table=False,
    )
    statement = _evidence(
        48,
        "[DOCUMENT: 3M_2022_10K | PAGE: 48]\n"
        "3M Company and Subsidiaries\n"
        "Consolidated Statement of Income\n"
        "Years ended December 31\n"
        "(Millions, except per share amounts) 2022 2021 2020\n"
        "Net sales $ 34,229 $ 35,355 $ 32,184",
        doc_name="3M_2022_10K",
        table=False,
    )
    header_only = _evidence(
        123,
        "Business Segment Information (Millions)\nNet Sales 2022 2021 2020",
        doc_name="3M_2022_10K",
        table=False,
    )

    answer = LocalSynthesizer._table_answer(
        "Compare net sales for 3M and AMD in 2022.",
        [segment, statement, header_only],
    )

    assert answer is not None
    assert "Page 48" in answer
    assert "34,229" in answer
    assert "18,400" not in answer
    assert "Page 123" not in answer


def test_a_statement_tie_falls_back_to_the_earlier_page() -> None:
    """Adobe 2022: the summary and the segment table both state total revenue 17,606."""
    summary = _evidence(
        40,
        "EXTRACTED TABLES (PAGE 40)\n| (Millions) | Col2 |\n| --- | --- |\n| Total revenue | 17,606 |",
        doc_name="ADOBE_2022_10K",
    )
    segments = _evidence(
        41,
        "EXTRACTED TABLES (PAGE 41)\n| (Millions) | Col2 |\n| --- | --- |\n| Total revenue | 17,606 |",
        doc_name="ADOBE_2022_10K",
    )

    for chunks in ([summary, segments], [segments, summary]):
        answer = LocalSynthesizer._table_answer(
            "Compare total revenue for Adobe and AMD in 2022.", chunks
        )
        assert answer is not None
        assert "Page 40" in answer
        assert "Page 41" not in answer


def test_net_revenues_plural_is_recognised_as_the_revenue_statement() -> None:
    """Activision 2019 words it "net revenues"; the singular-only pattern sent it to a note."""
    note = _evidence(
        29,
        "EXTRACTED TABLES (PAGE 29)\n"
        "| Valuation | Col2 |\n"
        "| --- | --- |\n"
        "| Activision Blizzard, Inc. | 100.00 |",
        doc_name="ACTIVISIONBLIZZARD_2019_10K",
        score=0.9,
    )
    statement = _evidence(
        37,
        "EXTRACTED TABLES (PAGE 37)\n"
        "| (Millions) | Col3 |\n"
        "| --- | --- |\n"
        "| Net revenues | 6,489 |",
        doc_name="ACTIVISIONBLIZZARD_2019_10K",
        score=0.1,
    )

    answer = LocalSynthesizer._table_answer(
        "What were Activision Blizzard's net revenues in 2019?", [note, statement]
    )

    assert answer is not None
    assert "Page 37" in answer
    assert "6,489" in answer
    assert "Page 29" not in answer
