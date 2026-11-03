"""Pin the page-role and statement-line rules that keep a citation on the audited statement.

An annual report states each reported figure several times: in the consolidated statement, in an
MD&A table, in a note, and in a selected-financial-data summary. A figure repeated in a note is
still the company's figure, but it is not the page that reports it. These tests pin the rules
that decide page role from a page's opening chunk and read the reporting line off the statement.

The labelled sets in ``data/retrieval_quality_*.json`` score the whole path end to end. These
tests pin each rule alone, because each one is enough to cite a page that restates the number
instead of the statement that reports it.
"""

from qdrant_client import QdrantClient

import pytest

from src.domain.documents import DocumentChunk, RetrievedDocument
from src.domain.financial_statements import asked_statement_families, is_statement_heading
from src.retrieval.hybrid_retriever import HybridRetriever
from src.usecases.synthesis import LocalSynthesizer, wants_table_evidence

DOC = "STATEMENT_DOC"


@pytest.fixture()
def retriever():
    return HybridRetriever(
        collection_name="statement_selection",
        in_memory=True,
        client=QdrantClient(":memory:"),
    )


def _evidence(
    page: int,
    text: str,
    *,
    doc_name: str,
    score: float = 0.5,
    chunk_index: int = 0,
) -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id=f"{doc_name}-{page}-{chunk_index}",
        doc_name=doc_name,
        page_number=page,
        text=text,
        contains_table=False,
        dense_score=score,
        bm25_score=score,
        rrf_score=score,
        rerank_score=score,
        chunk_index=chunk_index,
    )


def test_a_heading_broken_by_extraction_is_still_a_statement_heading() -> None:
    """3M's FY2018 heading extracts as "Consolidated Statement of Incom e"."""
    assert is_statement_heading(
        "[DOCUMENT: 3M_2018_10K | PAGE: 56] Table of Contents 3M Company and Subsidiaries "
        "Consolidated Statement of Incom e Years ended December 31"
    )


def test_a_balance_sheet_heading_is_a_statement_heading() -> None:
    assert is_statement_heading(
        "[DOCUMENT: ACTIVISIONBLIZZARD_2019_10K | PAGE: 69] Table of Contents "
        "ACTIVISION BLIZZARD, INC. AND SUBSIDIARIES CONSOLIDATED BALANCE SHEETS "
        "(Amounts in millions, except share data) At December 31, 2019"
    )


def test_an_audit_report_that_describes_the_statements_is_not_one() -> None:
    assert not is_statement_heading(
        "We have audited the accompanying consolidated balance sheets of Activision Blizzard, "
        "Inc. and its subsidiaries as of December 31, 2019 and 2018"
    )


def test_a_note_that_mentions_the_statement_is_not_one() -> None:
    assert not is_statement_heading(
        "Stock-based compensation expense was allocated in the consolidated statements of "
        "operations as follows: Year Ended December 31, 2022"
    )


def test_a_table_of_contents_listing_several_statements_is_not_one() -> None:
    assert not is_statement_heading(
        "Table of Contents Report of Independent Registered Public Accounting Firm F-1 "
        "Consolidated Balance Sheets at December 31, 2019 and 2018 F-4 "
        "Consolidated Statements of Operations for the Years Ended December 31 F-5"
    )


def test_a_question_names_the_statement_it_asks_for() -> None:
    assert "statements?ofcashflows" in asked_statement_families(
        "What is capex? Rely on the details shown in the cash flow statement."
    )
    assert "balancesheets?" in asked_statement_families("What were total assets on the balance sheet?")
    assert asked_statement_families("What were total assets?") == frozenset()


def test_a_named_cash_flow_statement_beats_a_balance_sheet_that_repeats_the_metric() -> None:
    """The balance sheet also reports property, plant and equipment, in a different role.

    Asked for capital expenditure from the cash flow statement, the answer must cite the spending
    line on that statement, not the asset balance the balance sheet carries under the same words.
    """
    balance_sheet = _evidence(
        58,
        "[DOCUMENT: 3M_2018_10K | PAGE: 58] Table of Contents 3M Company and Subsidiaries "
        "Consolidated Balance Sheets At December 31 2018 2017 "
        "Property, plant and equipment 24,873 24,914 "
        "Property, plant and equipment - net 8,738 8,866",
        doc_name="3M_2018_10K",
        score=0.9,
    )
    cash_flow = _evidence(
        60,
        "[DOCUMENT: 3M_2018_10K | PAGE: 60] Table of Contents 3M Company and Subsidiaries "
        "Consolidated Statement of Cash Flows Years ended December 31 "
        "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)",
        doc_name="3M_2018_10K",
        score=0.4,
    )

    answer = LocalSynthesizer().answer(
        "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response "
        "to the question by relying on the details shown in the cash flow statement.",
        [balance_sheet, cash_flow],
        ["3M_2018_10K"],
    )

    assert "Page 60" in answer
    assert "1,577" in answer
    assert "Page 58" not in answer


def test_the_statement_scan_returns_every_chunk_of_the_statement_pages(retriever) -> None:
    retriever.index_chunks(
        [
            DocumentChunk(
                chunk_id="stmt-heading",
                doc_name=DOC,
                page_number=54,
                text=(
                    "[DOCUMENT: STATEMENT_DOC | PAGE: 54] ADOBE INC. "
                    "CONSOLIDATED STATEMENTS OF INCOME (In millions) 2022 2021 2020"
                ),
                contains_table=False,
                metadata={"chunk_index": 0},
            ),
            DocumentChunk(
                chunk_id="stmt-rows",
                doc_name=DOC,
                page_number=54,
                text="Gross profit 15,441 13,920 11,146",
                contains_table=False,
                metadata={"chunk_index": 1},
            ),
            DocumentChunk(
                chunk_id="note-rows",
                doc_name=DOC,
                page_number=65,
                text="Gross profit 12,281 2,920",
                contains_table=False,
                metadata={"chunk_index": 0},
            ),
        ],
        use_embedding_cache=False,
    )

    assert [chunk.chunk_id for chunk in retriever.statement_candidates(DOC)] == [
        "stmt-heading",
        "stmt-rows",
    ]
    assert retriever.statement_candidates("OTHER_DOC") == []


def test_a_note_that_restates_the_figure_does_not_win_the_citation() -> None:
    """Adobe 2022: the statement (p54) reports gross profit; a note table (p65) repeats it."""
    note = _evidence(
        65,
        "EXTRACTED TABLES (PAGE 65)\n"
        "| (Millions) | Col2 |\n"
        "| --- | --- |\n"
        "| Gross profit | 15,441 |",
        doc_name="ADOBE_2022_10K",
        score=0.9,
    )
    statement = _evidence(
        54,
        "[DOCUMENT: ADOBE_2022_10K | PAGE: 54] ADOBE INC. CONSOLIDATED STATEMENTS OF INCOME "
        "(In millions, except per share data) Years Ended 2022 2021 2020\n"
        "Gross profit 15,441 13,920 11,146",
        doc_name="ADOBE_2022_10K",
        score=0.1,
    )

    answer = LocalSynthesizer._table_answer(
        "What was Adobe's gross profit in 2022?", [note, statement]
    )

    assert answer is not None
    assert "Page 54" in answer
    assert "Gross profit 15,441 13,920 11,146" in answer
    assert "Page 65" not in answer


def test_a_total_row_beats_a_subtotal_that_adds_a_word_the_question_never_used() -> None:
    """Activision 2019: "Total assets", not the shorter "Total current assets"."""
    statement = _evidence(
        69,
        "[DOCUMENT: ACTIVISIONBLIZZARD_2019_10K | PAGE: 69] Table of Contents "
        "ACTIVISION BLIZZARD, INC. AND SUBSIDIARIES CONSOLIDATED BALANCE SHEETS "
        "(Amounts in millions, except share data) At December 31, 2019\n"
        "Total current assets 7,292 6,106\n"
        "Total assets $ 19,845 $ 17,890",
        doc_name="ACTIVISIONBLIZZARD_2019_10K",
    )

    answer = LocalSynthesizer._table_answer(
        "What were Activision Blizzard's total assets at December 31, 2019?", [statement]
    )

    assert answer is not None
    assert "Total assets $ 19,845 $ 17,890" in answer
    assert "Total current assets" not in answer


def test_a_cost_row_does_not_answer_a_revenue_question() -> None:
    """"Total cost of sales" contains "sales" without being revenue."""
    statement = _evidence(
        54,
        "EXTRACTED TABLES (PAGE 54)\n"
        "| (Millions) | Col2 |\n"
        "| --- | --- |\n"
        "| Total net revenue | 23,601 |\n"
        "| Total cost of sales | 12,998 |",
        doc_name="AMD_2022_10K",
    )

    answer = LocalSynthesizer._table_answer(
        "What was AMD's total revenue in 2022?", [statement]
    )

    assert answer is not None
    assert "23,601" in answer
    assert "12,998" not in answer


def test_a_cost_row_still_answers_a_cost_question() -> None:
    """The exclusion is about the metric asked for, not about the word "cost"."""
    statement = _evidence(
        54,
        "[DOCUMENT: AMD_2022_10K | PAGE: 54] Advanced Micro Devices, Inc. "
        "Consolidated Statements of Operations Year Ended December 31 2022 2021 2020\n"
        "Total cost of sales 12,998 8,505 5,416",
        doc_name="AMD_2022_10K",
    )

    answer = LocalSynthesizer._table_answer(
        "What was AMD's total cost of sales in 2022?", [statement]
    )

    assert answer is not None
    assert "12,998" in answer


def test_an_idiomatic_make_up_is_not_a_table_question() -> None:
    """"Which four functions make up the AI RMF Core?" was routed to table synthesis by "make"."""
    assert not wants_table_evidence("Which four functions make up the AI RMF Core?")


def test_a_revenue_question_with_make_is_still_a_table_question() -> None:
    """Dropping the bare verbs must not drop genuine financial questions."""
    assert wants_table_evidence("How much revenue did AMD make in 2022?")


def test_a_yes_no_question_is_not_a_table_question_for_its_year_token() -> None:
    """FY22 matched the year pattern and sent a risk-factor question to a "customer" table row."""
    assert not wants_table_evidence("Did AMD report customer concentration in FY22?")
    assert not wants_table_evidence("Is 3M a capital-intensive business based on FY2022 data?")
