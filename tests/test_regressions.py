"""Regression coverage for real-corpus and guardrail defects."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.main import app, retriever
from src.cache_and_guardrails.numeric_verifier import DeterministicNumericVerifier
from src.cache_and_guardrails.semantic_cache import SemanticCache
from src.inference.llm_client import UnifiedLLMClient
from src.retrieval.table_ast import TableASTParser

client = TestClient(app)


def test_match_query_cells_preserves_source_column_order_for_deltas() -> None:
    markdown = (
        "| Item | FY 2022 | FY 2021 |\n"
        "| --- | --- | --- |\n"
        "| Net income | $1,200 | $800 |\n"
    )
    matched = TableASTParser.match_query_cells("net income FY 2021", markdown, "D", 1)
    assert [cell.col_header for cell in matched] == ["FY 2021", "FY 2022"]
    deltas = TableASTParser.compute_row_deltas(matched)
    assert deltas == [
        {
            "row_key": "Net income",
            "comparison": "FY 2022 vs FY 2021",
            "absolute_change": "+400.00",
            "pct_change": "+50.00%",
        }
    ]


def test_numeric_verifier_marks_citation_free_answers_not_applicable() -> None:
    result = DeterministicNumericVerifier().verify("The model is Llama-3.1-8B.", [])
    assert result.verification_applicable is False
    assert result.is_fully_grounded is False
    assert result.numeric_grounding_score == 0.0


def test_semantic_cache_rejects_entity_year_and_top_k_mismatches() -> None:
    cache = SemanticCache(redis_url="redis://localhost:6399/0")
    cache._redis_client = None
    cache.store(
        "AMD::top_k=5::What was AMD revenue in 2022?",
        {"answer": "2022 answer"},
    )
    assert cache.lookup("AMD::top_k=5::What was AMD revenue in 2022?") is not None
    assert cache.lookup("AMD::top_k=5::What was AMD revenue in 2023?") is None
    assert cache.lookup("AMD::top_k=5::What was MSFT revenue in 2022?") is None
    assert cache.lookup("AMD::top_k=10::What was AMD revenue in 2022?") is None


def test_answer_contributors_accepts_lowercase_and_page_abbreviations() -> None:
    from src.api.answer_service import answer_contributors

    chunks = [
        SimpleNamespace(chunk_id="c1", doc_name="AMD_2022_10K", page_number=51),
        SimpleNamespace(chunk_id="c2", doc_name="AMD_2022_10K", page_number=52),
    ]
    answer = "Revenue was $23,601 [AMD_2022_10K, p. 51] and [AMD_2022_10K, page 52]."
    contributors = answer_contributors(answer, chunks)
    assert [chunk.chunk_id for chunk in contributors] == ["c1", "c2"]


def test_cross_document_detection_uses_indexed_document_names() -> None:
    docs = ["AMD_2022_10K", "MSFT_2023_10K"]
    assert UnifiedLLMClient._is_cross_document_request(
        "Compare AMD and MSFT revenue.", docs
    )
    assert not UnifiedLLMClient._is_cross_document_request(
        "What was AMD revenue?", docs
    )


def test_scope_inference_does_not_pick_one_side_of_a_comparison() -> None:
    from src.api.main import _infer_scope_from_question

    client.get("/documents")
    assert _infer_scope_from_question("Compare AMD and MSFT revenue.") is None
    assert _infer_scope_from_question("What was AMD revenue in 2022?") == "AMD_2022_10K"


def test_document_inspector_paginates_and_rejects_out_of_range_pages() -> None:
    response = client.get("/documents/AMD_2022_10K?limit=2")
    assert response.status_code == 200
    body = response.json()
    assert body["returned_chunks"] <= 2
    assert body["limit"] == 2
    assert body["truncated"] is True

    image = client.get("/documents/AMD_2022_10K/pages/999/image")
    assert image.status_code == 404


def test_scenarios_exclude_documents_that_are_not_indexed() -> None:
    try:
        assert client.post("/documents/clear-demos").status_code == 200
        scenarios = client.get("/scenarios").json()["scenarios"]
        assert scenarios == []
    finally:
        assert client.post("/documents/restore-defaults").status_code == 200


def test_retriever_health_reports_reachability() -> None:
    client.get("/documents")
    health = retriever.health()
    assert health["reachable"] is True
    assert health["degraded"] is False
