"""Contract pins for the two endpoints that had no test of their own.

`/metrics` and `/evaluate` back the workbench dashboard and the evaluation flow, but neither had a
pinned response shape. A silent field rename would break the UI without failing a test.
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

EVALUATION_MODES = ("dense_only", "hybrid_rrf", "hybrid_plus_reranker")


def test_metrics_reports_index_cache_and_provider_state() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    for key in (
        "indexed_chunks",
        "indexed_documents",
        "semantic_cache",
        "embedding_model",
        "reranker_model",
        "llm_model",
        "llm_api_key_configured",
        "llm_remote_enabled",
        "retriever",
    ):
        assert key in body

    assert isinstance(body["indexed_chunks"], int)
    assert isinstance(body["indexed_documents"], int)
    assert isinstance(body["llm_api_key_configured"], bool)
    assert body["llm_remote_enabled"] is False

    cache = body["semantic_cache"]
    for key in ("backend", "vector_index", "degraded", "hits", "misses", "hit_rate"):
        assert key in cache
    assert isinstance(cache["hits"], int)
    assert 0.0 <= cache["hit_rate"] <= 1.0


def test_a_non_json_body_is_a_readable_422_not_a_server_error() -> None:
    """A multipart body sent to a JSON endpoint used to fail while encoding the error itself."""
    response = client.post(
        "/ingest/pdf",
        content=b"%PDF-1.7 not really json",
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body"]


def test_a_binary_query_body_is_a_readable_422() -> None:
    response = client.post(
        "/query",
        content=b"\xff\xfe\x00binary",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code in (400, 422)
    assert "detail" in response.json()


def test_evaluate_runs_the_ablation_and_returns_a_signature() -> None:
    response = client.post("/evaluate")

    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {
        *EVALUATION_MODES,
        "per_query_breakdown",
        "skipped_cases",
        "evaluation_signature",
    }
    assert isinstance(body["evaluation_signature"], str) and body["evaluation_signature"]
    for mode in EVALUATION_MODES:
        summary = body[mode]
        for metric in ("precision_at_1", "recall_at_3", "mrr", "ndcg_at_3"):
            assert 0.0 <= summary[metric] <= 1.0
        assert summary["avg_latency_ms"] >= 0.0
        assert summary["queries_evaluated"] >= 0
