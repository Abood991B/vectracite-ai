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


def test_numeric_verification_does_not_claim_text_without_figures_is_grounded() -> None:
    evidence = [SimpleNamespace(text="The agreement requires annual review.", contains_table=False)]
    result = DeterministicNumericVerifier().verify("The agreement expires tomorrow.", evidence)
    assert result.verification_applicable is False
    assert result.is_fully_grounded is False


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
    from src.usecases.synthesis import LocalSynthesizer

    docs = ["AMD_2022_10K", "MSFT_2023_10K"]
    assert LocalSynthesizer._is_cross_document_request(
        "Compare AMD and MSFT revenue.", docs
    )
    assert not LocalSynthesizer._is_cross_document_request(
        "What was AMD revenue?", docs
    )


def test_scope_inference_does_not_pick_one_side_of_a_comparison() -> None:
    from src.api.main import retriever
    from src.usecases.scoping import infer_scope_from_question

    client.get("/documents")
    indexed = retriever.indexed_document_names()
    assert infer_scope_from_question("Compare AMD and MSFT revenue.", indexed) is None
    assert (
        infer_scope_from_question("What was AMD revenue in 2022?", indexed)
        == "AMD_2022_10K"
    )


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


async def test_document_question_is_not_treated_as_product_question() -> None:
    from src.inference.llm_client import UnifiedLLMClient

    answer = await UnifiedLLMClient().generate_answer("What is this report about?", [])
    assert "No matching information" in answer


def test_uploaded_pdf_citations_and_images_use_physical_pages() -> None:
    import base64
    import pymupdf as fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 80), "SOURCE PAGE: 999 This is text on the first physical page.")
    payload = document.tobytes()
    document.close()
    name = "PHYSICAL_PAGE_REGRESSION"
    try:
        response = client.post("/ingest/pdf", json={
            "doc_name": name,
            "pdf_base64": base64.b64encode(payload).decode("ascii"),
        })
        assert response.status_code == 200
        chunks = client.get(f"/documents/{name}").json()["chunks"]
        assert {chunk["page_number"] for chunk in chunks} == {1}
        assert client.get(f"/documents/{name}/pages/1/image").status_code == 200
        assert client.get(f"/documents/{name}/pages/999/image").status_code == 404
    finally:
        client.delete(f"/documents/{name}")


def test_cleared_demos_stay_cleared_when_index_is_rebuilt() -> None:
    import base64
    import pymupdf as fitz
    from src.api.main import _ensure_indexed

    document = fitz.open()
    document.new_page().insert_text((50, 80), "A user document that survives an index rebuild.")
    pdf = document.tobytes()
    document.close()
    name = "PERSISTED_USER_DOCUMENT"
    try:
        assert client.get("/documents").json()["total_documents"] > 0
        assert client.post("/documents/clear-demos").status_code == 200
        assert client.post("/ingest/pdf", json={
            "doc_name": name,
            "pdf_base64": base64.b64encode(pdf).decode("ascii"),
        }).status_code == 200
        retriever.clear()
        _ensure_indexed()
        names = {item["doc_name"] for item in client.get("/documents").json()["documents"]}
        assert names == {name}
    finally:
        client.delete(f"/documents/{name}")
        client.post("/documents/restore-defaults")


def test_two_column_prose_is_not_interpreted_as_a_table() -> None:
    import pymupdf as fitz
    from src.config import settings
    from src.ingestion.pdf_parser import MultimodalFinancialParser

    source = fitz.open(settings.BASE_DIR / "data/domain_corpus_pdfs/CDC_MMWR_72_42.pdf")
    page = fitz.open()
    page.insert_pdf(source, from_page=1, to_page=1)
    chunks = MultimodalFinancialParser().parse_pdf_bytes(page.tobytes(), "cdc_prose")
    page.close()
    source.close()
    assert chunks
    assert not any(chunk.contains_table for chunk in chunks)


async def test_timed_out_ingestion_keeps_its_slot_until_worker_stops(monkeypatch) -> None:
    import asyncio
    import threading
    import pytest
    from fastapi import HTTPException
    from src.api.main import _run_ingestion_step
    from src.config import settings

    stopping = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow(*, cancel_event):
        try:
            assert cancel_event.wait(5)
            stopping.set()
            release.wait(5)
        finally:
            finished.set()

    monkeypatch.setattr(settings, "INGEST_TIMEOUT_SECONDS", 0.05)
    try:
        with pytest.raises(HTTPException) as timeout:
            await _run_ingestion_step(slow)
        assert timeout.value.status_code == 408
        assert await asyncio.to_thread(stopping.wait, 2)
        with pytest.raises(HTTPException) as busy:
            await _run_ingestion_step(slow)
        assert busy.value.status_code == 429
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 2)


def test_old_answer_cache_is_not_reused_after_page_number_migration() -> None:
    import fnmatch
    import json

    query = "ALL::top_k=4::What was AMD revenue?"
    entry = {
        "query": query,
        "vector": SemanticCache._embed_query_fast(query).tolist(),
        "response": {"answer": "old citation", "citations": [{"page_number": 999}]},
    }

    class OldCache:
        def get(self, key):
            return None

        def scan_iter(self, match, count):
            del count
            return iter(["vectracite:semcache:old"] if fnmatch.fnmatch("vectracite:semcache:old", match) else [])

        def mget(self, keys):
            return [json.dumps(entry) for _ in keys]

    cache = SemanticCache(redis_url="redis://localhost:6399/0")
    cache._redis_client = OldCache()
    assert cache.lookup(query) is None


def test_new_cache_namespace_preserves_old_entries_on_clear() -> None:
    import fnmatch

    class RedisStore:
        def __init__(self):
            self.values = {"vectracite:semcache:old": "old data"}

        def get(self, key):
            return self.values.get(key)

        def scan_iter(self, match, count):
            del count
            return iter([key for key in self.values if fnmatch.fnmatch(key, match)])

        def mget(self, keys):
            return [self.values.get(key) for key in keys]

        def pipeline(self):
            return self

        def set(self, key, value, ex):
            del ex
            self.values[key] = value

        def execute(self):
            return None

        def delete(self, *keys):
            for key in keys:
                self.values.pop(key, None)

    redis = RedisStore()
    cache = SemanticCache(redis_url="redis://localhost:6399/0")
    cache._redis_client = redis
    cache.store("What was AMD revenue?", {"answer": "new answer"})
    assert cache.lookup("What was AMD revenue?")["answer"] == "new answer"
    cache.clear()
    assert cache.lookup("What was AMD revenue?") is None
    assert redis.values == {"vectracite:semcache:old": "old data"}


def test_chunk_id_changes_when_text_after_first_100_characters_changes() -> None:
    import pymupdf as fitz
    from src.ingestion.pdf_parser import MultimodalFinancialParser

    def pdf_with_tail(tail):
        document = fitz.open()
        page = document.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 560, 250), ("shared sentence " * 12) + tail)
        payload = document.tobytes()
        document.close()
        return payload

    parser = MultimodalFinancialParser()
    first = parser.parse_pdf_bytes(pdf_with_tail("Revenue is 100."), "report")
    second = parser.parse_pdf_bytes(pdf_with_tail("Revenue is 200."), "report")
    assert first[0].text != second[0].text
    assert first[0].chunk_id != second[0].chunk_id


def test_streamed_upload_reports_pages_and_commits_document() -> None:
    import base64
    import json
    import pymupdf as fitz

    document = fitz.open()
    for text in ("First page has useful content.", "Second page has other useful content."):
        document.new_page().insert_text((50, 80), text)
    payload = document.tobytes()
    document.close()
    name = "STREAM_UPLOAD_REGRESSION"
    try:
        response = client.post("/ingest/pdf/stream", json={
            "doc_name": name, "pdf_base64": base64.b64encode(payload).decode("ascii"),
        })
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert [(e["page"], e["total_pages"]) for e in events if e["stage"] == "parsing"] == [(1, 2), (2, 2)]
        assert events[-1]["stage"] == "complete"
        assert client.get(f"/documents/{name}").status_code == 200
    finally:
        client.delete(f"/documents/{name}")


async def test_disconnected_upload_cancels_parsing_before_commit(monkeypatch) -> None:
    import asyncio
    import base64
    import threading
    import pymupdf as fitz
    from src.api import main
    from src.inference.schemas import PdfUploadRequest

    document = fitz.open()
    document.new_page().insert_text((50, 80), "A page for a cancelled upload.")
    pdf = document.tobytes()
    document.close()
    stopped = threading.Event()

    def blocked_parse(raw_bytes, clean_name, *, cancel_event, progress):
        progress(1, 2)
        if cancel_event.wait(5):
            stopped.set()
        raise ValueError("cancelled")

    monkeypatch.setattr(main, "_parse_uploaded_pdf", blocked_parse)
    response = await main.ingest_pdf_stream(PdfUploadRequest(
        doc_name="CANCELLED_UPLOAD", pdf_base64=base64.b64encode(pdf).decode("ascii")
    ))
    assert (await asyncio.wait_for(anext(response.body_iterator), 5)).startswith(":")
    event = await asyncio.wait_for(anext(response.body_iterator), 5)
    assert '"stage": "parsing"' in event
    await response.body_iterator.aclose()
    assert await asyncio.to_thread(stopped.wait, 2)
    assert not (main.settings.USER_UPLOADS_DIR / "CANCELLED_UPLOAD.pdf").exists()


def test_empty_catalog_is_ready_to_accept_uploads(monkeypatch, tmp_path) -> None:
    from src.adapters.uploads import PdfFileStore
    from src.api import main
    from src.retrieval.hybrid_retriever import HybridRetriever

    (tmp_path / ".defaults-cleared").touch()
    monkeypatch.setattr(main.settings, "USER_UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(main, "upload_store", PdfFileStore(tmp_path, tmp_path))
    monkeypatch.setattr(main, "retriever", HybridRetriever(in_memory=True, use_lightweight_embedder=True))
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["indexed_documents"] == 0


async def test_load_benchmark_rejects_stream_without_answer_and_verification() -> None:
    import httpx
    import pytest
    from training_and_vllm.benchmark_vllm import _send_query

    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, text="event: done\ndata: [DONE]\n\n"
    ))
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(ValueError, match="incomplete"):
            await _send_query(http, "http://test/query/stream", "A question?")


async def test_load_benchmark_accepts_complete_answer_stream() -> None:
    import httpx
    from training_and_vllm.benchmark_vllm import _send_query

    body = (
        'event: token\ndata: {"text":"An answer"}\n\n'
        'event: verification\ndata: {"answer":"An answer","verification":{}}\n\n'
        'event: done\ndata: [DONE]\n\n'
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=body)
    )) as http:
        latency, ttft = await _send_query(http, "http://test/query/stream", "A question?")
    assert ttft is not None and 0 <= ttft <= latency


def test_load_benchmark_percentile_uses_nearest_rank() -> None:
    from training_and_vllm.benchmark_vllm import _percentile

    assert _percentile([1.0, 10.0], 0.95) == 10.0


def test_provider_metadata_does_not_expose_url_credentials_or_exception_text() -> None:
    provider = UnifiedLLMClient(base_url="https://user:private-value@example.org/v1?key=private-value")
    provider._mark_remote_failure(RuntimeError("provider returned private-value"))
    status = provider.get_provider_status()
    assert "private-value" not in str(status)
    assert status["base_url"] == "https://example.org/v1"
    assert status["last_error"] == "RuntimeError"
