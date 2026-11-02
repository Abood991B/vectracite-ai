"""Integration checks that exercise the checked-in real SEC filing corpus."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import settings
from src.ingestion.corpus_cache import load_corpus_chunks
from src.ingestion.pdf_parser import MultimodalFinancialParser
from src.retrieval.hybrid_retriever import HybridRetriever

pytestmark = pytest.mark.integration
client = TestClient(app)


def _real_corpus() -> tuple:
    corpus = settings.BASE_DIR / "data" / "corpus_pdfs"
    index = settings.BASE_DIR / "data" / "corpus_index.json"
    embeddings = settings.BASE_DIR / "data" / "corpus_embeddings.npz"
    return corpus, index, embeddings


def test_real_manifest_matches_files(monkeypatch) -> None:
    from scripts.fetch_real_pdf_corpus import verify

    corpus, _, _ = _real_corpus()
    monkeypatch.setattr(settings, "CORPUS_PDF_DIR", corpus)
    verify()


def test_real_corpus_chunks_and_benchmark_pages_exist(monkeypatch) -> None:
    corpus, index, _ = _real_corpus()
    monkeypatch.setattr(settings, "CORPUS_PDF_DIR", corpus)
    parser = MultimodalFinancialParser(
        chunk_size=settings.PARSER_CHUNK_SIZE,
        chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
    )
    chunks = load_corpus_chunks(corpus, parser, cache_path=index, write=False)
    docs = {chunk.doc_name for chunk in chunks}
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert docs == set(manifest["documents"])
    assert len(chunks) >= 4000

    available = {(chunk.doc_name, chunk.page_number) for chunk in chunks}
    queries = json.loads(
        (settings.BASE_DIR / "data" / "benchmark_queries.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(queries) == 16
    for query in queries:
        pages = query.get("evidence_pages") or [query["page_number"]]
        assert any((query["doc_name"], page) in available for page in pages)


def test_real_retrieval_finds_a_known_evidence_page(monkeypatch) -> None:
    corpus, index, embeddings = _real_corpus()
    monkeypatch.setattr(settings, "CORPUS_PDF_DIR", corpus)
    monkeypatch.setattr(settings, "CORPUS_INDEX_PATH", index)
    monkeypatch.setattr(settings, "EMBEDDING_CACHE_PATH", embeddings)
    parser = MultimodalFinancialParser(
        chunk_size=settings.PARSER_CHUNK_SIZE,
        chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
    )
    chunks = load_corpus_chunks(corpus, parser, cache_path=index, write=False)
    retriever = HybridRetriever(in_memory=True, use_lightweight_embedder=False)
    retriever.index_chunks(chunks, use_embedding_cache=True)
    query = "Did AMD report customer concentration in FY22?"
    hits = retriever.retrieve_hybrid(query, top_k=5)
    assert any(hit.doc_name == "AMD_2022_10K" and hit.page_number == 12 for hit in hits)


def test_real_pdf_and_page_image_endpoints(monkeypatch) -> None:
    corpus, _, _ = _real_corpus()
    monkeypatch.setattr(settings, "CORPUS_PDF_DIR", corpus)
    pdf = client.get("/documents/AMD_2022_10K/pdf")
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF-")
    image = client.get("/documents/AMD_2022_10K/pages/56/image?highlight=quick%20ratio")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")
