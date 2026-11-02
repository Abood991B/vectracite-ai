"""Unit and integration checks for the domain-agnostic benchmark harness."""

from __future__ import annotations

import json

import pytest

from evaluation.domain_benchmark import (
    DomainBenchmarkSpec,
    DomainCase,
    _ragbench_relevant_doc_indices,
    load_benchmark_chunks,
    load_domain_benchmark_spec,
    run_domain_benchmark,
)
from src.config import settings
from src.ingestion.pdf_parser import DocumentChunk


def _chunk(chunk_id: str, doc_name: str, page: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_name=doc_name,
        page_number=page,
        text=text,
        contains_table=False,
        metadata={"source": f"{doc_name}.pdf", "page": page, "chunk_index": 0},
    )


def _synthetic_case(
    case_id: str,
    domain: str,
    doc_name: str,
    page: int,
    question: str,
) -> DomainCase:
    return DomainCase(
        case_id=case_id,
        domain=domain,
        doc_name=doc_name,
        evidence_pages=(page,),
        question=question,
        gold_answer="synthetic",
        source="unit",
    )


def test_run_domain_benchmark_produces_summary_and_paired_comparisons(tmp_path) -> None:
    chunks = [
        _chunk("a1", "ALPHA", 1, "alpha revenue grew by ten percent"),
        _chunk("b1", "BETA", 2, "beta safety review found no incidents"),
        _chunk("c1", "GAMMA", 3, "gamma legal contract requires notice"),
    ]
    cases = (
        _synthetic_case("case_alpha", "finance", "ALPHA", 1, "alpha revenue growth"),
        _synthetic_case("case_beta", "public_health", "BETA", 2, "beta safety review"),
        _synthetic_case("case_gamma", "legal", "GAMMA", 3, "gamma legal notice"),
    )
    report = run_domain_benchmark(
        cases=cases,
        chunks=chunks,
        spec=DomainBenchmarkSpec(cases=cases),
        corpus_dirs=(),
        top_k=3,
        bootstrap_samples=100,
        seed=7,
        lightweight=True,
        output_path=tmp_path / "report.json",
    )

    assert report["case_count"] == 3
    assert report["domains"] == ["finance", "legal", "public_health"]
    assert set(report["summary"]) == {
        "dense_only",
        "hybrid_rrf",
        "hybrid_plus_reranker",
    }
    assert report["summary"]["dense_only"]["queries_evaluated"] == 3
    assert "recall_at_3" in report["summary"]["hybrid_rrf"]["confidence_intervals"]
    assert "hybrid_rrf_vs_dense_only" in report["pairwise"]
    assert report["per_query"][0]["ranks"]["dense_only"] == 1
    assert (tmp_path / "report.json").exists()
    assert len(report["signature"]) == 64


def test_load_domain_spec_rejects_duplicate_case_ids(tmp_path) -> None:
    payload = {
        "schema_version": 1,
        "corpus_dirs": ["data/domain_corpus_pdfs"],
        "cases": [
            {
                "case_id": "duplicate",
                "domain": "finance",
                "doc_name": "DOC",
                "evidence_pages": [1],
                "question": "first",
            },
            {
                "case_id": "duplicate",
                "domain": "finance",
                "doc_name": "DOC",
                "evidence_pages": [1],
                "question": "second",
            },
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_domain_benchmark_spec(path)


def test_load_domain_spec_rejects_missing_evidence_locator(tmp_path) -> None:
    payload = {
        "schema_version": 1,
        "cases": [
            {
                "case_id": "missing",
                "domain": "finance",
                "question": "no evidence",
            }
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_chunk_ids"):
        load_domain_benchmark_spec(path)


def test_ragbench_relevance_parser_handles_document_and_sentence_keys() -> None:
    row = {
        "all_relevant_sentence_keys": ["0ac", "1b"],
        "all_utilized_sentence_keys": ["2d"],
        "sentence_support_information": [
            {"supporting_sentence_keys": ["3a", "ignored"]},
        ],
    }
    assert _ragbench_relevant_doc_indices(row, 4) == {0, 1, 2, 3}
    assert _ragbench_relevant_doc_indices(row, 3) == {0, 1, 2}


@pytest.mark.integration
def test_checked_in_domain_corpus_manifest_and_evidence_pages(monkeypatch) -> None:
    corpus = settings.BASE_DIR / "data" / "domain_corpus_pdfs"
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    documents = manifest["documents"]
    assert set(documents) == {
        "CDC_MMWR_72_42",
        "NIST_AI_RMF_1_0",
        "SCOTUS_303_Creative_v_Elenis",
        "ARXIV_LLaMA_Open_Efficient_Foundation_LMs",
    }
    for document in documents.values():
        path = corpus / document["filename"]
        assert path.exists()
        assert path.stat().st_size == document["bytes"]
        assert path.read_bytes().startswith(b"%PDF-")

    monkeypatch.setattr(settings, "CORPUS_PDF_DIR", settings.BASE_DIR / "data" / "corpus_pdfs")
    monkeypatch.setattr(
        settings,
        "CORPUS_INDEX_PATH",
        settings.BASE_DIR / "data" / "corpus_index.json",
    )
    spec = load_domain_benchmark_spec(settings.BASE_DIR / "data" / "domain_benchmark_cases.json")
    chunks = load_benchmark_chunks(spec.corpus_dirs)
    available_pages = {(chunk.doc_name, chunk.page_number) for chunk in chunks}
    assert len({chunk.doc_name for chunk in chunks}) == 9
    for case in spec.cases:
        assert any((case.doc_name, page) in available_pages for page in case.evidence_pages)


@pytest.mark.integration
def test_checked_in_domain_benchmark_runs_end_to_end(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "CORPUS_PDF_DIR", settings.BASE_DIR / "data" / "corpus_pdfs")
    monkeypatch.setattr(
        settings,
        "CORPUS_INDEX_PATH",
        settings.BASE_DIR / "data" / "corpus_index.json",
    )
    spec = load_domain_benchmark_spec(settings.BASE_DIR / "data" / "domain_benchmark_cases.json")
    chunks = load_benchmark_chunks(spec.corpus_dirs)
    report = run_domain_benchmark(
        cases=spec.cases,
        chunks=chunks,
        spec=spec,
        corpus_dirs=spec.corpus_dirs,
        top_k=5,
        bootstrap_samples=50,
        seed=7,
        lightweight=True,
        output_path=tmp_path / "domain_report.json",
    )
    assert report["case_count"] == len(spec.cases)
    assert set(report["domains"]) == {
        "finance",
        "public_health",
        "ai_governance",
        "legal",
        "academic_ml",
    }
    assert all(
        mode["queries_evaluated"] == len(spec.cases)
        for mode in report["summary"].values()
    )
