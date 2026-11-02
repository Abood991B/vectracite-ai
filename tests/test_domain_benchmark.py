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
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedDocument
from src.retrieval.reranker import CrossEncoderReranker


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


def test_custom_retrieval_knobs_change_collection_and_fusion() -> None:
    retriever = HybridRetriever(
        in_memory=True,
        use_lightweight_embedder=True,
        embedding_dim=512,
        dense_weight=0.5,
        bm25_weight=0.5,
        candidate_multiplier=2,
    )
    retriever.index_chunks(
        [
            _chunk("a1", "ALPHA", 1, "alpha revenue"),
            _chunk("b1", "BETA", 2, "beta safety"),
        ]
    )
    assert retriever.collection_name.endswith("__lightweight__d512")
    assert retriever.embedding_dim == 512
    assert retriever.dense_weight == 0.5
    assert retriever.bm25_weight == 0.5
    results = retriever.retrieve_hybrid("alpha revenue", top_k=2)
    assert {result.doc_name for result in results} == {"ALPHA", "BETA"}


def test_page_fusion_mode_executes_and_reports_custom_modes() -> None:
    chunks = [
        _chunk("a1", "ALPHA", 1, "alpha revenue first passage"),
        _chunk("a2", "ALPHA", 1, "alpha revenue second passage"),
        _chunk("b1", "BETA", 2, "beta safety review"),
    ]
    cases = (
        _synthetic_case("case_alpha", "finance", "ALPHA", 1, "alpha revenue"),
        _synthetic_case("case_beta", "public_health", "BETA", 2, "beta safety"),
    )
    report = run_domain_benchmark(
        cases=cases,
        chunks=chunks,
        spec=DomainBenchmarkSpec(cases=cases),
        corpus_dirs=(),
        modes=("hybrid_rrf", "hybrid_rrf_page_fusion"),
        top_k=3,
        bootstrap_samples=20,
        seed=7,
        lightweight=True,
    )
    assert report["modes"] == ["hybrid_rrf", "hybrid_rrf_page_fusion"]
    assert set(report["summary"]) == {"hybrid_rrf", "hybrid_rrf_page_fusion"}
    assert report["pairwise"] == {}
    assert report["per_query"][0]["ranks"]["hybrid_rrf_page_fusion"] == 1


def test_forced_cross_encoder_scores_every_candidate(monkeypatch) -> None:
    calls: list[list[tuple[str, str]]] = []

    class FakeEncoder:
        def predict(self, pairs, show_progress_bar=False):
            del show_progress_bar
            calls.append(list(pairs))
            return [1.0 for _ in pairs]

    import src.retrieval.reranker as reranker_module

    monkeypatch.setattr(
        reranker_module,
        "get_shared_cross_encoder",
        lambda *args, **kwargs: FakeEncoder(),
    )
    candidates = [
        RetrievedDocument(
            chunk_id=f"c{index}",
            doc_name="ALPHA",
            page_number=index,
            text=f"plain evidence {index}",
            contains_table=False,
            dense_score=0.1,
            bm25_score=0.1,
            rrf_score=1.0 / index,
        )
        for index in range(1, 4)
    ]
    adaptive = CrossEncoderReranker(
        use_lightweight_scorer=True,
        cross_encoder_gate="adaptive",
    )
    adaptive.rerank("query", list(candidates), top_n=3)
    assert calls == []

    top2 = CrossEncoderReranker(
        use_lightweight_scorer=True,
        cross_encoder_gate="always_top2",
    )
    top2.rerank("query", list(candidates), top_n=3)
    assert len(calls) == 1
    assert len(calls[0]) == 2

    forced = CrossEncoderReranker(
        use_lightweight_scorer=True,
        force_cross_encoder=True,
    )
    forced.rerank("query", list(candidates), top_n=3)
    assert len(calls) == 2
    assert len(calls[1]) == 3


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
