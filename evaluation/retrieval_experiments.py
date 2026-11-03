"""Controlled retrieval experiments over the multi-domain benchmark suites."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from evaluation.domain_benchmark import (
    MODES,
    PAGE_FUSION_MODES,
    DomainBenchmarkSpec,
    DomainCase,
    _paired_bootstrap,
    load_benchmark_chunks,
    load_domain_benchmark_spec,
    load_ragbench_benchmark,
    run_domain_benchmark,
)
from src.config import settings
from src.ingestion.pdf_parser import DocumentChunk
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    clear_embedder_cache,
    get_embedder,
)
from src.retrieval.reranker import CrossEncoderReranker

CACHE_DIR = settings.DATA_DIR / "retrieval_experiment_cache"
DEFAULT_OUTPUT = "data/retrieval_experiments_report.json"

MODEL_CANDIDATES: Dict[str, Tuple[str, int]] = {
    "bge-small": ("BAAI/bge-small-en-v1.5", 384),
    "bge-base": ("BAAI/bge-base-en-v1.5", 768),
    "e5-base": ("intfloat/e5-base-v2", 768),
    "gte-base": ("thenlper/gte-base", 768),
    "bge-large": ("BAAI/bge-large-en-v1.5", 1024),
}


@dataclass(frozen=True)
class RetrievalVariant:
    name: str
    embedding_model: str
    embedding_dim: int
    dense_weight: float
    bm25_weight: float
    candidate_multiplier: int
    force_cross_encoder: bool
    modes: Tuple[str, ...]
    cross_encoder_gate: str = "adaptive"
    confidence_threshold: float = 0.5
    reranker_model: str = settings.RERANKER_MODEL

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _cache_path(suite: str, model_name: str, dimension: int) -> Path:
    return CACHE_DIR / f"{suite}__{_slug(model_name)}__d{dimension}.npz"


def _load_vectors(
    *,
    suite: str,
    chunks: Sequence[DocumentChunk],
    model_name: str,
    dimension: int,
    rebuild: bool,
) -> np.ndarray:
    path = _cache_path(suite, model_name, dimension)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if path.exists() and not rebuild:
        try:
            with np.load(path, allow_pickle=False) as payload:
                cached_model = str(payload["model"][0])
                cached_dim = int(payload["dimension"][0])
                cached_ids = [str(value) for value in payload["chunk_ids"].tolist()]
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
        except (OSError, KeyError, ValueError):
            vectors = np.empty((0, 0), dtype=np.float32)
        else:
            if (
                cached_model == model_name
                and cached_dim == dimension
                and cached_ids == chunk_ids
                and vectors.shape == (len(chunks), dimension)
            ):
                return vectors

    embedder = get_embedder(model_name, dimension, use_lightweight=False)
    started_at = time.perf_counter()
    vectors = np.asarray(
        embedder.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        with open(temp_path, "wb") as handle:
            np.savez_compressed(
                handle,
                model=np.asarray([model_name]),
                dimension=np.asarray([dimension], dtype=np.int64),
                chunk_ids=np.asarray(chunk_ids),
                vectors=vectors,
            )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    print(
        f"embedded {suite} with {model_name} ({len(chunks)} chunks) in "
        f"{time.perf_counter() - started_at:.2f}s"
    )
    return vectors


def _compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        key: report[key]
        for key in (
            "case_count",
            "domains",
            "top_k",
            "bootstrap_samples",
            "seed",
            "modes",
            "signature",
            "retrieval_config",
            "summary",
            "per_domain",
            "pairwise",
        )
    }
    compact["per_query_ranks"] = [
        {
            "case_id": item["case_id"],
            "domain": item["domain"],
            "ranks": item["ranks"],
        }
        for item in report["per_query"]
    ]
    return compact


def _metric_arrays_from_ranks(
    per_query: Sequence[Dict[str, Any]],
    mode: str,
) -> Dict[str, np.ndarray]:
    ranks = np.asarray(
        [float(item["ranks"].get(mode) or 0) for item in per_query],
        dtype=np.float64,
    )
    return {
        "precision_at_1": (ranks == 1).astype(np.float64),
        "recall_at_3": ((ranks >= 1) & (ranks <= 3)).astype(np.float64),
        "mrr": np.asarray(
            [0.0 if rank <= 0 else 1.0 / rank for rank in ranks],
            dtype=np.float64,
        ),
        "ndcg_at_3": np.asarray(
            [
                0.0 if rank <= 0 or rank > 3 else 1.0 / math.log2(rank + 1)
                for rank in ranks
            ],
            dtype=np.float64,
        ),
    }


def _compare_against_baseline(
    results: Dict[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> None:
    baseline_suites = results.get("baseline", {}).get("suites", {})
    for variant_name, payload in results.items():
        if variant_name == "baseline":
            continue
        for suite_name, suite_report in payload["suites"].items():
            baseline_report = baseline_suites.get(suite_name)
            if not baseline_report:
                continue
            comparison: Dict[str, Any] = {}
            for mode in suite_report["summary"]:
                if mode not in baseline_report["summary"]:
                    continue
                left = _metric_arrays_from_ranks(
                    suite_report["per_query_ranks"], mode
                )
                right = _metric_arrays_from_ranks(
                    baseline_report["per_query_ranks"], mode
                )
                comparison[mode] = {
                    metric: _paired_bootstrap(
                        left[metric],
                        right[metric],
                        metric,
                        samples=bootstrap_samples,
                        rng=np.random.default_rng(seed),
                    )
                    for metric in left
                }
            suite_report["comparisons_vs_baseline"] = comparison


def _run_variant(
    *,
    variant: RetrievalVariant,
    suite: str,
    chunks: Sequence[DocumentChunk],
    cases: Sequence[DomainCase],
    spec: DomainBenchmarkSpec,
    corpus_dirs: Sequence[Path],
    vectors: np.ndarray,
    top_k: int,
    bootstrap_samples: int,
    seed: int,
    allowed_modes: Sequence[str],
) -> Dict[str, Any]:
    modes = tuple(mode for mode in variant.modes if mode in allowed_modes)
    if not modes:
        raise ValueError(f"variant {variant.name} has no modes for suite {suite}")
    retriever = HybridRetriever(
        in_memory=True,
        use_lightweight_embedder=False,
        embedding_model=variant.embedding_model,
        embedding_dim=variant.embedding_dim,
        dense_weight=variant.dense_weight,
        bm25_weight=variant.bm25_weight,
        candidate_multiplier=variant.candidate_multiplier,
    )
    retriever.index_chunks(list(chunks), vectors=vectors)
    reranker = CrossEncoderReranker(
        model_name=variant.reranker_model,
        use_lightweight_scorer=False,
        force_cross_encoder=variant.force_cross_encoder,
        cross_encoder_gate=variant.cross_encoder_gate,
        confidence_threshold=variant.confidence_threshold,
    )
    report = run_domain_benchmark(
        cases=cases,
        chunks=chunks,
        spec=spec,
        corpus_dirs=corpus_dirs,
        modes=modes,
        top_k=top_k,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        lightweight=False,
        retriever_override=retriever,
        reranker_override=reranker,
    )
    return _compact_report(report)


def _metric_delta(
    summary: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for metric in ("precision_at_1", "recall_at_3", "mrr", "ndcg_at_3"):
        if metric in summary and metric in baseline:
            deltas[metric] = round(
                float(summary[metric]) - float(baseline[metric]), 4
            )
    return deltas


def _variant_groups(
    selected_models: Sequence[str],
) -> List[List[RetrievalVariant]]:
    baseline_model = settings.EMBEDDING_MODEL
    baseline_dim = settings.EMBEDDING_DIM
    baseline = RetrievalVariant(
        name="baseline",
        embedding_model=baseline_model,
        embedding_dim=baseline_dim,
        dense_weight=settings.HYBRID_DENSE_WEIGHT,
        bm25_weight=settings.HYBRID_BM25_WEIGHT,
        candidate_multiplier=4,
        force_cross_encoder=False,
        modes=MODES + PAGE_FUSION_MODES,
    )
    groups: Dict[Tuple[str, int], List[RetrievalVariant]] = {}
    groups.setdefault((baseline_model, baseline_dim), []).append(baseline)
    groups[(baseline_model, baseline_dim)].extend(
        [
            RetrievalVariant(
                name="fusion_70_30",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=0.70,
                bm25_weight=0.30,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_rrf",),
            ),
            RetrievalVariant(
                name="fusion_50_50",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=0.50,
                bm25_weight=0.50,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_rrf",),
            ),
            RetrievalVariant(
                name="fusion_30_70",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=0.30,
                bm25_weight=0.70,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_rrf",),
            ),
            RetrievalVariant(
                name="candidates_2",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=2,
                force_cross_encoder=False,
                modes=("hybrid_rrf",),
            ),
            RetrievalVariant(
                name="candidates_8",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=8,
                force_cross_encoder=False,
                modes=("hybrid_rrf",),
            ),
            RetrievalVariant(
                name="candidates_20",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=20,
                force_cross_encoder=False,
                modes=("hybrid_rrf",),
            ),
            RetrievalVariant(
                name="reranker_adaptive_top2",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_plus_reranker",),
                cross_encoder_gate="adaptive_top2",
            ),
            RetrievalVariant(
                name="reranker_top1",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_plus_reranker",),
                cross_encoder_gate="always_top1",
            ),
            RetrievalVariant(
                name="reranker_top1_then_top2_50",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_plus_reranker",),
                cross_encoder_gate="top1_then_top2",
                confidence_threshold=0.5,
            ),
            RetrievalVariant(
                name="reranker_top1_then_top2_60",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_plus_reranker",),
                cross_encoder_gate="top1_then_top2",
                confidence_threshold=0.6,
            ),
            RetrievalVariant(
                name="reranker_top1_then_top2_65",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_plus_reranker",),
                cross_encoder_gate="top1_then_top2",
                confidence_threshold=0.65,
            ),
            RetrievalVariant(
                name="reranker_top1_then_top2_70",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_plus_reranker",),
                cross_encoder_gate="top1_then_top2",
                confidence_threshold=0.7,
            ),
            RetrievalVariant(
                name="reranker_top2",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_plus_reranker",),
                cross_encoder_gate="always_top2",
            ),
            RetrievalVariant(
                name="forced_reranker",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=True,
                modes=("hybrid_plus_reranker",),
            ),
            RetrievalVariant(
                name="reranker_top2_page_fusion",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_plus_reranker_page_fusion",),
                cross_encoder_gate="always_top2",
            ),
            RetrievalVariant(
                name="forced_reranker_page_fusion",
                embedding_model=baseline_model,
                embedding_dim=baseline_dim,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=True,
                modes=("hybrid_plus_reranker_page_fusion",),
            ),
        ]
    )
    for model_key in selected_models:
        if model_key not in MODEL_CANDIDATES:
            raise ValueError(
                f"unknown model {model_key}; choose from {sorted(MODEL_CANDIDATES)}"
            )
        model_name, dimension = MODEL_CANDIDATES[model_key]
        if (model_name, dimension) == (baseline_model, baseline_dim):
            continue
        groups.setdefault((model_name, dimension), []).append(
            RetrievalVariant(
                name=f"embed_{model_key.replace('-', '_')}",
                embedding_model=model_name,
                embedding_dim=dimension,
                dense_weight=settings.HYBRID_DENSE_WEIGHT,
                bm25_weight=settings.HYBRID_BM25_WEIGHT,
                candidate_multiplier=4,
                force_cross_encoder=False,
                modes=("hybrid_rrf", "hybrid_plus_reranker"),
            )
        )
    return list(groups.values())


def run_experiments(
    *,
    selected_models: Sequence[str],
    suites: Sequence[str],
    top_k: int,
    bootstrap_samples: int,
    seed: int,
    max_cases_per_config: int,
    rebuild_cache: bool,
    output_path: Path | str,
) -> Dict[str, Any]:
    pdf_spec = load_domain_benchmark_spec()
    pdf_chunks = load_benchmark_chunks(pdf_spec.corpus_dirs)
    rag_chunks: List[DocumentChunk] = []
    rag_cases: List[DomainCase] = []
    if "ragbench" in suites:
        rag_chunks, rag_cases = load_ragbench_benchmark(
            max_cases_per_config=max_cases_per_config
        )
    rag_spec = DomainBenchmarkSpec(cases=tuple(rag_cases))

    results: Dict[str, Any] = {}
    for group in _variant_groups(selected_models):
        model_name, dimension = group[0].embedding_model, group[0].embedding_dim
        pdf_vectors = None
        rag_vectors = None
        if "pdf" in suites:
            pdf_vectors = _load_vectors(
                suite="pdf",
                chunks=pdf_chunks,
                model_name=model_name,
                dimension=dimension,
                rebuild=rebuild_cache,
            )
        if "ragbench" in suites:
            rag_vectors = _load_vectors(
                suite="ragbench",
                chunks=rag_chunks,
                model_name=model_name,
                dimension=dimension,
                rebuild=rebuild_cache,
            )
        for variant in group:
            print(f"running {variant.name} ({model_name})")
            variant_results: Dict[str, Any] = {}
            pdf_modes = tuple(
                mode for mode in variant.modes if mode in MODES + PAGE_FUSION_MODES
            )
            rag_modes = tuple(mode for mode in variant.modes if mode in MODES)
            if "pdf" in suites and pdf_modes:
                variant_results["pdf"] = _run_variant(
                    variant=variant,
                    suite="pdf",
                    chunks=pdf_chunks,
                    cases=pdf_spec.cases,
                    spec=pdf_spec,
                    corpus_dirs=pdf_spec.corpus_dirs,
                    vectors=pdf_vectors,
                    top_k=top_k,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                    allowed_modes=pdf_modes,
                )
            if "ragbench" in suites and rag_modes:
                variant_results["ragbench"] = _run_variant(
                    variant=variant,
                    suite="ragbench",
                    chunks=rag_chunks,
                    cases=rag_cases,
                    spec=rag_spec,
                    corpus_dirs=(),
                    vectors=rag_vectors,
                    top_k=top_k,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                    allowed_modes=rag_modes,
                )
            results[variant.name] = {
                "variant": variant.to_dict(),
                "suites": variant_results,
            }
        clear_embedder_cache()

    baseline = results.get("baseline", {}).get("suites", {})
    for variant_name, payload in results.items():
        if variant_name == "baseline":
            continue
        for suite_name, suite_report in payload["suites"].items():
            baseline_report = baseline.get(suite_name)
            if not baseline_report:
                continue
            suite_report["delta_vs_baseline"] = {
                mode: _metric_delta(
                    suite_report["summary"].get(mode, {}),
                    baseline_report["summary"].get(mode, {}),
                )
                for mode in suite_report["summary"]
                if mode in baseline_report["summary"]
            }
    _compare_against_baseline(
        results,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dim": settings.EMBEDDING_DIM,
            "reranker_model": settings.RERANKER_MODEL,
            "top_k": top_k,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        },
        "suites": list(suites),
        "results": results,
    }
    path = Path(output_path)
    if not path.is_absolute():
        path = settings.BASE_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bge-base", "e5-base", "gte-base"],
        choices=sorted(MODEL_CANDIDATES),
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["pdf", "ragbench"],
        choices=["pdf", "ragbench"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases-per-config", type=int, default=12)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_experiments(
        selected_models=args.models,
        suites=args.suites,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        max_cases_per_config=args.max_cases_per_config,
        rebuild_cache=args.rebuild_cache,
        output_path=args.output,
    )
    for name, payload in report["results"].items():
        print(f"\n{name}")
        for suite_name, suite_report in payload["suites"].items():
            for mode, summary in suite_report["summary"].items():
                print(
                    f"  {suite_name:8s} {mode:32s} "
                    f"P@1={summary['precision_at_1']:.4f} "
                    f"R@3={summary['recall_at_3']:.4f} "
                    f"MRR={summary['mrr']:.4f} "
                    f"nDCG@3={summary['ndcg_at_3']:.4f}"
                )


if __name__ == "__main__":
    main()
