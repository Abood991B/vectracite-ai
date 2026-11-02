"""Domain-agnostic retrieval and citation benchmark harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.config import settings
from src.ingestion.corpus_cache import load_corpus_chunks
from src.ingestion.pdf_parser import DocumentChunk, MultimodalFinancialParser
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedDocument
from src.retrieval.reranker import CrossEncoderReranker, get_shared_cross_encoder

DOMAIN_BENCHMARK_VERSION = 1
MODES = ("dense_only", "hybrid_rrf", "hybrid_plus_reranker")
DEFAULT_CORPUS_DIRS = ("data/corpus_pdfs", "data/domain_corpus_pdfs")
DEFAULT_CASES_PATH = "data/domain_benchmark_cases.json"
DEFAULT_OUTPUT_PATH = "data/domain_benchmark_report.json"
RAGBENCH_CONFIGS = ("covidqa", "cuad", "emanual", "finqa", "expertqa")
RAGBENCH_DOMAINS = {
    "covidqa": "biomedical",
    "cuad": "legal",
    "emanual": "technical",
    "finqa": "finance",
    "expertqa": "expert_qa",
    "delucionqa": "enterprise",
    "tatqa": "finance_tables",
}


@dataclass(frozen=True)
class DomainCase:
    """A single retrieval/citation case with explicit evidence identifiers."""

    case_id: str
    domain: str
    question: str
    gold_answer: str = ""
    gold_keywords: Tuple[str, ...] = ()
    doc_name: str | None = None
    evidence_pages: Tuple[int, ...] = ()
    expected_chunk_ids: Tuple[str, ...] = ()
    evidence_kind: str = "page"
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DomainBenchmarkSpec:
    """Cases plus the corpus roots that should be indexed for them."""

    cases: Tuple[DomainCase, ...]
    corpus_dirs: Tuple[Path, ...] = ()
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "corpus_dirs": [str(path) for path in self.corpus_dirs],
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass
class CaseRun:
    """Per-case retrieval output used for metrics and bootstrap resampling."""

    case: DomainCase
    ranks: Dict[str, int | None] = field(default_factory=dict)
    latencies_ms: Dict[str, float] = field(default_factory=dict)
    retrieved: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case.case_id,
            "domain": self.case.domain,
            "doc_name": self.case.doc_name,
            "evidence_pages": list(self.case.evidence_pages),
            "expected_chunk_ids": list(self.case.expected_chunk_ids),
            "evidence_kind": self.case.evidence_kind,
            "question": self.case.question,
            "gold_answer": self.case.gold_answer,
            "source": self.case.source,
            "ranks": self.ranks,
            "latencies_ms": self.latencies_ms,
            "retrieved": self.retrieved,
        }


def _require_text(value: Any, field_name: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{case_id}: {field_name} must be a non-empty string")
    return value.strip()


def _parse_case(raw: Dict[str, Any], index: int) -> DomainCase:
    if not isinstance(raw, dict):
        raise ValueError(f"case {index} must be an object")
    case_id = _require_text(raw.get("case_id"), "case_id", f"case {index}")
    domain = _require_text(raw.get("domain"), "domain", case_id)
    question = _require_text(raw.get("question"), "question", case_id)
    gold_answer = str(raw.get("gold_answer") or "").strip()
    evidence_pages = tuple(
        sorted({int(page) for page in raw.get("evidence_pages") or []})
    )
    expected_chunk_ids = tuple(
        sorted({str(chunk_id) for chunk_id in raw.get("expected_chunk_ids") or []})
    )
    if any(page < 1 for page in evidence_pages):
        raise ValueError(f"{case_id}: evidence_pages must be positive")
    doc_name = raw.get("doc_name")
    if doc_name is not None:
        doc_name = _require_text(doc_name, "doc_name", case_id)
    if not expected_chunk_ids and not (doc_name and evidence_pages):
        raise ValueError(
            f"{case_id}: provide expected_chunk_ids or doc_name plus evidence_pages"
        )
    evidence_kind = str(raw.get("evidence_kind") or "page").strip().lower()
    if evidence_kind not in {"page", "passage"}:
        raise ValueError(f"{case_id}: evidence_kind must be page or passage")
    gold_keywords = tuple(
        str(keyword).strip()
        for keyword in raw.get("gold_keywords") or []
        if str(keyword).strip()
    )
    return DomainCase(
        case_id=case_id,
        domain=domain,
        question=question,
        gold_answer=gold_answer,
        gold_keywords=gold_keywords,
        doc_name=doc_name,
        evidence_pages=evidence_pages,
        expected_chunk_ids=expected_chunk_ids,
        evidence_kind=evidence_kind,
        source=str(raw.get("source") or "").strip(),
    )


def load_domain_benchmark_spec(path: Path | str = DEFAULT_CASES_PATH) -> DomainBenchmarkSpec:
    """Load and validate a page/passage benchmark definition."""
    cases_path = Path(path)
    if not cases_path.is_absolute():
        cases_path = settings.BASE_DIR / cases_path
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("domain benchmark schema_version must be 1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("domain benchmark must contain at least one case")
    cases = tuple(_parse_case(raw, index) for index, raw in enumerate(raw_cases))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("domain benchmark case_id values must be unique")
    corpus_dirs = tuple(
        (settings.BASE_DIR / str(path)).resolve()
        for path in payload.get("corpus_dirs") or DEFAULT_CORPUS_DIRS
    )
    return DomainBenchmarkSpec(
        cases=cases,
        corpus_dirs=corpus_dirs,
        description=str(payload.get("description") or "").strip(),
    )


def _cache_path_for_corpus(corpus_dir: Path) -> Path:
    if corpus_dir.resolve() == settings.CORPUS_PDF_DIR.resolve():
        return settings.CORPUS_INDEX_PATH
    return settings.DATA_DIR / f"{corpus_dir.name}_index.json"


def load_benchmark_chunks(
    corpus_dirs: Iterable[Path],
    *,
    parser: MultimodalFinancialParser | None = None,
    write_cache: bool = False,
) -> List[DocumentChunk]:
    """Load all PDFs under each corpus root, using the checked-in cache when valid."""
    active_parser = parser or MultimodalFinancialParser(
        chunk_size=settings.PARSER_CHUNK_SIZE,
        chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
    )
    chunks: List[DocumentChunk] = []
    seen_documents: Dict[str, Path] = {}
    for corpus_dir in corpus_dirs:
        if not corpus_dir.exists():
            raise FileNotFoundError(f"benchmark corpus directory not found: {corpus_dir}")
        loaded = load_corpus_chunks(
            corpus_dir,
            active_parser,
            cache_path=_cache_path_for_corpus(corpus_dir),
            write=write_cache,
        )
        for chunk in loaded:
            previous = seen_documents.get(chunk.doc_name)
            if previous is not None and previous != corpus_dir:
                raise ValueError(
                    f"duplicate document name {chunk.doc_name} in {previous} and {corpus_dir}"
                )
            seen_documents[chunk.doc_name] = corpus_dir
        chunks.extend(loaded)
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.doc_name,
            chunk.page_number,
            int(chunk.metadata.get("chunk_index", 0)),
            chunk.chunk_id,
        ),
    )


def _is_hit(result: RetrievedDocument, case: DomainCase) -> bool:
    if case.expected_chunk_ids:
        return result.chunk_id in case.expected_chunk_ids
    if not case.doc_name or not case.evidence_pages:
        return False
    return (
        result.doc_name == case.doc_name
        and result.page_number in case.evidence_pages
    )


def _rank_for(results: Sequence[RetrievedDocument], case: DomainCase) -> int | None:
    for rank, result in enumerate(results, start=1):
        if _is_hit(result, case):
            return rank
    return None


def _retrieval_result_payload(result: RetrievedDocument) -> Dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "doc_name": result.doc_name,
        "page_number": result.page_number,
        "contains_table": result.contains_table,
        "dense_score": round(float(result.dense_score), 6),
        "bm25_score": round(float(result.bm25_score), 6),
        "rrf_score": round(float(result.rrf_score), 6),
        "rerank_score": (
            None if result.rerank_score is None else round(float(result.rerank_score), 6)
        ),
    }


def _run_mode(
    *,
    mode: str,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    case: DomainCase,
    top_k: int,
) -> List[RetrievedDocument]:
    if mode == "dense_only":
        return retriever.retrieve_dense_only(case.question, top_k=top_k)
    if mode == "hybrid_rrf":
        return retriever.retrieve_hybrid(case.question, top_k=top_k)
    if mode == "hybrid_plus_reranker":
        candidates = retriever.retrieve_hybrid(
            case.question, top_k=max(top_k * 2, 6)
        )
        return reranker.rerank(case.question, candidates, top_n=top_k)
    raise ValueError(f"unsupported benchmark mode: {mode}")


def _warmup(
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    *,
    top_k: int,
) -> None:
    candidates = retriever.retrieve_hybrid("warmup query", top_k=top_k)
    if candidates:
        reranker.rerank("warmup query", candidates, top_n=1)
    encoder = get_shared_cross_encoder(
        settings.RERANKER_MODEL,
        use_lightweight=retriever.use_lightweight_embedder,
    )
    encoder.predict(
        [
            (
                "warmup query",
                "| metric | 2022 | 2021 |\n| --- | --- | --- |\n| revenue | 1 | 2 |",
            )
        ],
        show_progress_bar=False,
    )


def _case_metric_values(runs: Sequence[CaseRun], mode: str) -> Dict[str, np.ndarray]:
    ranks = np.asarray(
        [
            float(run.ranks.get(mode) or 0)
            for run in runs
        ],
        dtype=np.float64,
    )
    hit_at_1 = (ranks == 1).astype(np.float64)
    hit_at_3 = ((ranks >= 1) & (ranks <= 3)).astype(np.float64)
    reciprocal_rank = np.asarray(
        [0.0 if rank <= 0 else 1.0 / rank for rank in ranks],
        dtype=np.float64,
    )
    ndcg_at_3 = np.asarray(
        [
            0.0 if rank <= 0 or rank > 3 else 1.0 / math.log2(rank + 1)
            for rank in ranks
        ],
        dtype=np.float64,
    )
    return {
        "precision_at_1": hit_at_1,
        "recall_at_3": hit_at_3,
        "mrr": reciprocal_rank,
        "ndcg_at_3": ndcg_at_3,
    }


def _metric_scalar(values: np.ndarray, metric: str) -> float:
    if values.size == 0:
        return 0.0
    if metric == "precision_at_1":
        return float(np.mean(values == 1.0))
    return float(np.mean(values))


def _bootstrap_ci(
    values: np.ndarray,
    metric: str,
    *,
    samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    if values.size == 0 or samples <= 0:
        return 0.0, 0.0
    if values.size == 1:
        scalar = _metric_scalar(values, metric)
        return scalar, scalar
    indices = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap = np.asarray(
        [
            float(np.mean(values[sample] == 1.0))
            if metric == "precision_at_1"
            else float(np.mean(values[sample]))
            for sample in indices
        ],
        dtype=np.float64,
    )
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def _paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    metric: str,
    *,
    samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    if left.size == 0 or left.size != right.size:
        raise ValueError("paired bootstrap requires equally sized non-empty arrays")
    differences = left - right
    observed = _metric_scalar(left, metric) - _metric_scalar(right, metric)
    tolerance = 1e-12
    wins = int(np.sum(differences > tolerance))
    losses = int(np.sum(differences < -tolerance))
    ties = int(left.size - wins - losses)
    if samples <= 0 or left.size == 1:
        low, high = observed, observed
    else:
        indices = rng.integers(0, left.size, size=(samples, left.size))
        bootstrap = np.asarray(
            [
                _metric_scalar(left[sample], metric)
                - _metric_scalar(right[sample], metric)
                for sample in indices
            ],
            dtype=np.float64,
        )
        low, high = np.percentile(bootstrap, [2.5, 97.5])
    p_low = float(np.mean(differences <= tolerance))
    p_high = float(np.mean(differences >= -tolerance))
    return {
        "delta": round(float(observed), 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
        "p_value": round(min(1.0, 2.0 * min(p_low, p_high)), 4),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def _summary_for_mode(
    runs: Sequence[CaseRun],
    mode: str,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    values = _case_metric_values(runs, mode)
    summary: Dict[str, Any] = {
        metric: round(_metric_scalar(values[metric], metric), 4)
        for metric in ("precision_at_1", "recall_at_3", "mrr", "ndcg_at_3")
    }
    latencies = [run.latencies_ms.get(mode, 0.0) for run in runs]
    summary["avg_latency_ms"] = round(float(np.mean(latencies)), 2) if latencies else 0.0
    summary["p95_latency_ms"] = (
        round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    )
    summary["queries_evaluated"] = len(runs)
    summary["confidence_intervals"] = {
        metric: _bootstrap_ci(
            values[metric],
            metric,
            samples=bootstrap_samples,
            rng=rng,
        )
        for metric in ("precision_at_1", "recall_at_3", "mrr", "ndcg_at_3")
    }
    return summary


def _per_domain_summary(
    runs: Sequence[CaseRun],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    domains = sorted({run.case.domain for run in runs})
    output: Dict[str, Any] = {}
    for domain in domains:
        domain_runs = [run for run in runs if run.case.domain == domain]
        output[domain] = {
            mode: _summary_for_mode(
                domain_runs,
                mode,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )
            for mode in MODES
        }
    return output


def _pairwise_summary(
    runs: Sequence[CaseRun],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    comparisons = (
        ("hybrid_rrf_vs_dense_only", "hybrid_rrf", "dense_only"),
        ("hybrid_plus_reranker_vs_hybrid_rrf", "hybrid_plus_reranker", "hybrid_rrf"),
        (
            "hybrid_plus_reranker_vs_dense_only",
            "hybrid_plus_reranker",
            "dense_only",
        ),
    )
    output: Dict[str, Any] = {}
    for name, left_mode, right_mode in comparisons:
        left_values = _case_metric_values(runs, left_mode)
        right_values = _case_metric_values(runs, right_mode)
        output[name] = {
            metric: _paired_bootstrap(
                left_values[metric],
                right_values[metric],
                metric,
                samples=bootstrap_samples,
                rng=rng,
            )
            for metric in ("precision_at_1", "recall_at_3", "mrr", "ndcg_at_3")
        }
    return output


def _corpus_files(corpus_dirs: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for corpus_dir in corpus_dirs:
        files.extend(sorted(corpus_dir.glob("*.pdf")))
    return files


def compute_domain_signature(
    chunks: Sequence[DocumentChunk],
    cases: Sequence[DomainCase],
    corpus_dirs: Iterable[Path],
) -> str:
    """Fingerprint the exact corpus, models, parser settings, and cases."""
    digest = hashlib.sha256()
    digest.update(str(DOMAIN_BENCHMARK_VERSION).encode("utf-8"))
    digest.update(b"\0")
    for value in (
        settings.EMBEDDING_MODEL,
        settings.RERANKER_MODEL,
        str(settings.EMBEDDING_DIM),
        str(settings.PARSER_CHUNK_SIZE),
        str(settings.PARSER_CHUNK_OVERLAP),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for path in _corpus_files(corpus_dirs):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("utf-8"))
        digest.update(b"\0")
    for case in sorted(cases, key=lambda item: item.case_id):
        digest.update(
            json.dumps(case.to_dict(), sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        )
        digest.update(b"\0")
    for chunk in sorted(
        chunks,
        key=lambda item: (item.doc_name, item.page_number, item.chunk_id),
    ):
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _corpus_summary(chunks: Sequence[DocumentChunk], corpus_dirs: Iterable[Path]) -> Dict[str, Any]:
    documents = sorted({chunk.doc_name for chunk in chunks})
    return {
        "corpus_dirs": [str(path) for path in corpus_dirs],
        "documents": len(documents),
        "document_names": documents,
        "chunks": len(chunks),
        "table_chunks": sum(1 for chunk in chunks if chunk.contains_table),
    }


def run_domain_benchmark(
    *,
    cases: Sequence[DomainCase] | None = None,
    chunks: Sequence[DocumentChunk] | None = None,
    spec: DomainBenchmarkSpec | None = None,
    corpus_dirs: Iterable[Path] | None = None,
    top_k: int = 5,
    bootstrap_samples: int = 2000,
    seed: int = 42,
    lightweight: bool = False,
    output_path: Path | str | None = None,
) -> Dict[str, Any]:
    """Run dense, hybrid, and reranked retrieval over a domain benchmark."""
    active_spec = spec or load_domain_benchmark_spec()
    active_cases = tuple(cases if cases is not None else active_spec.cases)
    active_dirs = tuple(corpus_dirs if corpus_dirs is not None else active_spec.corpus_dirs)
    if not active_cases:
        raise ValueError("benchmark requires at least one case")
    case_ids = [case.case_id for case in active_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark case_id values must be unique")

    active_chunks = (
        list(chunks)
        if chunks is not None
        else load_benchmark_chunks(active_dirs)
    )
    if not active_chunks:
        raise ValueError("benchmark corpus produced no chunks")
    available_chunk_ids = {chunk.chunk_id for chunk in active_chunks}
    available_pages = {
        (chunk.doc_name, chunk.page_number) for chunk in active_chunks
    }
    missing_chunks = sorted(
        {
            chunk_id
            for case in active_cases
            for chunk_id in case.expected_chunk_ids
            if chunk_id not in available_chunk_ids
        }
    )
    missing_pages = sorted(
        {
            (case.doc_name, page)
            for case in active_cases
            if not case.expected_chunk_ids
            for page in case.evidence_pages
            if (case.doc_name, page) not in available_pages
        }
    )
    if missing_chunks or missing_pages:
        raise ValueError(
            "benchmark evidence is missing from the indexed corpus: "
            f"chunks={missing_chunks[:5]} pages={missing_pages[:5]}"
        )

    retriever = HybridRetriever(
        in_memory=True,
        use_lightweight_embedder=lightweight,
    )
    retriever.index_chunks(active_chunks, use_embedding_cache=False)
    reranker = CrossEncoderReranker(use_lightweight_scorer=lightweight)
    _warmup(retriever, reranker, top_k=top_k)

    rng = np.random.default_rng(seed)
    runs: List[CaseRun] = []
    for case in active_cases:
        run = CaseRun(case=case)
        for mode in MODES:
            started_at = time.perf_counter()
            results = _run_mode(
                mode=mode,
                retriever=retriever,
                reranker=reranker,
                case=case,
                top_k=top_k,
            )
            run.latencies_ms[mode] = round(
                (time.perf_counter() - started_at) * 1000.0, 3
            )
            run.ranks[mode] = _rank_for(results, case)
            run.retrieved[mode] = [
                _retrieval_result_payload(result) for result in results
            ]
        runs.append(run)

    summary = {
        mode: _summary_for_mode(
            runs,
            mode,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        for mode in MODES
    }
    report: Dict[str, Any] = {
        "schema_version": 1,
        "benchmark_version": DOMAIN_BENCHMARK_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "signature": compute_domain_signature(active_chunks, active_cases, active_dirs),
        "corpus": _corpus_summary(active_chunks, active_dirs),
        "case_count": len(active_cases),
        "domains": sorted({case.domain for case in active_cases}),
        "evidence_kinds": sorted({case.evidence_kind for case in active_cases}),
        "top_k": top_k,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "summary": summary,
        "per_domain": _per_domain_summary(
            runs,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        ),
        "pairwise": _pairwise_summary(
            runs,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        ),
        "per_query": [run.to_dict() for run in runs],
    }
    if output_path is not None:
        path = Path(output_path)
        if not path.is_absolute():
            path = settings.BASE_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report


def _ragbench_relevant_doc_indices(row: Dict[str, Any], document_count: int) -> set[int]:
    keys: List[str] = []
    for field_name in ("all_relevant_sentence_keys", "all_utilized_sentence_keys"):
        values = row.get(field_name) or []
        if isinstance(values, list):
            keys.extend(str(value) for value in values)
    for item in row.get("sentence_support_information") or []:
        if not isinstance(item, dict):
            continue
        for field_name in ("supporting_sentence_keys",):
            values = item.get(field_name) or []
            if isinstance(values, list):
                keys.extend(str(value) for value in values)
    indices: set[int] = set()
    for key in keys:
        match = re.match(r"^(\d+)", key)
        if not match:
            continue
        index = int(match.group(1))
        if 0 <= index < document_count:
            indices.add(index)
    return indices


def _stable_chunk_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def _first_available_split(config: str) -> str:
    from datasets import get_dataset_split_names

    splits = set(get_dataset_split_names("galileo-ai/ragbench", config))
    for split in ("test", "validation", "train"):
        if split in splits:
            return split
    raise RuntimeError(f"RAGBench config {config} has no usable split: {sorted(splits)}")


def load_ragbench_benchmark(
    *,
    configs: Sequence[str] = RAGBENCH_CONFIGS,
    max_cases_per_config: int = 12,
    domain_map: Dict[str, str] | None = None,
) -> Tuple[List[DocumentChunk], List[DomainCase]]:
    """Load a streaming RAGBench subset as passage-level retrieval cases."""
    from datasets import load_dataset

    active_domains = domain_map or RAGBENCH_DOMAINS
    chunks_by_id: Dict[str, DocumentChunk] = {}
    cases: List[DomainCase] = []
    for config in configs:
        split = _first_available_split(config)
        dataset = load_dataset(
            "galileo-ai/ragbench",
            config,
            split=split,
            streaming=True,
        )
        accepted = 0
        for row in dataset:
            if accepted >= max_cases_per_config:
                break
            documents = row.get("documents") or []
            if not isinstance(documents, list) or not documents:
                continue
            question = str(row.get("question") or "").strip()
            if not question:
                continue
            relevant_indices = _ragbench_relevant_doc_indices(row, len(documents))
            if not relevant_indices:
                continue
            case_chunk_ids: List[str] = []
            case_pages: List[int] = []
            for index, document in enumerate(documents):
                text = str(document or "").strip()
                if not text:
                    continue
                chunk_id = _stable_chunk_id("ragbench", config, text)
                if chunk_id not in chunks_by_id:
                    chunks_by_id[chunk_id] = DocumentChunk(
                        chunk_id=chunk_id,
                        doc_name=f"RAGBench_{config}",
                        page_number=index + 1,
                        text=text,
                        contains_table=False,
                        metadata={
                            "source": "RAGBench",
                            "config": config,
                            "passage_index": index,
                        },
                    )
                if index in relevant_indices:
                    case_chunk_ids.append(chunk_id)
                    case_pages.append(index + 1)
            if not case_chunk_ids:
                continue
            row_id = str(row.get("id") or f"case_{accepted}")
            case_id = f"ragbench::{config}::{row_id}"
            cases.append(
                DomainCase(
                    case_id=case_id,
                    domain=active_domains.get(config, config),
                    question=question,
                    gold_answer=str(row.get("response") or "").strip(),
                    doc_name=f"RAGBench_{config}",
                    evidence_pages=tuple(sorted(set(case_pages))),
                    expected_chunk_ids=tuple(sorted(set(case_chunk_ids))),
                    evidence_kind="passage",
                    source=f"RAGBench:{config}:{split}",
                )
            )
            accepted += 1
    if not cases:
        raise RuntimeError("RAGBench adapter produced no benchmark cases")
    return sorted(
        chunks_by_id.values(),
        key=lambda chunk: (chunk.doc_name, chunk.page_number, chunk.chunk_id),
    ), cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--source", choices=("pdf", "ragbench"), default="pdf")
    parser.add_argument("--ragbench-configs", nargs="+", default=list(RAGBENCH_CONFIGS))
    parser.add_argument("--max-cases-per-config", type=int, default=12)
    args = parser.parse_args()

    if args.source == "ragbench":
        chunks, cases = load_ragbench_benchmark(
            configs=args.ragbench_configs,
            max_cases_per_config=args.max_cases_per_config,
        )
        spec = DomainBenchmarkSpec(cases=tuple(cases))
        report = run_domain_benchmark(
            cases=cases,
            chunks=chunks,
            spec=spec,
            corpus_dirs=(),
            top_k=args.top_k,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            lightweight=args.lightweight,
            output_path=args.output,
        )
    else:
        spec = load_domain_benchmark_spec(args.cases)
        report = run_domain_benchmark(
            spec=spec,
            top_k=args.top_k,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            lightweight=args.lightweight,
            output_path=args.output,
        )
    print(json.dumps({"summary": report["summary"], "signature": report["signature"]}, indent=2))


if __name__ == "__main__":
    main()
