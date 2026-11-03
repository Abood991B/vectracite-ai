"""Score the labelled retrieval-quality set against the real answer pipeline.

The set in ``data/retrieval_quality_cases.json`` pins two known gaps: vague attribute
questions (headcount) and table/statement selection for comparison questions. This script
measures them at the two layers that matter:

* retrieval: the rank of the gold page in the evidence the synthesizer actually receives;
* answer: whether the rendered answer cites the gold page and contains the gold value.

Both layers run through production code (``AnswerPipeline._select_evidence`` and
``AnswerService.execute``), so a gate or selection change is scored end to end. Run two
gates in one process to get paired per-query deltas with a bootstrap interval.

    python scripts/score_retrieval_quality.py --gates adaptive adaptive_top2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from qdrant_client import QdrantClient

from evaluation.domain_benchmark import DomainCase, load_benchmark_chunks, load_domain_benchmark_spec
from src.api.answer_service import AnswerService
from src.cache_and_guardrails.guardrails import EnterpriseGuardrails
from src.cache_and_guardrails.numeric_verifier import DeterministicNumericVerifier
from src.cache_and_guardrails.semantic_cache import SemanticCache
from src.config import settings
from src.inference.llm_client import UnifiedLLMClient
from src.inference.schemas import QueryRequest
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.usecases.query_normalization import correct_query_terms
from src.usecases.scoping import infer_scope_from_question
from src.usecases.synthesis import expand_query

DEFAULT_CASES = "data/retrieval_quality_cases.json"

warnings.filterwarnings("ignore", message="Payload indexes have no effect in the local Qdrant")


@dataclass
class CaseScore:
    case: DomainCase
    rank: int | None = None
    cited: bool = False
    value_found: bool = False
    cited_pages: List[str] = field(default_factory=list)
    answer: str = ""
    latency_ms: float = 0.0


def _rank_of(evidence: Sequence[Any], case: DomainCase) -> int | None:
    for rank, chunk in enumerate(evidence, start=1):
        if chunk.doc_name == case.doc_name and chunk.page_number in case.evidence_pages:
            return rank
    return None


def _build_service(
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> AnswerService:
    return AnswerService(
        retriever=retriever,
        reranker=reranker,
        semantic_cache=SemanticCache(),
        guardrails=EnterpriseGuardrails(),
        numeric_verifier=DeterministicNumericVerifier(),
        llm_client=UnifiedLLMClient(remote_enabled=False),
        ensure_indexed=lambda: None,
        normalize_scope=lambda doc_name: doc_name,
        infer_scope=infer_scope_from_question,
    )


async def _score_gate(
    gate: str,
    cases: Sequence[DomainCase],
    retriever: HybridRetriever,
    top_k: int,
) -> List[CaseScore]:
    reranker = CrossEncoderReranker(cross_encoder_gate=gate)
    service = _build_service(retriever, reranker)
    indexed_docs = retriever.indexed_document_names()
    vocabulary = retriever.corpus_vocabulary()

    scores: List[CaseScore] = []
    for case in cases:
        started_at = time.perf_counter()
        normalized = correct_query_terms(case.question, vocabulary)
        _, evidence = service._select_evidence(
            question=normalized,
            retrieval_query=expand_query(normalized),
            payload=QueryRequest(question=case.question, top_k=top_k),
            scope=None,
            indexed_docs=indexed_docs,
        )
        response = await service.execute(
            QueryRequest(question=case.question, top_k=top_k, use_cache=False)
        )
        score = CaseScore(
            case=case,
            rank=_rank_of(evidence, case),
            cited=any(
                citation.doc_name == case.doc_name
                and citation.page_number in case.evidence_pages
                for citation in response.citations
            ),
            value_found=bool(case.gold_keywords)
            and all(keyword in response.answer for keyword in case.gold_keywords),
            cited_pages=[
                f"{citation.doc_name} p{citation.page_number}"
                for citation in response.citations
            ],
            answer=" ".join(response.answer.split())[:220],
            latency_ms=round((time.perf_counter() - started_at) * 1000.0, 1),
        )
        scores.append(score)
    return scores


def _family_summary(scores: Sequence[CaseScore]) -> Dict[str, Any]:
    ranks = np.asarray(
        [float(score.rank or 0) for score in scores], dtype=np.float64
    )
    reciprocal = np.asarray(
        [0.0 if rank <= 0 else 1.0 / rank for rank in ranks], dtype=np.float64
    )
    return {
        "cases": len(scores),
        "recall_at_1": round(float(np.mean(ranks == 1)), 4),
        "recall_at_3": round(float(np.mean((ranks >= 1) & (ranks <= 3))), 4),
        "mrr": round(float(np.mean(reciprocal)), 4),
        "cited_hit": round(float(np.mean([score.cited for score in scores])), 4),
        "value_hit": round(float(np.mean([score.value_found for score in scores])), 4),
    }


def _per_family(scores: Sequence[CaseScore]) -> Dict[str, Any]:
    families: Dict[str, List[CaseScore]] = {}
    for score in scores:
        families.setdefault(score.case.domain, []).append(score)
    return {family: _family_summary(rows) for family, rows in sorted(families.items())}


def _metric_vectors(scores: Sequence[CaseScore]) -> Dict[str, np.ndarray]:
    ranks = np.asarray([float(score.rank or 0) for score in scores], dtype=np.float64)
    return {
        "mrr": np.asarray(
            [0.0 if rank <= 0 else 1.0 / rank for rank in ranks], dtype=np.float64
        ),
        "recall_at_3": ((ranks >= 1) & (ranks <= 3)).astype(np.float64),
        "cited_hit": np.asarray([float(score.cited) for score in scores], dtype=np.float64),
        "value_hit": np.asarray(
            [float(score.value_found) for score in scores], dtype=np.float64
        ),
    }


def _paired_delta(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    differences = left - right
    observed = float(np.mean(differences))
    if samples <= 0 or left.size < 2:
        return {"delta": round(observed, 4), "ci_low": round(observed, 4), "ci_high": round(observed, 4)}
    indices = rng.integers(0, left.size, size=(samples, left.size))
    bootstrap = np.asarray(
        [float(np.mean(differences[index])) for index in indices], dtype=np.float64
    )
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "delta": round(observed, 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
        "wins": int(np.sum(differences > 1e-12)),
        "losses": int(np.sum(differences < -1e-12)),
        "ties": int(left.size - np.sum(np.abs(differences) > 1e-12)),
    }


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    spec = load_domain_benchmark_spec(Path(args.cases))
    chunks = load_benchmark_chunks(spec.corpus_dirs)
    client = QdrantClient(":memory:")
    retriever = HybridRetriever(
        collection_name="retrieval_quality_score",
        in_memory=True,
        use_lightweight_embedder=args.lightweight,
        client=client,
    )
    retriever.index_chunks(chunks)
    reranker = CrossEncoderReranker(cross_encoder_gate=args.gates[0], use_lightweight_scorer=args.lightweight)
    warmup = retriever.retrieve_hybrid("warmup query", top_k=3)
    if warmup:
        reranker.rerank("warmup query", warmup, top_n=1)

    order = {case.case_id: index for index, case in enumerate(spec.cases)}
    report: Dict[str, Any] = {
        "cases_path": args.cases,
        "top_k": args.top_k,
        "chunks": len(chunks),
        "documents": retriever.indexed_document_names(),
        "gates": {},
    }
    per_gate: Dict[str, List[CaseScore]] = {}
    for gate in args.gates:
        scores = await _score_gate(gate, spec.cases, retriever, args.top_k)
        scores.sort(key=lambda score: order[score.case.case_id])
        per_gate[gate] = scores
        report["gates"][gate] = {
            "overall": _family_summary(scores),
            "families": _per_family(scores),
            "per_case": [
                {
                    "case_id": score.case.case_id,
                    "family": score.case.domain,
                    "question": score.case.question,
                    "gold": f"{score.case.doc_name} p{list(score.case.evidence_pages)}",
                    "retrieval_rank": score.rank,
                    "cited": score.cited,
                    "value_found": score.value_found,
                    "cited_pages": score.cited_pages,
                    "answer": score.answer,
                }
                for score in scores
            ],
        }

    if len(args.gates) > 1:
        rng = np.random.default_rng(args.seed)
        baseline = per_gate[args.gates[0]]
        baseline_vectors = _metric_vectors(baseline)
        comparisons: Dict[str, Any] = {}
        for gate in args.gates[1:]:
            candidate_vectors = _metric_vectors(per_gate[gate])
            comparisons[f"{gate}_vs_{args.gates[0]}"] = {
                metric: _paired_delta(
                    candidate_vectors[metric],
                    baseline_vectors[metric],
                    samples=args.bootstrap_samples,
                    rng=rng,
                )
                for metric in candidate_vectors
            }
        report["paired"] = comparisons

    client.close()
    return report


def _print_report(report: Dict[str, Any]) -> None:
    for gate, block in report["gates"].items():
        print(f"\n=== gate: {gate} ===")
        print(
            f"{'family':<24}{'n':>3}{'R@1':>7}{'R@3':>7}{'MRR':>7}{'cited':>8}{'value':>8}"
        )
        for family, row in block["families"].items():
            print(
                f"{family:<24}{row['cases']:>3}{row['recall_at_1']:>7.3f}"
                f"{row['recall_at_3']:>7.3f}{row['mrr']:>7.3f}"
                f"{row['cited_hit']:>8.3f}{row['value_hit']:>8.3f}"
            )
        overall = block["overall"]
        print(
            f"{'ALL':<24}{overall['cases']:>3}{overall['recall_at_1']:>7.3f}"
            f"{overall['recall_at_3']:>7.3f}{overall['mrr']:>7.3f}"
            f"{overall['cited_hit']:>8.3f}{overall['value_hit']:>8.3f}"
        )
        for row in block["per_case"]:
            rank = row["retrieval_rank"] if row["retrieval_rank"] is not None else "-"
            print(
                f"    {row['case_id']:<32} rank={rank!s:<4}"
                f"cited={'Y' if row['cited'] else 'N':<2}"
                f"value={'Y' if row['value_found'] else 'N'}"
            )

    for name, metrics in report.get("paired", {}).items():
        print(f"\n=== paired: {name} ===")
        for metric, values in metrics.items():
            print(
                f"    {metric:<12} delta={values['delta']:+.3f} "
                f"CI95=[{values['ci_low']:+.3f}, {values['ci_high']:+.3f}] "
                f"w/l/t={values.get('wins', '-')}/{values.get('losses', '-')}/{values.get('ties', '-')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--gates", nargs="+", default=["adaptive"])
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    started_at = time.perf_counter()
    report = asyncio.run(_run(args))
    report["elapsed_s"] = round(time.perf_counter() - started_at, 1)
    _print_report(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json} in {report['elapsed_s']}s")
    print(f"\nsettings: embedding={settings.EMBEDDING_MODEL} reranker={settings.RERANKER_MODEL}")


if __name__ == "__main__":
    main()
