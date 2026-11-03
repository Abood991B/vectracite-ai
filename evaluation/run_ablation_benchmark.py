"""Data-driven multi-domain retrieval and adaptive SCAR ablation benchmark.

The evaluator reads the same real PDF corpus used by the API, loads checked-in
benchmark questions, and adds schema-derived cases for new table-bearing PDFs.
"""

import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:
    import mlflow
except ModuleNotFoundError:  # Optional for local evaluation and CI environments.
    mlflow = None

from src.config import settings
from src.ingestion.corpus_cache import load_corpus_chunks
from src.ingestion.pdf_parser import DocumentChunk, MultimodalFinancialParser
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.table_ast import TableASTParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_corpus_chunks_from_pdfs(
    parser: MultimodalFinancialParser | None = None,
    *,
    skip_errors: bool = False,
) -> List[DocumentChunk]:
    """Parse every real PDF stored in the configured corpus directory."""
    active_parser = parser or MultimodalFinancialParser(
        chunk_size=settings.PARSER_CHUNK_SIZE,
        chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
    )
    settings.CORPUS_PDF_DIR.mkdir(parents=True, exist_ok=True)
    return load_corpus_chunks(
        settings.CORPUS_PDF_DIR, active_parser, skip_errors=skip_errors
    )


EVALUATION_VERSION = 1


def compute_evaluation_signature(chunks: List[DocumentChunk]) -> str:
    """Fingerprint the indexed content so stale persisted reports are never served."""
    digest = hashlib.sha256()
    digest.update(str(EVALUATION_VERSION).encode("utf-8"))
    digest.update(b"\0")
    digest.update(settings.EMBEDDING_MODEL.encode("utf-8"))
    digest.update(b"\0")
    digest.update(settings.RERANKER_MODEL.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(settings.PARSER_CHUNK_SIZE).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(settings.PARSER_CHUNK_OVERLAP).encode("utf-8"))
    digest.update(b"\0")
    queries_path = settings.DATA_DIR / "benchmark_queries.json"
    if queries_path.exists():
        digest.update(queries_path.read_bytes())
        digest.update(b"\0")
    for chunk in sorted(
        chunks,
        key=lambda item: (
            item.doc_name,
            item.page_number,
            item.chunk_id,
            item.text,
        ),
    ):
        digest.update(chunk.doc_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk.page_number).encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_dynamic_eval_queries(chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
    """Load baseline queries and derive one query for each new table document."""
    queries_path = settings.DATA_DIR / "benchmark_queries.json"
    eval_cases: List[Dict[str, Any]] = []
    covered_docs: set[str] = set()

    if queries_path.exists():
        loaded = json.loads(queries_path.read_text(encoding="utf-8"))
        for item in loaded:
            eval_cases.append(item)
            covered_docs.add(str(item["doc_name"]))

    for chunk in chunks:
        if chunk.doc_name in covered_docs or not chunk.contains_table:
            continue

        cells = TableASTParser.extract_cells(chunk.text, chunk.doc_name, chunk.page_number)
        if not cells:
            continue

        first_row = cells[0].row_key
        columns = sorted({cell.col_header for cell in cells if cell.row_key == first_row})
        if not columns:
            continue

        eval_cases.append(
            {
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "question": (
                    f"What was {first_row} for {' vs '.join(columns[:2])}?"
                ),
            }
        )
        covered_docs.add(chunk.doc_name)

    return eval_cases


def _expected_pages(case: Dict[str, Any]) -> List[int]:
    pages = case.get("evidence_pages") or [case.get("page_number")]
    return sorted({int(page) for page in pages if isinstance(page, int)})


def run_evaluation_suite(
    existing_chunks: List[DocumentChunk] | None = None,
    *,
    reranker_gate: str | None = None,
    reranker_confidence_threshold: float | None = None,
) -> Dict[str, Any]:
    """Run all retrieval modes against disk-backed or currently indexed chunks."""
    parser = MultimodalFinancialParser(
        chunk_size=settings.PARSER_CHUNK_SIZE,
        chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
    )
    all_chunks = (
        existing_chunks
        if existing_chunks is not None
        else load_corpus_chunks_from_pdfs(parser, skip_errors=True)
    )
    eval_cases = build_dynamic_eval_queries(all_chunks)
    available_pages = {(chunk.doc_name, chunk.page_number) for chunk in all_chunks}
    skipped_cases: List[Dict[str, Any]] = []
    valid_cases: List[Dict[str, Any]] = []
    for case in eval_cases:
        expected_pages = _expected_pages(case)
        if not expected_pages or not any(
            (case["doc_name"], page) in available_pages for page in expected_pages
        ):
            skipped_cases.append(
                {
                    "doc_name": case.get("doc_name"),
                    "expected_pages": expected_pages,
                    "question": case.get("question"),
                }
            )
            logger.warning(
                "Skipping benchmark case with no indexed evidence page: %s %s",
                case.get("doc_name"),
                expected_pages,
            )
            continue
        valid_cases.append(case)
    eval_cases = valid_cases

    retriever = HybridRetriever(in_memory=True)
    reranker = CrossEncoderReranker(
        cross_encoder_gate=reranker_gate or settings.RERANKER_GATE,
        confidence_threshold=(
            settings.RERANKER_CONFIDENCE_THRESHOLD
            if reranker_confidence_threshold is None
            else reranker_confidence_threshold
        ),
    )
    if all_chunks:
        retriever.index_chunks(all_chunks, use_embedding_cache=True)

    # Warm up both shared PyTorch models before the latency timer starts.
    warmup_candidates = retriever.retrieve_hybrid("warmup query", top_k=2)
    reranker.rerank("warmup query", warmup_candidates, top_n=1)

    modes = ["dense_only", "hybrid_rrf", "hybrid_plus_reranker"]
    summary: Dict[str, Any] = {}
    per_query_breakdown: List[Dict[str, Any]] = [
        {
            "doc_name": case["doc_name"],
            "expected_page": _expected_pages(case)[0],
            "expected_pages": _expected_pages(case),
            "question": case["question"],
            "ranks": {},
        }
        for case in eval_cases
    ]

    for mode in modes:
        hits_at_1 = 0
        hits_at_3 = 0
        mrr_total = 0.0
        ndcg_total = 0.0
        start_t = time.perf_counter()

        for idx, case in enumerate(eval_cases):
            query = case["question"]
            expected_doc = case["doc_name"]
            expected_pages = set(_expected_pages(case))

            if mode == "dense_only":
                results = retriever.retrieve_dense_only(query, top_k=4)
            elif mode == "hybrid_rrf":
                results = retriever.retrieve_hybrid(query, top_k=4)
            else:
                candidates = retriever.retrieve_hybrid(query, top_k=4)
                results = reranker.rerank(query, candidates, top_n=4)

            rank_found = next(
                (
                    rank_idx
                    for rank_idx, result in enumerate(results, start=1)
                    if result.doc_name == expected_doc
                    and result.page_number in expected_pages
                ),
                None,
            )
            per_query_breakdown[idx]["ranks"][mode] = rank_found

            if rank_found == 1:
                hits_at_1 += 1
            if rank_found is not None and rank_found <= 3:
                hits_at_3 += 1
            if rank_found is not None:
                mrr_total += 1.0 / rank_found
                if rank_found <= 3:
                    ndcg_total += 1.0 / math.log2(rank_found + 1)

        query_count = max(len(eval_cases), 1)
        avg_latency = ((time.perf_counter() - start_t) * 1000.0) / query_count
        summary[mode] = {
            "precision_at_1": round(hits_at_1 / query_count, 4),
            "recall_at_3": round(hits_at_3 / query_count, 4),
            "mrr": round(mrr_total / query_count, 4),
            "ndcg_at_3": round(ndcg_total / query_count, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "queries_evaluated": len(eval_cases),
        }

    summary["per_query_breakdown"] = per_query_breakdown
    summary["skipped_cases"] = skipped_cases
    summary["evaluation_signature"] = compute_evaluation_signature(all_chunks)

    if mlflow is not None:
        try:
            tracking_uri = settings.MLFLOW_TRACKING_URI
            if tracking_uri in {
                "http://localhost:5000",
                "http://127.0.0.1:5000",
                "http://localhost:5000/",
            }:
                tracking_uri = "file:./mlruns"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            for mode_name in modes:
                with mlflow.start_run(run_name=f"eval_{mode_name}"):
                    mlflow.log_param("retrieval_mode", mode_name)
                    mlflow.log_param("embedding_model", settings.EMBEDDING_MODEL)
                    mlflow.log_param("reranker_model", settings.RERANKER_MODEL)
                    mlflow.log_metrics(
                        {
                            key: float(value)
                            for key, value in summary[mode_name].items()
                            if isinstance(value, (int, float))
                        }
                    )
        except Exception as exc:
            logger.warning("MLflow logging note: %s", exc)

    report_path = settings.DATA_DIR / "latest_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Evaluation complete:\n%s", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--reranker-gate", default=settings.RERANKER_GATE)
    argument_parser.add_argument(
        "--reranker-confidence-threshold",
        type=float,
        default=settings.RERANKER_CONFIDENCE_THRESHOLD,
    )
    args = argument_parser.parse_args()
    run_evaluation_suite(
        reranker_gate=args.reranker_gate,
        reranker_confidence_threshold=args.reranker_confidence_threshold,
    )
