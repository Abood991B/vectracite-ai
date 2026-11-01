"""Data-driven multi-domain retrieval and adaptive SCAR ablation benchmark.

The evaluator reads the same real PDF corpus used by the API, loads checked-in
benchmark questions, and adds schema-derived cases for new table-bearing PDFs.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import mlflow
except ModuleNotFoundError:  # Optional for local evaluation and CI environments.
    mlflow = None

from src.config import settings
from src.ingestion.pdf_parser import DocumentChunk, MultimodalFinancialParser
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.table_ast import TableASTParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_corpus_chunks_from_pdfs(
    parser: MultimodalFinancialParser | None = None,
) -> List[DocumentChunk]:
    """Parse every real PDF stored in the configured corpus directory."""
    active_parser = parser or MultimodalFinancialParser(chunk_size=900, chunk_overlap=150)
    settings.CORPUS_PDF_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: List[DocumentChunk] = []
    for pdf_path in sorted(settings.CORPUS_PDF_DIR.glob("*.pdf")):
        all_chunks.extend(active_parser.parse_pdf(pdf_path))
    return all_chunks


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
                    f"In {chunk.doc_name}, what was {first_row} for "
                    f"{' vs '.join(columns[:2])}?"
                ),
            }
        )
        covered_docs.add(chunk.doc_name)

    return eval_cases


def run_evaluation_suite(existing_chunks: List[DocumentChunk] | None = None) -> Dict[str, Any]:
    """Run all retrieval modes against disk-backed or currently indexed chunks."""
    parser = MultimodalFinancialParser(chunk_size=900, chunk_overlap=150)
    all_chunks = existing_chunks if existing_chunks is not None else load_corpus_chunks_from_pdfs(parser)
    eval_cases = build_dynamic_eval_queries(all_chunks)

    retriever = HybridRetriever(in_memory=True)
    reranker = CrossEncoderReranker()
    if all_chunks:
        retriever.index_chunks(all_chunks)

    # Warm up both shared PyTorch models before the latency timer starts.
    warmup_candidates = retriever.retrieve_hybrid("warmup query", top_k=2)
    reranker.rerank("warmup query", warmup_candidates, top_n=1)

    modes = ["dense_only", "hybrid_rrf", "hybrid_plus_reranker"]
    summary: Dict[str, Any] = {}
    per_query_breakdown: List[Dict[str, Any]] = [
        {
            "doc_name": case["doc_name"],
            "expected_page": case["page_number"],
            "question": case["question"],
            "ranks": {},
        }
        for case in eval_cases
    ]

    for mode in modes:
        hits_at_1 = 0
        hits_at_3 = 0
        mrr_total = 0.0
        start_t = time.perf_counter()

        for idx, case in enumerate(eval_cases):
            query = case["question"]
            expected_doc = case["doc_name"]
            expected_page = case["page_number"]

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
                    if result.doc_name == expected_doc and result.page_number == expected_page
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

        query_count = max(len(eval_cases), 1)
        avg_latency = ((time.perf_counter() - start_t) * 1000.0) / query_count
        summary[mode] = {
            "precision_at_1": round(hits_at_1 / query_count, 4),
            "recall_at_3": round(hits_at_3 / query_count, 4),
            "mrr": round(mrr_total / query_count, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "queries_evaluated": len(eval_cases),
        }

    summary["per_query_breakdown"] = per_query_breakdown

    if mlflow is not None:
        try:
            mlflow.set_tracking_uri("file:./mlruns")
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
    run_evaluation_suite()
