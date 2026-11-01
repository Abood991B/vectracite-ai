"""
Deterministic Retrieval & SCAR™ AST Reranking Ablation Benchmark.
Evaluates three retrieval architectures across multi-industry table queries:
  1. dense_only: Baseline Qdrant cosine similarity search (BAAI/bge-small-en-v1.5)
  2. hybrid_rrf: Pure Reciprocal Rank Fusion (Qdrant Dense + Sparse BM25Okapi)
  3. hybrid_plus_reranker: SCAR™ (Speculative 2D Table-AST + BAAI/bge-reranker-base)
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

import mlflow
from src.config import settings
from src.ingestion.pdf_parser import MultimodalFinancialParser
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CURATED_FINANCEBENCH_EVAL_SET: List[Dict[str, Any]] = [
    {
        "doc_name": "AMD_2022_10K",
        "page_number": 51,
        "question": "In AMD's Consolidated Statements of Operations table, what were total net revenue, gross profit, and operating income for FY 2022 vs FY 2021?",
        "gold_keywords": ["23,601", "1,264"],
        "table_page_text": (
            "ITEM 8. AMD CONSOLIDATED STATEMENTS OF OPERATIONS TABLE (in millions)\n"
            "| Consolidated Line Item | FY 2022 | FY 2021 | FY 2020 |\n"
            "| --- | --- | --- | --- |\n"
            "| Total net revenue | $ 23,601 | $ 16,434 | $ 9,763 |\n"
            "| Cost of sales | $ 12,998 | $ 8,505 | $ 5,416 |\n"
            "| Gross profit | $ 10,603 | $ 7,929 | $ 4,347 |\n"
            "| Operating income | $ 1,264 | $ 3,648 | $ 1,369 |"
        ),
        "distractor_pages": [
            {
                "page_number": 34,
                "text": (
                    "ITEM 7. MANAGEMENT'S DISCUSSION OF CONSOLIDATED STATEMENTS OF OPERATIONS TABLE. "
                    "In AMD's Consolidated Statements of Operations table, what were total net revenue, "
                    "gross profit, and operating income for FY 2022 vs FY 2021? Total net revenue, gross profit, "
                    "and operating income for FY 2022 vs FY 2021 reflected strong Data Center demand."
                ),
            },
        ],
    },
    {
        "doc_name": "MSFT_2023_10K",
        "page_number": 62,
        "question": "According to Microsoft's Note 19 Segment Revenue table, what was Intelligent Cloud and Productivity revenue in FY23 compared to FY22?",
        "gold_keywords": ["87,907", "74,965"],
        "table_page_text": (
            "MICROSOFT NOTE 19 — SEGMENT REVENUE TABLE (FY23 COMPARED TO FY22, in millions)\n"
            "| Microsoft Segment | FY23 Revenue | FY22 Revenue | YoY % |\n"
            "| --- | --- | --- | --- |\n"
            "| Productivity and Business Processes | $ 69,274 | $ 63,364 | +9% |\n"
            "| Intelligent Cloud (Azure, Server, SQL) | $ 87,907 | $ 74,965 | +17% |\n"
            "| More Personal Computing | $ 54,734 | $ 59,941 | -9% |"
        ),
        "distractor_pages": [
            {
                "page_number": 41,
                "text": (
                    "GENERAL CORPORATE COMMENTARY ON COMMERCIAL CLOUD AND WORKPLACE SOFTWARE TRENDS. "
                    "Across recent annual reporting cycles, enterprise demand for scalable cloud infrastructure, "
                    "relational database workloads, and collaborative workplace subscriptions continued to expand "
                    "across North America, Europe, and Asia-Pacific commercial customer accounts, offset partially "
                    "by foreign currency exchange headwinds and cyclical personal computer hardware shipments in "
                    "consumer retail distribution channels worldwide."
                ),
            },
        ],
    },
    {
        "doc_name": "JPM_2022_10K",
        "page_number": 88,
        "question": "In JPMorgan Chase's Basel III Regulatory Capital table, what were the Standardized CET1 capital ratio and Tier 1 capital ratio on Dec 31, 2022?",
        "gold_keywords": ["13.2%", "14.9%"],
        "table_page_text": (
            "JPMORGAN CHASE BASEL III REGULATORY CAPITAL TABLE (DEC 31, 2022)\n"
            "| JPMorgan Chase Regulatory Ratio | Dec 31, 2022 | Dec 31, 2021 |\n"
            "| --- | --- | --- | --- |\n"
            "| Standardized CET1 capital ratio | 13.2% | 13.1% |\n"
            "| Tier 1 capital ratio | 14.9% | 15.0% |\n"
            "| Total capital ratio | 16.8% | 16.8% |"
        ),
        "distractor_pages": [
            {
                "page_number": 79,
                "text": (
                    "OVERVIEW OF BANKING LIQUIDITY AND BALANCE SHEET RISK GOVERNANCE FRAMEWORKS. "
                    "Global systemically important financial institutions maintain comprehensive capital adequacy "
                    "governance policies, quarterly internal stress testing scenarios, and regulatory buffer "
                    "allocations designed to support wholesale lending, investment banking underwriting, and "
                    "treasury clearing operations under macroeconomic volatility and interest rate shifts."
                ),
            },
        ],
    },
    {
        "doc_name": "ONCOLOGY_TRIAL_2025",
        "page_number": 14,
        "question": "In the Phase-III Clinical Trial Efficacy & Dosage table, what were the Median Progression-Free Survival (months) and Objective Response Rate (%) for the 80mg Cohort?",
        "gold_keywords": ["18.4 months", "64.2%"],
        "table_page_text": (
            "PHASE-III CLINICAL TRIAL EFFICACY & DOSAGE TABLE (80MG COHORT)\n"
            "| Clinical Endpoint | 80mg Cohort (n=240) | Standard Care (n=238) | p-value |\n"
            "| --- | --- | --- | --- |\n"
            "| Median Progression-Free Survival (months) | 18.4 months | 10.2 months | <0.0001 |\n"
            "| Objective Response Rate (%) | 64.2% | 38.7% | <0.0001 |\n"
            "| Grade >=3 Adverse Events (%) | 14.1% | 29.4% | 0.0012 |"
        ),
        "distractor_pages": [
            {
                "page_number": 6,
                "text": (
                    "PATIENT ENROLLMENT CRITERIA AND MULTI-CENTER STUDY METHODOLOGY BACKGROUND. "
                    "Adult patients diagnosed with locally advanced or metastatic non-small cell carcinoma "
                    "were randomized across forty-two international academic medical centers following "
                    "histological confirmation, baseline imaging assessment, and institutional review board "
                    "ethics approval to evaluate long-term tolerability and safety monitoring protocols."
                ),
            },
        ],
    },
]


def build_seed_chunks(parser: MultimodalFinancialParser):
    """Builds the evaluation and startup seed corpus across Finance, Banking, Cloud & Healthcare."""
    all_chunks = []
    for item in CURATED_FINANCEBENCH_EVAL_SET:
        pages = [{"page_number": item["page_number"], "text": item["table_page_text"]}]
        pages.extend(item["distractor_pages"])
        all_chunks.extend(parser.parse_raw_pages(item["doc_name"], pages))
    return all_chunks


def run_evaluation_suite(use_lightweight: bool | None = None) -> Dict[str, Dict[str, float]]:
    lightweight = settings.USE_LIGHTWEIGHT_MODELS if use_lightweight is None else use_lightweight

    parser = MultimodalFinancialParser(chunk_size=900, chunk_overlap=150)
    retriever = HybridRetriever(in_memory=True, use_lightweight_embedder=lightweight)
    reranker = CrossEncoderReranker(use_lightweight_scorer=lightweight)

    all_chunks = build_seed_chunks(parser)
    retriever.index_chunks(all_chunks)

    # Warm up BGE embedder and Cross-Encoder reranker weights in memory before timing
    warmup_candidates = retriever.retrieve_hybrid("warmup query", top_k=2)
    reranker.rerank("warmup query", warmup_candidates, top_n=1)

    modes = ["dense_only", "hybrid_rrf", "hybrid_plus_reranker"]
    summary: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        hits_at_1 = 0
        hits_at_3 = 0
        mrr_total = 0.0
        start_t = time.perf_counter()

        for case in CURATED_FINANCEBENCH_EVAL_SET:
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

            rank_found = None
            for idx, r in enumerate(results, start=1):
                if r.doc_name == expected_doc and r.page_number == expected_page:
                    rank_found = idx
                    break

            if rank_found == 1:
                hits_at_1 += 1
            if rank_found is not None and rank_found <= 3:
                hits_at_3 += 1
            if rank_found is not None:
                mrr_total += 1.0 / rank_found

        n = len(CURATED_FINANCEBENCH_EVAL_SET)
        avg_latency = ((time.perf_counter() - start_t) * 1000.0) / n
        summary[mode] = {
            "precision_at_1": round(hits_at_1 / n, 4),
            "recall_at_3": round(hits_at_3 / n, 4),
            "mrr": round(mrr_total / n, 4),
            "avg_latency_ms": round(avg_latency, 2),
        }

    try:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        for mode_name, metrics in summary.items():
            with mlflow.start_run(run_name=f"eval_{mode_name}"):
                mlflow.log_param("retrieval_mode", mode_name)
                mlflow.log_param("embedding_model", settings.EMBEDDING_MODEL)
                mlflow.log_param("reranker_model", settings.RERANKER_MODEL)
                mlflow.log_metrics(metrics)
    except Exception as exc:
        logger.warning("MLflow logging note: %s", exc)

    report_path = settings.DATA_DIR / "latest_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Evaluation Complete:\n%s", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_evaluation_suite(use_lightweight=False)
