"""Latency harness for the six hot paths measured by the architecture-optimization journey.

Two modes, so a repeatable number and a realistic one can be compared:

    inproc  in-memory Qdrant, lightweight models, fixture corpus, no network.
            Deterministic and fast enough to run on every change.
    http    a running server (real corpus, real BGE models, real Redis/Qdrant).

Usage:
    python scripts/bench_hot_paths.py --mode inproc --iterations 15
    python scripts/bench_hot_paths.py --mode http --base-url http://127.0.0.1:8080

Every scenario reports p50/p95/p99 with nearest-rank percentiles. Results are written to
``data/bench_hot_paths_<mode>.json`` unless ``--output`` points elsewhere.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CORPUS = REPO_ROOT / "tests" / "fixtures" / "corpus_pdfs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_QUESTIONS = [
    ("What was net revenue in FY2022?", "AMD_2022_10K"),
    ("What was operating income in FY2023?", "MSFT_2023_10K"),
    ("What was total revenue for the year?", "JPM_2022_10K"),
    ("What was the primary endpoint result?", "ONCOLOGY_TRIAL_2025"),
    ("How did gross margin change year over year?", None),
]

# Deliberately generous: these catch a catastrophic regression on shared CI hardware, not a
# percentage drift. The real budgets live in the internal performance ledger and are watched on a known machine.
GATE_CAPS_MS = {
    "query_cold": 1000.0,
    "query_cache_hit": 500.0,
    "query_stream_ttfb": 1000.0,
    "retrieve_hybrid": 200.0,
    "rerank": 200.0,
    "ingest_pdf": 20000.0,
    "workbench_render": 500.0,
    "evaluate_latest": 2000.0,
}

INGEST_PAGES = [
    {
        "page_number": page,
        "text": (
            f"| Metric | FY 2022 | FY 2021 |\n"
            f"| --- | --- | --- |\n"
            f"| Net revenue | ${23_601 + page}M | ${16_434 + page}M |\n"
            f"| Operating income | ${1_264 + page}M | ${3_648 + page}M |"
        ),
    }
    for page in range(1, 6)
]


def percentile(samples: Sequence[float], p: float) -> float:
    """Nearest-rank percentile, matching the convention used by the evaluation harness."""
    ordered = sorted(samples)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[rank - 1]


def summarize(name: str, samples: Sequence[float], note: str = "") -> Dict[str, Any]:
    values = [round(float(value), 2) for value in samples]
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
        "min": round(min(values), 2) if values else 0.0,
        "max": round(max(values), 2) if values else 0.0,
        "mean": round(statistics.fmean(values), 2) if values else 0.0,
        "samples_ms": values,
        "note": note,
    }


def timed(call: Callable[[], Any]) -> float:
    started = time.perf_counter()
    call()
    return (time.perf_counter() - started) * 1000.0


def _stream_measure(client: Any, question: str, doc_name: str | None) -> tuple[float, float]:
    """Return (time to first byte, total time) for one SSE query."""
    payload = {"question": question, "doc_name": doc_name, "top_k": 3, "use_cache": False}
    started = time.perf_counter()
    first_byte = 0.0
    with client.stream("POST", "/query/stream", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: ") and not first_byte:
                first_byte = (time.perf_counter() - started) * 1000.0
    total = (time.perf_counter() - started) * 1000.0
    return first_byte, total


def _ingest_pdf(client: Any, parser_module: Any, doc_name: str) -> None:
    pdf_bytes = parser_module.render_pages_to_pdf_bytes(doc_name, INGEST_PAGES)
    response = client.post(
        "/ingest/pdf",
        json={
            "doc_name": doc_name,
            "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
        },
    )
    response.raise_for_status()
    client.delete(f"/documents/{doc_name}")


def _record(
    results: Dict[str, Dict[str, Any]],
    name: str,
    samples: List[float],
    note: str = "",
) -> None:
    results[name] = summarize(name, samples, note)
    row = results[name]
    print(f"  {name:<22} n={row['n']:<3} p50={row['p50']:>9.2f}ms p95={row['p95']:>9.2f}ms")


def bench_inproc(iterations: int, warmup: int) -> Dict[str, Dict[str, Any]]:
    """Measure every scenario in-process against the fixture corpus."""
    os.environ["USE_LIGHTWEIGHT_MODELS"] = "true"
    os.environ["QDRANT_IN_MEMORY"] = "true"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    from fastapi.testclient import TestClient

    from src.config import settings

    data_dir = Path(tempfile.mkdtemp(prefix="vectracite-bench-"))
    settings.DATA_DIR = data_dir
    settings.PDF_DIR = data_dir / "pdfs"
    settings.CORPUS_PDF_DIR = FIXTURE_CORPUS
    settings.CORPUS_INDEX_PATH = data_dir / "corpus_index.json"
    settings.EMBEDDING_CACHE_PATH = data_dir / "corpus_embeddings.npz"
    settings.USER_UPLOADS_DIR = data_dir / "uploads"
    settings.MLFLOW_TRACKING_URI = f"file:{(data_dir / 'mlruns').as_posix()}"
    for directory in (settings.PDF_DIR, settings.USER_UPLOADS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    from src.api import main as api
    from src.ingestion import pdf_parser

    client = TestClient(api.app)
    results: Dict[str, Dict[str, Any]] = {}

    cold_start = timed(lambda: client.post("/query", json={"question": FIXTURE_QUESTIONS[0][0], "use_cache": False}))
    _record(
        results,
        "cold_start_first_query",
        [cold_start],
        "process start to first answered query, includes imports and index build",
    )

    def query(question: str, doc_name: str | None, use_cache: bool) -> float:
        return timed(
            lambda: client.post(
                "/query",
                json={"question": question, "doc_name": doc_name, "top_k": 3, "use_cache": use_cache},
            )
        )

    for _ in range(warmup):
        query(*FIXTURE_QUESTIONS[0], use_cache=False)

    cold_samples = [
        query(*FIXTURE_QUESTIONS[index % len(FIXTURE_QUESTIONS)], use_cache=False)
        for index in range(iterations)
    ]
    _record(results, "query_cold", cold_samples, "/query, cache miss")

    for question, doc_name in FIXTURE_QUESTIONS:
        client.post("/query", json={"question": question, "doc_name": doc_name, "top_k": 3, "use_cache": True})
    hit_samples = [
        query(*FIXTURE_QUESTIONS[index % len(FIXTURE_QUESTIONS)], use_cache=True)
        for index in range(iterations)
    ]
    _record(results, "query_cache_hit", hit_samples, "/query, semantic cache hit")

    ttfb: List[float] = []
    stream_totals: List[float] = []
    for index in range(iterations):
        question, doc_name = FIXTURE_QUESTIONS[index % len(FIXTURE_QUESTIONS)]
        first, total = _stream_measure(client, question, doc_name)
        ttfb.append(first)
        stream_totals.append(total)
    _record(results, "query_stream_ttfb", ttfb, "/query/stream to first data event")
    _record(results, "query_stream_total", stream_totals, "/query/stream to last event")

    retriever = api.retriever
    hybrid_samples = [
        timed(lambda: retriever.retrieve_hybrid(query=FIXTURE_QUESTIONS[0][0], top_k=6, doc_name=None))
        for _ in range(iterations)
    ]
    _record(results, "retrieve_hybrid", hybrid_samples, "dense + BM25 + RRF only")

    candidates = retriever.retrieve_hybrid(query=FIXTURE_QUESTIONS[0][0], top_k=6, doc_name=None)
    rerank_samples = [
        timed(lambda: api.reranker.rerank(query=FIXTURE_QUESTIONS[0][0], candidates=candidates, top_n=3))
        for _ in range(iterations)
    ]
    _record(results, "rerank", rerank_samples, "cross-encoder over hybrid candidates")

    _ingest_pdf(client, pdf_parser, "BENCH_WARMUP")
    ingest_samples = [
        timed(lambda index=index: _ingest_pdf(client, pdf_parser, f"BENCH_INGEST_{index}"))
        for index in range(iterations)
    ]
    _record(results, "ingest_pdf", ingest_samples, "5-page PDF upload, parse, index, delete")

    workbench_samples = [timed(lambda: client.get("/")) for _ in range(iterations)]
    _record(results, "workbench_render", workbench_samples, "GET / dashboard HTML")

    client.get("/evaluate/latest")
    evaluate_samples = [timed(lambda: client.get("/evaluate/latest")) for _ in range(iterations)]
    _record(results, "evaluate_latest", evaluate_samples, "GET /evaluate/latest")

    return results


def bench_http(base_url: str, iterations: int, warmup: int, include_evaluate: bool) -> Dict[str, Dict[str, Any]]:
    """Measure the in-scope scenarios against a live server."""
    import httpx

    results: Dict[str, Dict[str, Any]] = {}
    print(f"  target {base_url}")

    with httpx.Client(base_url=base_url, timeout=180.0) as client:
        benchmark_path = REPO_ROOT / "data" / "benchmark_queries.json"
        cases = json.loads(benchmark_path.read_text(encoding="utf-8"))
        questions = [(case["question"], case.get("doc_name")) for case in cases[:8]]

        cold_start = timed(lambda: client.post("/query", json={"question": questions[0][0], "use_cache": False}))
        _record(
            results,
            "cold_start_first_query",
            [cold_start],
            "measured from a warm server process; true cold start is timed separately",
        )

        def query(pair: tuple[str, str | None], use_cache: bool) -> float:
            question, doc_name = pair
            return timed(
                lambda: client.post(
                    "/query",
                    json={"question": question, "doc_name": doc_name, "top_k": 3, "use_cache": use_cache},
                )
            )

        for _ in range(warmup):
            query(questions[0], use_cache=False)

        cold_samples = [query(questions[index % len(questions)], use_cache=False) for index in range(iterations)]
        _record(results, "query_cold", cold_samples, "/query, cache miss")

        for pair in questions:
            query(pair, use_cache=True)
        hit_samples = [query(questions[index % len(questions)], use_cache=True) for index in range(iterations)]
        _record(results, "query_cache_hit", hit_samples, "/query, semantic cache hit")

        ttfb: List[float] = []
        stream_totals: List[float] = []
        for index in range(iterations):
            question, doc_name = questions[index % len(questions)]
            started = time.perf_counter()
            first = 0.0
            with client.stream(
                "POST",
                "/query/stream",
                json={"question": question, "doc_name": doc_name, "top_k": 3, "use_cache": False},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: ") and not first:
                        first = (time.perf_counter() - started) * 1000.0
            ttfb.append(first)
            stream_totals.append((time.perf_counter() - started) * 1000.0)
        _record(results, "query_stream_ttfb", ttfb, "/query/stream to first data event")
        _record(results, "query_stream_total", stream_totals, "/query/stream to last event")

        from src.ingestion.pdf_parser import render_pages_to_pdf_bytes

        ingest_samples: List[float] = []
        for index in range(iterations):
            pdf_bytes = render_pages_to_pdf_bytes(f"BENCH_HTTP_{index}", INGEST_PAGES)
            body = {
                "doc_name": f"BENCH_HTTP_{index}",
                "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
            }
            ingest_samples.append(timed(lambda body=body: client.post("/ingest/pdf", json=body)))
            client.delete(f"/documents/BENCH_HTTP_{index}")
        _record(results, "ingest_pdf", ingest_samples, "5-page PDF upload, parse, index, delete")

        workbench_samples = [timed(lambda: client.get("/")) for _ in range(iterations)]
        _record(results, "workbench_render", workbench_samples, "GET / dashboard HTML")

        client.get("/evaluate/latest")
        evaluate_samples = [timed(lambda: client.get("/evaluate/latest")) for _ in range(iterations)]
        _record(results, "evaluate_latest", evaluate_samples, "GET /evaluate/latest")

        if include_evaluate:
            full = timed(lambda: client.post("/evaluate"))
            _record(results, "evaluate_run", [full], "POST /evaluate full ablation, single sample")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inproc", "http"), default="inproc")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--label", default="")
    parser.add_argument("--include-evaluate", action="store_true", help="http mode: also time one full POST /evaluate")
    parser.add_argument("--gate", action="store_true", help="exit non-zero when a scenario exceeds its CI cap")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    logging.getLogger().setLevel(logging.WARNING)
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.mode == "inproc":
        results = bench_inproc(args.iterations, args.warmup)
    else:
        results = bench_http(args.base_url, args.iterations, args.warmup, args.include_evaluate)

    payload = {
        "mode": args.mode,
        "label": args.label,
        "timestamp": started_at,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "scenarios": results,
    }

    output = Path(args.output) if args.output else REPO_ROOT / "data" / f"bench_hot_paths_{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")

    if args.gate:
        breaches = [
            f"{name}: p95 {row['p95']} ms > cap {cap} ms"
            for name, row in results.items()
            if (cap := GATE_CAPS_MS.get(name)) is not None and row["p95"] > cap
        ]
        if breaches:
            print("\nBUDGET GATE FAILED")
            for breach in breaches:
                print(f"  {breach}")
            return 1
        print("\nbudget gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
