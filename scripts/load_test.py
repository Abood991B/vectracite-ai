"""Load-test the query API and report latency, throughput, and errors.

Usage:
    python scripts/load_test.py --base-url http://127.0.0.1:8080 --duration 30 --concurrency 4

The question mix comes from the checked-in benchmark; scope is varied so retrieval does real work.
Results are written to data/load_test_report.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent


def percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(round(len(ordered) * fraction)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--output", default="data/load_test_report.json")
    args = parser.parse_args()

    cases = json.loads(
        (REPO_ROOT / "data" / "benchmark_queries.json").read_text(encoding="utf-8")
    )
    questions = [(case["question"], case.get("doc_name")) for case in cases]
    latencies: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    stop_at = time.perf_counter() + args.duration

    def worker(worker_id: int) -> None:
        index = worker_id
        with httpx.Client(base_url=args.base_url, timeout=180.0) as client:
            while time.perf_counter() < stop_at:
                question, doc = questions[index % len(questions)]
                index += args.concurrency
                started = time.perf_counter()
                try:
                    response = client.post(
                        "/query",
                        json={
                            "question": question,
                            "doc_name": doc,
                            "top_k": 4,
                            "use_cache": args.use_cache,
                        },
                    )
                    response.raise_for_status()
                    elapsed = (time.perf_counter() - started) * 1000.0
                    with lock:
                        latencies.append(elapsed)
                except Exception as exc:
                    with lock:
                        errors.append(f"{type(exc).__name__}: {exc}")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(worker, range(args.concurrency)))
    wall = time.perf_counter() - started

    report = {
        "base_url": args.base_url,
        "duration_seconds": round(wall, 2),
        "concurrency": args.concurrency,
        "use_cache": args.use_cache,
        "requests": len(latencies) + len(errors),
        "errors": len(errors),
        "error_samples": errors[:3],
        "throughput_rps": round((len(latencies) + len(errors)) / wall, 2),
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "p99": round(percentile(latencies, 0.99), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        },
    }
    output = REPO_ROOT / args.output
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
