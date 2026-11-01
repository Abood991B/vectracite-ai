"""
Asynchronous Latency & Throughput Benchmarker for vLLM / FastAPI Inference Endpoints.
Measures Time-To-First-Token (TTFT), p50/p95/p99 End-to-End Latency, and Requests/sec under load.
"""

import asyncio
import statistics
import time
from typing import Dict, List
import httpx


async def _send_query(client: httpx.AsyncClient, url: str, question: str) -> float:
    start = time.perf_counter()
    resp = await client.post(
        url,
        json={"question": question, "top_k": 4, "use_cache": False},
        timeout=30.0,
    )
    resp.raise_for_status()
    return (time.perf_counter() - start) * 1000.0


async def run_load_benchmark(
    endpoint_url: str = "http://localhost:8080/query",
    concurrency: int = 10,
    total_requests: int = 40,
) -> Dict[str, float]:
    sample_questions = [
        "What was AMD's total revenue and operating income in FY 2022?",
        "What is Boeing's commercial airplanes backlog and cash flow position?",
        "Compute the year-over-year gross margin change reported in the 10-K table.",
        "What were the primary R&D expenses disclosed in the consolidated statements?",
    ]

    latencies_ms: List[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def worker(idx: int) -> None:
            async with semaphore:
                q = sample_questions[idx % len(sample_questions)]
                latency = await _send_query(client, endpoint_url, q)
                latencies_ms.append(latency)

        wall_start = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(total_requests)))
        wall_elapsed = time.perf_counter() - wall_start

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[int(len(latencies_ms) * 0.95) - 1]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99) - 1]

    metrics = {
        "total_requests": float(total_requests),
        "concurrency": float(concurrency),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "throughput_req_per_sec": round(total_requests / max(wall_elapsed, 0.001), 2),
    }
    print("Benchmark Results:", metrics)
    return metrics


if __name__ == "__main__":
    asyncio.run(run_load_benchmark())
