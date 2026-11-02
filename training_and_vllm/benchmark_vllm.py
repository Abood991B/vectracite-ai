"""
Asynchronous Latency & Throughput Benchmarker for vLLM / FastAPI Inference Endpoints.
Measures Time-To-First-Token (TTFT), p50/p95/p99 end-to-end latency, throughput, and failures.
"""

import asyncio
import json
import math
import statistics
import time
from typing import Dict, List
import httpx


def _percentile(sorted_values: List[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * fraction) - 1))
    return sorted_values[index]


async def _send_query(
    client: httpx.AsyncClient, url: str, question: str
) -> tuple[float, float | None]:
    """Return end-to-end latency and TTFT for one real SSE query."""
    start = time.perf_counter()
    first_token_at: float | None = None
    event_name = "message"
    verified = False
    done = False
    async with client.stream(
        "POST",
        url,
        json={"question": question, "top_k": 4, "use_cache": False},
        timeout=30.0,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            if event_name == "verification":
                data = json.loads(payload)
                verified = bool(data.get("answer")) and "verification" in data
            if event_name == "token" and json.loads(payload).get("text") and first_token_at is None:
                first_token_at = time.perf_counter()
    if first_token_at is None or not verified or not done:
        raise ValueError("incomplete answer stream")
    end_to_end = (time.perf_counter() - start) * 1000.0
    ttft = None if first_token_at is None else (first_token_at - start) * 1000.0
    return end_to_end, ttft


async def run_load_benchmark(
    endpoint_url: str = "http://localhost:8080/query/stream",
    concurrency: int = 10,
    total_requests: int = 40,
) -> Dict[str, float]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if total_requests < 1:
        raise ValueError("total_requests must be at least 1")
    if not endpoint_url.strip():
        raise ValueError("endpoint_url cannot be empty")

    sample_questions = [
        "What was AMD's total revenue and operating income in FY 2022?",
        "What is Boeing's commercial airplanes backlog and cash flow position?",
        "Compute the year-over-year gross margin change reported in the 10-K table.",
        "What were the primary R&D expenses disclosed in the consolidated statements?",
    ]

    latencies_ms: List[float] = []
    ttfts_ms: List[float] = []
    failures = 0
    failure_details: List[str] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def worker(idx: int) -> None:
            nonlocal failures
            async with semaphore:
                question = sample_questions[idx % len(sample_questions)]
                try:
                    latency, ttft = await _send_query(client, endpoint_url, question)
                    latencies_ms.append(latency)
                    if ttft is not None:
                        ttfts_ms.append(ttft)
                except Exception as exc:
                    failures += 1
                    if len(failure_details) < 10:
                        failure_details.append(type(exc).__name__)

        wall_start = time.perf_counter()
        await asyncio.gather(*(worker(index) for index in range(total_requests)))
        wall_elapsed = time.perf_counter() - wall_start

    latencies_ms.sort()
    ttfts_ms.sort()
    metrics = {
        "total_requests": float(total_requests),
        "successful_requests": float(len(latencies_ms)),
        "failed_requests": float(failures),
        "error_rate": round(failures / total_requests, 4),
        "concurrency": float(concurrency),
        "p50_ttft_ms": round(statistics.median(ttfts_ms), 2) if ttfts_ms else 0.0,
        "p95_ttft_ms": round(_percentile(ttfts_ms, 0.95), 2),
        "p50_latency_ms": round(statistics.median(latencies_ms), 2) if latencies_ms else 0.0,
        "p95_latency_ms": round(_percentile(latencies_ms, 0.95), 2),
        "p99_latency_ms": round(_percentile(latencies_ms, 0.99), 2),
        "throughput_req_per_sec": round(
            len(latencies_ms) / max(wall_elapsed, 0.001), 2
        ),
        "failure_types": ", ".join(sorted(set(failure_details))),
    }
    print("Benchmark Results:", metrics)
    return metrics


if __name__ == "__main__":
    asyncio.run(run_load_benchmark())
