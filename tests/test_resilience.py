"""Fail-fast behavior for outbound dependencies.

The breaker exists so a down dependency costs one timeout, not one per request. These tests pin the
state machine and the two wirings that matter on the hot path: the shared index counter and the
remote LLM.
"""

from __future__ import annotations

import redis

from src.adapters.circuit_breaker import CircuitBreaker


def test_breaker_opens_after_threshold_and_recovers_after_window() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0, clock=lambda: now[0])

    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False

    now[0] = 29.0
    assert breaker.allow() is False
    now[0] = 30.0
    assert breaker.allow() is True
    assert breaker.state == "half_open"

    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow() is True


def test_breaker_reopens_when_the_probe_fails() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, clock=lambda: now[0])
    breaker.record_failure()
    now[0] = 10.0
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False


def test_semantic_cache_reconnects_after_redis_returns(monkeypatch) -> None:
    """A Redis blip must not degrade the cache until the API restarts."""
    from src.cache_and_guardrails import semantic_cache as module

    class FakeRedis:
        def ping(self):
            return True

        def close(self):
            return None

    monkeypatch.setattr(
        module, "build_redis_client", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    )
    cache = module.SemanticCache(redis_url="redis://127.0.0.1:6399/0")
    assert cache._redis_client is None
    assert cache._degraded is True

    monkeypatch.setattr(module, "build_redis_client", lambda *a, **k: FakeRedis())
    cache._last_reconnect_attempt = 0.0
    assert cache.check_health() is True
    assert cache._redis_client is not None
    assert cache._degraded is False


def test_index_counter_fails_fast_while_redis_is_down(monkeypatch) -> None:
    from src.api import index_state

    index_state._BREAKER.reset()
    attempts = {"count": 0}

    def failing(*args, **kwargs):
        attempts["count"] += 1
        raise redis.ConnectionError("down")

    monkeypatch.setattr(index_state, "build_redis_client", failing)

    assert index_state.bump_index_generation() is None
    assert index_state.read_index_generation() is None
    assert attempts["count"] == 2
    # Breaker is open: further calls return without attempting a connection.
    assert index_state.bump_index_generation() is None
    assert attempts["count"] == 2
    index_state._BREAKER.reset()


async def test_remote_llm_falls_back_without_retrying_while_the_breaker_is_open() -> None:
    import httpx

    from src.domain.documents import RetrievedDocument
    from src.inference.llm_client import UnifiedLLMClient

    calls = {"count": 0}

    def failing(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectError("provider down", request=request)

    chunk = RetrievedDocument(
        chunk_id="c1",
        doc_name="AMD_2022_10K",
        page_number=1,
        text="Net revenue was $23,601M in FY2022.",
        contains_table=False,
        dense_score=1.0,
        bm25_score=1.0,
        rrf_score=1.0,
        rerank_score=1.0,
    )
    client = UnifiedLLMClient(
        remote_enabled=True,
        transport=httpx.MockTransport(failing),
        timeout_seconds=1.0,
    )
    try:
        first = await client.generate_answer("What was AMD revenue?", [chunk])
        second = await client.generate_answer("What was AMD revenue?", [chunk])
        third = await client.generate_answer("What was AMD revenue?", [chunk])

        assert calls["count"] == 2
        assert first == second == third
        assert client.last_provider == "local_fallback"
        assert client.remote_failures == 2
    finally:
        await client.aclose()
