"""Phase 3 cache, streaming, index-state, and UI checks."""

from __future__ import annotations

import src.api.index_state as index_state
from src.api.ui import WORKBENCH_HTML
from src.cache_and_guardrails.guardrails import EnterpriseGuardrails
from src.cache_and_guardrails.semantic_cache import SemanticCache


def test_streaming_redactor_handles_split_pii() -> None:
    redactor = EnterpriseGuardrails().streaming_redactor(tail_chars=32)
    emitted = ""
    for token in ["contact john.doe@", "example.com, SSN 123-45-", "6789."]:
        emitted += redactor.feed(token)
    emitted += redactor.finish()
    assert "john.doe@example.com" not in emitted
    assert "123-45-6789" not in emitted
    assert "[REDACTED_EMAIL]" in emitted
    assert "[REDACTED_SSN]" in emitted
    assert redactor.triggered_rules == [
        "PII_REDACTED:EMAIL",
        "PII_REDACTED:SSN",
    ]


def test_semantic_cache_memory_fallback_supports_semantic_hit() -> None:
    cache = SemanticCache(
        redis_url="redis://localhost:6399/0",
        similarity_threshold=0.8,
    )
    cache.clear()
    cache.store(
        "What was AMD revenue in FY22?",
        {"answer": "cached", "citations": []},
    )
    exact = cache.lookup("What was AMD revenue in FY22?")
    semantic = cache.lookup("What was AMD revenue in FY22")
    assert exact is not None and exact["cache_similarity"] == 1.0
    assert semantic is not None
    assert semantic["cache_similarity"] >= 0.8
    assert cache.stats()["semantic_hits"] >= 1


def test_index_generation_helpers_use_shared_redis_counter(monkeypatch) -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.value = 6
            self.closed = False

        def incr(self, key):
            assert key == index_state.INDEX_GENERATION_KEY
            self.value += 1
            return self.value

        def get(self, key):
            assert key == index_state.INDEX_GENERATION_KEY
            return str(self.value)

        def close(self) -> None:
            self.closed = True

    fake = FakeRedis()
    monkeypatch.setattr(index_state.redis, "from_url", lambda *args, **kwargs: fake)
    assert index_state.bump_index_generation() == 7
    assert index_state.read_index_generation() == 7
    assert fake.closed is True


def test_workbench_uses_streaming_query_endpoint() -> None:
    assert "/query/stream" in WORKBENCH_HTML
    assert "streamQuestion" in WORKBENCH_HTML
