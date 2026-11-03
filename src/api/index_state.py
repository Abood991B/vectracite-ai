"""Shared index generation state used to avoid full catalog rehydrations."""

from __future__ import annotations

import logging

import redis

from src.adapters.circuit_breaker import CircuitBreaker
from src.adapters.redis_clients import INDEX_TIMEOUT_SECONDS, build_redis_client
from src.config import settings

logger = logging.getLogger(__name__)

INDEX_GENERATION_KEY = "vectracite:index:generation"

# When Redis is down, every mutation used to pay the 1 s connect timeout. Two consecutive failures
# open the breaker, so the sync loop and mutations fail fast to the "generation unavailable" path.
_BREAKER = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0)


def _client() -> redis.Redis | None:
    if not _BREAKER.allow():
        return None
    try:
        return build_redis_client(
            settings.REDIS_URL,
            timeout=INDEX_TIMEOUT_SECONDS,
            decode_responses=False,
        )
    except Exception as exc:
        _BREAKER.record_failure()
        logger.info("Index generation unavailable: %s", type(exc).__name__)
        return None


def bump_index_generation() -> int | None:
    """Increment the shared generation after an index mutation."""
    client = _client()
    if client is None:
        return None
    try:
        try:
            value = int(client.incr(INDEX_GENERATION_KEY))
        finally:
            client.close()
    except redis.RedisError as exc:
        _BREAKER.record_failure()
        logger.info("Index generation unavailable: %s", type(exc).__name__)
        return None
    _BREAKER.record_success()
    return value


def read_index_generation() -> int | None:
    """Read the shared generation without treating Redis as required."""
    client = _client()
    if client is None:
        return None
    try:
        try:
            value = client.get(INDEX_GENERATION_KEY)
        finally:
            client.close()
    except redis.RedisError as exc:
        _BREAKER.record_failure()
        logger.info("Index generation unavailable: %s", type(exc).__name__)
        return None
    _BREAKER.record_success()
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
