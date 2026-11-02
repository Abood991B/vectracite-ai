"""Shared index generation state used to avoid full catalog rehydrations."""

from __future__ import annotations

import logging

import redis

from src.adapters.redis_clients import INDEX_TIMEOUT_SECONDS, build_redis_client
from src.config import settings

logger = logging.getLogger(__name__)

INDEX_GENERATION_KEY = "vectracite:index:generation"


def _client() -> redis.Redis:
    return build_redis_client(
        settings.REDIS_URL,
        timeout=INDEX_TIMEOUT_SECONDS,
        decode_responses=False,
    )


def bump_index_generation() -> int | None:
    """Increment the shared generation after an index mutation."""
    try:
        client = _client()
        try:
            return int(client.incr(INDEX_GENERATION_KEY))
        finally:
            client.close()
    except redis.RedisError as exc:
        logger.info("Index generation unavailable: %s", type(exc).__name__)
        return None


def read_index_generation() -> int | None:
    """Read the shared generation without treating Redis as required."""
    try:
        client = _client()
        try:
            value = client.get(INDEX_GENERATION_KEY)
        finally:
            client.close()
    except redis.RedisError as exc:
        logger.info("Index generation unavailable: %s", type(exc).__name__)
        return None
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
