"""Redis client construction.

Two modules need a Redis connection and both must fail fast when Redis is absent: the semantic cache
and the shared index-generation counter. Keeping the timeout policy in one place means a change to how
the service treats an unavailable Redis is one edit, not three.
"""

from __future__ import annotations

from typing import Any

import redis

CACHE_TIMEOUT_SECONDS = 1.5
INDEX_TIMEOUT_SECONDS = 1.0
LOCK_TIMEOUT_SECONDS = 2.0


def build_redis_client(url: str, *, timeout: float, decode_responses: bool = True) -> Any:
    """Return a Redis client that cannot block a request for longer than ``timeout`` seconds."""
    return redis.from_url(
        url,
        decode_responses=decode_responses,
        socket_connect_timeout=timeout,
        socket_timeout=timeout,
    )
