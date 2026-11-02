"""Cross-replica mutation serialization using Redis when enabled."""

import asyncio
import logging
import time
import uuid

import redis
from fastapi import HTTPException

from src.config import settings

logger = logging.getLogger(__name__)

_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


class DistributedMutationLock:
    """Best-effort global lease so multiple API replicas do not interleave writes."""

    def __init__(
        self,
        *,
        key: str = "vectracite:index:mutation",
        ttl_seconds: int = 300,
        acquire_timeout_seconds: float = 30.0,
    ) -> None:
        self.key = key
        self.ttl_ms = ttl_seconds * 1000
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self.token = uuid.uuid4().hex
        self._client: redis.Redis | None = None

    async def __aenter__(self) -> "DistributedMutationLock":
        if not settings.REQUIRE_REDIS:
            return self
        self._client = redis.Redis.from_url(
            settings.REDIS_URL,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
        deadline = time.monotonic() + self.acquire_timeout_seconds
        while True:
            try:
                acquired = await asyncio.to_thread(
                    self._client.set,
                    self.key,
                    self.token,
                    nx=True,
                    px=self.ttl_ms,
                )
            except redis.RedisError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Index mutation lock is unavailable. Retry shortly.",
                ) from exc
            if acquired:
                return self
            if time.monotonic() >= deadline:
                raise HTTPException(
                    status_code=503,
                    detail="Timed out waiting for the index mutation lock.",
                )
            await asyncio.sleep(0.05)

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self._client is None:
            return
        try:
            await asyncio.to_thread(
                self._client.eval,
                _RELEASE_SCRIPT,
                1,
                self.key,
                self.token,
            )
        except redis.RedisError:
            logger.warning("Unable to release the distributed mutation lock")
        finally:
            self._client.close()


def distributed_mutation_lock() -> DistributedMutationLock:
    """Create one per-request lease; no-op unless Redis is required."""
    return DistributedMutationLock()
