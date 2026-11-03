"""Cross-replica mutation serialization using Redis when enabled."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable

import redis
from fastapi import HTTPException

from src.adapters.circuit_breaker import CircuitBreaker
from src.adapters.redis_clients import LOCK_TIMEOUT_SECONDS, build_redis_client
from src.config import settings

logger = logging.getLogger(__name__)

_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""

# Acquisition also assigns a monotonic commit epoch. Writers stamp every point they write with it,
# so a holder that resumes after losing the lease cannot delete a newer commit. The counter is
# bootstrapped from wall-clock nanoseconds once, so it increases across Redis restarts and stays
# larger than any process-local counter.
_ACQUIRE_SCRIPT = """
if redis.call('set', KEYS[1], ARGV[1], 'NX', 'PX', ARGV[2]) then
    if redis.call('exists', KEYS[2]) == 0 then
        redis.call('set', KEYS[2], ARGV[3])
    end
    return redis.call('incr', KEYS[2])
end
return 0
"""

# Renew only a lease this holder still owns. Recreating a lost lease would let two replicas write,
# which is the exact failure the lease exists to prevent.
_RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""

_RETRY_INTERVAL_SECONDS = 0.05

# An unreachable Redis costs LOCK_TIMEOUT_SECONDS on every mutation before the 503 (measured
# 2.0 s per attempt on 2026-11-03). Two consecutive failures open the breaker so later mutations
# fail fast for the recovery window instead of repeating the timeout.
_LEASE_BREAKER = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0)


class DistributedMutationLock:
    """Best-effort global lease so multiple API replicas do not interleave writes.

    A heartbeat renews the lease every third of its TTL while a mutation is in flight, so a write
    that outlives one TTL keeps its exclusivity instead of being overtaken. The TTL still bounds
    recovery: a crashed holder stops renewing, and the lease expires for the next writer.

    Acquisition also returns a monotonic commit epoch (a Redis ``INCR`` beside the lease). The
    retriever stamps every point it writes with that epoch, so a stale holder cannot delete a newer
    commit even if it resumes after losing the lease.
    """

    def __init__(
        self,
        *,
        key: str = "vectracite:index:mutation",
        epoch_key: str = "vectracite:index:commit_epoch",
        ttl_seconds: float = 300.0,
        acquire_timeout_seconds: float = 30.0,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.key = key
        self.epoch_key = epoch_key
        self.ttl_ms = max(1, int(ttl_seconds * 1000))
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self.token = uuid.uuid4().hex
        self.epoch: int | None = None
        self._client_factory = client_factory or self._build_client
        self._client: Any = None
        self._heartbeat: asyncio.Task[None] | None = None
        self._lease_lost = False

    @staticmethod
    def _build_client() -> Any:
        return build_redis_client(
            settings.REDIS_URL,
            timeout=LOCK_TIMEOUT_SECONDS,
            decode_responses=False,
        )

    @property
    def lease_lost(self) -> bool:
        """True when another replica took the lease while this holder was still writing."""
        return self._lease_lost

    def require_lease(self) -> None:
        """Refuse a write once the lease belongs to another replica."""
        if self._lease_lost:
            raise HTTPException(
                status_code=503,
                detail="The index mutation lease was lost to another replica. Retry shortly.",
            )

    async def __aenter__(self) -> "DistributedMutationLock":
        if not settings.REQUIRE_REDIS:
            return self
        if not _LEASE_BREAKER.allow():
            raise HTTPException(
                status_code=503,
                detail="Index mutation lock is unavailable. Retry shortly.",
            )
        self._client = self._client_factory()
        deadline = time.monotonic() + self.acquire_timeout_seconds
        while True:
            try:
                acquired = await asyncio.to_thread(
                    self._client.eval,
                    _ACQUIRE_SCRIPT,
                    2,
                    self.key,
                    self.epoch_key,
                    self.token,
                    self.ttl_ms,
                    time.time_ns(),
                )
            except redis.RedisError as exc:
                _LEASE_BREAKER.record_failure()
                self._close()
                raise HTTPException(
                    status_code=503,
                    detail="Index mutation lock is unavailable. Retry shortly.",
                ) from exc
            if acquired:
                self.epoch = int(acquired)
                _LEASE_BREAKER.record_success()
                self._heartbeat = asyncio.create_task(self._renew_until_released())
                return self
            if time.monotonic() >= deadline:
                self._close()
                raise HTTPException(
                    status_code=503,
                    detail="Timed out waiting for the index mutation lock.",
                )
            await asyncio.sleep(_RETRY_INTERVAL_SECONDS)

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self._client is None:
            return
        heartbeat, self._heartbeat = self._heartbeat, None
        if heartbeat is not None:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
        try:
            await asyncio.to_thread(
                self._client.eval,
                _RELEASE_SCRIPT,
                1,
                self.key,
                self.token,
            )
        except redis.RedisError:
            _LEASE_BREAKER.record_failure()
            logger.warning("Unable to release the distributed mutation lock")
        finally:
            self._close()

    def _renew_interval(self) -> float:
        """Renew well inside the TTL so one slow round trip cannot drop the lease."""
        return max(self.ttl_ms / 3000.0, 0.05)

    async def _renew_until_released(self) -> None:
        while True:
            await asyncio.sleep(self._renew_interval())
            try:
                renewed = await asyncio.to_thread(
                    self._client.eval,
                    _RENEW_SCRIPT,
                    1,
                    self.key,
                    self.token,
                    self.ttl_ms,
                )
            except redis.RedisError as exc:
                logger.warning(
                    "Mutation lease renewal failed: %s", type(exc).__name__
                )
                continue
            if not renewed:
                self._lease_lost = True
                logger.error(
                    "Lost the index mutation lease %s before the write finished; "
                    "another replica may now be writing",
                    self.key,
                )
                return

    def _close(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            client.close()
        except redis.RedisError:
            logger.debug("Unable to close the mutation lock Redis client")


def distributed_mutation_lock() -> DistributedMutationLock:
    """Create one per-request lease; no-op unless Redis is required."""
    return DistributedMutationLock(
        ttl_seconds=settings.MUTATION_LEASE_TTL_SECONDS,
        acquire_timeout_seconds=settings.MUTATION_LEASE_ACQUIRE_TIMEOUT_SECONDS,
    )
