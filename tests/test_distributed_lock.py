"""Pin the cross-replica mutation lease.

The lease is the only thing serializing conflicting writes between API processes that share
Qdrant, Redis, and the uploads directory. These tests cover the four behaviors a deployment
depends on: exclusivity while held, renewal during a slow write, expiry after a crash, and a
clean 503 instead of a lost update when the lease is unavailable or taken over.
"""

import asyncio
import time
import uuid

import pytest
import redis
from fastapi import HTTPException

from src.api.distributed_lock import (
    _LEASE_BREAKER,
    DistributedMutationLock,
    distributed_mutation_lock,
)
from src.config import settings

LOCK_KEY = "vectracite:index:mutation"


@pytest.fixture(autouse=True)
def _reset_lease_breaker():
    """Each test starts from a closed breaker; failures must not leak between tests."""
    _LEASE_BREAKER.reset()
    yield
    _LEASE_BREAKER.reset()


class FakeRedis:
    """Minimal Redis double: NX/PX SET, compare-and-pexpire, compare-and-delete, real TTL."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expires_at: dict[str, float] = {}
        self.set_attempts = 0
        self.set_calls = 0
        self.renew_calls = 0
        self.release_calls = 0
        self.fail_set = False
        self.fail_renew = False
        self.lose_lease_on_renew = False
        self.closed = False

    def _purge(self, key: str) -> None:
        expires_at = self.expires_at.get(key)
        if expires_at is not None and time.monotonic() >= expires_at:
            self.values.pop(key, None)
            self.expires_at.pop(key, None)

    def hold_key(self, key: str, token: str, seconds: float) -> None:
        self.values[key] = token
        self.expires_at[key] = time.monotonic() + seconds

    def set(self, key, value, nx=False, px=None):
        self.set_attempts += 1
        if self.fail_set:
            raise redis.RedisError("connection refused")
        self.set_calls += 1
        self._purge(key)
        if nx and key in self.values:
            return None
        self.values[key] = value
        if px is not None:
            self.expires_at[key] = time.monotonic() + px / 1000.0
        return True

    def eval(self, script, numkeys, key, *args):
        del numkeys
        if "pexpire" in script:
            self.renew_calls += 1
            if self.fail_renew:
                raise redis.RedisError("connection reset")
            if self.lose_lease_on_renew:
                self.values[key] = "another-replica"
                return 0
            self._purge(key)
            if self.values.get(key) != args[0]:
                return 0
            self.expires_at[key] = time.monotonic() + args[1] / 1000.0
            return 1
        self.release_calls += 1
        self._purge(key)
        if self.values.get(key) == args[0]:
            del self.values[key]
            self.expires_at.pop(key, None)
            return 1
        return 0

    def close(self) -> None:
        self.closed = True


def test_lock_is_a_noop_when_redis_is_not_required(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_REDIS", False)
    fake = FakeRedis()

    async def scenario() -> None:
        async with DistributedMutationLock(client_factory=lambda: fake) as lease:
            lease.require_lease()

    asyncio.run(scenario())
    assert fake.set_calls == 0
    assert fake.release_calls == 0
    assert not fake.closed


def test_held_lease_is_renewed_while_a_slow_write_runs(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_REDIS", True)
    fake = FakeRedis()

    async def scenario() -> None:
        async with DistributedMutationLock(
            ttl_seconds=0.3, client_factory=lambda: fake
        ) as lease:
            deadline = time.monotonic() + 2.0
            while fake.renew_calls < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert fake.renew_calls >= 2
            assert not lease.lease_lost
            lease.require_lease()

    asyncio.run(scenario())
    assert fake.renew_calls >= 2
    assert fake.release_calls == 1
    assert fake.closed
    assert fake.values == {}


def test_release_never_deletes_another_replicas_lease(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_REDIS", True)
    fake = FakeRedis()

    async def scenario() -> None:
        async with DistributedMutationLock(ttl_seconds=30, client_factory=lambda: fake):
            fake.values[LOCK_KEY] = "another-replica"

    asyncio.run(scenario())
    assert fake.values[LOCK_KEY] == "another-replica"
    assert fake.release_calls == 1


def test_a_lost_lease_blocks_the_write_and_is_not_reacquired(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_REDIS", True)
    fake = FakeRedis()
    fake.lose_lease_on_renew = True

    async def scenario() -> None:
        async with DistributedMutationLock(
            ttl_seconds=0.3, client_factory=lambda: fake
        ) as lease:
            deadline = time.monotonic() + 2.0
            while not lease.lease_lost and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert lease.lease_lost
            with pytest.raises(HTTPException) as blocked:
                lease.require_lease()
            assert blocked.value.status_code == 503
            assert "lease was lost" in blocked.value.detail

    asyncio.run(scenario())
    assert fake.values[LOCK_KEY] == "another-replica"


def test_a_crashed_holder_lease_expires_for_the_next_writer(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_REDIS", True)
    fake = FakeRedis()

    async def hold_then_crash() -> None:
        # No __aexit__: the loop closes with the heartbeat cancelled, which is what a process
        # crash leaves behind — a lease in Redis that nobody is renewing.
        await DistributedMutationLock(
            ttl_seconds=0.2, client_factory=lambda: fake
        ).__aenter__()

    asyncio.run(hold_then_crash())
    assert fake.values[LOCK_KEY]
    time.sleep(0.3)

    async def take_over() -> None:
        async with DistributedMutationLock(ttl_seconds=5, client_factory=lambda: fake):
            pass

    asyncio.run(take_over())
    assert fake.set_calls == 2
    assert fake.values == {}


def test_acquire_timeout_is_a_clean_503_and_closes_the_client(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_REDIS", True)
    fake = FakeRedis()
    fake.hold_key(LOCK_KEY, "another-replica", seconds=60)
    lock = DistributedMutationLock(
        acquire_timeout_seconds=0.1, client_factory=lambda: fake
    )

    with pytest.raises(HTTPException) as blocked:
        asyncio.run(lock.__aenter__())
    assert blocked.value.status_code == 503
    assert "Timed out" in blocked.value.detail
    assert fake.closed


def test_redis_outage_during_acquire_is_a_clean_503(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_REDIS", True)
    fake = FakeRedis()
    fake.fail_set = True

    with pytest.raises(HTTPException) as blocked:
        asyncio.run(
            DistributedMutationLock(client_factory=lambda: fake).__aenter__()
        )
    assert blocked.value.status_code == 503
    assert "unavailable" in blocked.value.detail
    assert fake.closed


def test_breaker_skips_the_redis_timeout_after_repeated_failures(monkeypatch) -> None:
    """Two failed acquisitions open the breaker, so the third never pays the connect timeout."""
    monkeypatch.setattr(settings, "REQUIRE_REDIS", True)
    fake = FakeRedis()
    fake.fail_set = True

    async def scenario() -> list[int]:
        statuses = []
        for _ in range(3):
            with pytest.raises(HTTPException) as blocked:
                await DistributedMutationLock(
                    client_factory=lambda: fake
                ).__aenter__()
            statuses.append(blocked.value.status_code)
        return statuses

    assert asyncio.run(scenario()) == [503, 503, 503]
    assert fake.set_attempts == 2
    assert not _LEASE_BREAKER.allow()


def test_a_failed_renewal_keeps_the_lease_and_retries(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REQUIRE_REDIS", True)
    fake = FakeRedis()
    fake.fail_renew = True

    async def scenario() -> None:
        async with DistributedMutationLock(
            ttl_seconds=0.3, client_factory=lambda: fake
        ) as lease:
            deadline = time.monotonic() + 2.0
            while fake.renew_calls < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert fake.renew_calls >= 2
            assert not lease.lease_lost
            lease.require_lease()

    asyncio.run(scenario())
    assert fake.renew_calls >= 2


def test_factory_reads_the_configured_lease_budget(monkeypatch) -> None:
    monkeypatch.setattr(settings, "MUTATION_LEASE_TTL_SECONDS", 45.0)
    monkeypatch.setattr(settings, "MUTATION_LEASE_ACQUIRE_TIMEOUT_SECONDS", 3.0)

    lock = distributed_mutation_lock()
    assert lock.ttl_ms == 45_000
    assert lock.acquire_timeout_seconds == 3.0


@pytest.mark.integration
def test_heartbeat_keeps_a_real_redis_lease_past_its_ttl(monkeypatch) -> None:
    """Without renewal the key is gone after one TTL; with it, no intruder can take it.

    This is the check that separates the heartbeat from a fixed TTL: the holder sleeps 2.5
    lease lifetimes and an unrelated ``SET NX`` must still fail on every attempt.
    """
    monkeypatch.setattr(settings, "REQUIRE_REDIS", True)
    broker = redis.Redis.from_url(
        settings.REDIS_URL, socket_connect_timeout=0.5, socket_timeout=2.0
    )
    try:
        broker.ping()
    except redis.RedisError:
        pytest.skip(f"Redis is not reachable at {settings.REDIS_URL}")

    key = f"vectracite:test:mutation:{uuid.uuid4().hex}"

    async def scenario() -> None:
        async with DistributedMutationLock(
            key=key,
            ttl_seconds=1.0,
            acquire_timeout_seconds=2.0,
            client_factory=lambda: redis.Redis.from_url(
                settings.REDIS_URL, socket_timeout=2.0
            ),
        ) as lease:
            assert broker.set(key, "intruder", nx=True, px=5000) is None
            await asyncio.sleep(2.5)
            assert not lease.lease_lost
            assert broker.set(key, "intruder", nx=True, px=5000) is None
            lease.require_lease()

    try:
        asyncio.run(scenario())
        assert broker.get(key) is None
    finally:
        broker.delete(key)
        broker.close()
