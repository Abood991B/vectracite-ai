"""Prove that two API processes serialize conflicting writes through the Redis lease.

Start two API processes against the same Qdrant, Redis, and uploads directory, then run this
checker. It starts nothing itself: it drives the two servers and reads the shared stores.

    # terminal 1 and 2, from the repository root
    $env:QDRANT_HOST="127.0.0.1"; $env:QDRANT_PORT="6333"
    $env:QDRANT_IN_MEMORY="false"; $env:REQUIRE_REDIS="true"
    $env:REDIS_URL="redis://127.0.0.1:6379/0"; $env:AUTH_REQUIRED="false"
    $env:INDEX_SYNC_SECONDS="1"
    $env:MUTATION_LEASE_TTL_SECONDS="60"
    $env:MUTATION_LEASE_ACQUIRE_TIMEOUT_SECONDS="4"
    .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8081
    # repeat with --port 8082

    python scripts/multi_replica_check.py \
        --base-url http://127.0.0.1:8081 --base-url http://127.0.0.1:8082

Checks:
    1. a lease held by another process blocks every replica with a clean 503
    2. concurrent conflicting replaces leave exactly one complete version in Qdrant
    3. a concurrent delete and replace leave the document either absent or complete, never partial
    4. a crashed holder's lease expires and the next writer proceeds without a restart
"""

from __future__ import annotations

import argparse
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import httpx
import redis
from qdrant_client import QdrantClient, models

LEASE_KEY = "vectracite:index:mutation"
REQUEST_TIMEOUT_SECONDS = 180.0
MARKERS = ("ALPHA", "BETA", "GAMMA", "DELTA")


def _server_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> httpx.Response:
    with httpx.Client(timeout=timeout) as client:
        return client.request(method, f"{_server_url(base_url)}{path}", json=payload)


def _payload(doc_name: str, marker: str, pages: int) -> dict[str, Any]:
    return {
        "doc_name": doc_name,
        "pages": [
            {
                "page_number": index + 1,
                "text": (
                    f"{marker} segment {index + 1}: consolidated revenue was "
                    f"{1_000_000 + index * 25_000} thousand with a gross margin of 42 percent."
                ),
            }
            for index in range(pages)
        ],
    }


def _points(qdrant: QdrantClient, collection: str, doc_name: str) -> list[Any]:
    records, _ = qdrant.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_name", match=models.MatchValue(value=doc_name)
                )
            ]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    return records


def _summarize(points: Sequence[Any]) -> tuple[int, set[str], set[str]]:
    versions = {str(point.payload.get("version")) for point in points}
    markers = {
        marker
        for marker in MARKERS
        for point in points
        if marker in str(point.payload.get("text", ""))
    }
    return len(points), versions, markers


def check_ready(base_urls: Sequence[str]) -> None:
    for base_url in base_urls:
        response = _request(base_url, "GET", "/health", timeout=15.0)
        assert response.status_code == 200, f"{base_url}/health -> {response.status_code}"
    print(f"[ok] both servers answer /health: {', '.join(base_urls)}")


def check_held_lease_blocks_every_replica(
    base_urls: Sequence[str],
    broker: redis.Redis,
    acquire_timeout: float,
    probe_name: str,
) -> None:
    assert broker.set(LEASE_KEY, "external-holder", nx=True, px=30_000), (
        "the mutation lease was already held before the check started"
    )
    try:
        for base_url in base_urls:
            response = _request(
                base_url,
                "DELETE",
                f"/documents/{probe_name}",
                timeout=acquire_timeout + 30.0,
            )
            assert response.status_code == 503, (
                f"{base_url} wrote while the lease was held: {response.status_code} "
                f"{response.text}"
            )
            assert "lock" in response.text.lower(), response.text
    finally:
        broker.delete(LEASE_KEY)

    released = _request(base_urls[0], "DELETE", f"/documents/{probe_name}", timeout=30.0)
    assert released.status_code == 404, (
        f"after release the probe should reach the handler and 404, got "
        f"{released.status_code} {released.text}"
    )
    print("[ok] a held lease returns 503 on every replica; release restores the handler")


def check_concurrent_replaces(
    base_urls: Sequence[str],
    qdrant: QdrantClient,
    collection: str,
    doc_name: str,
    rounds: int,
    sync_wait: float,
) -> None:
    for round_index in range(rounds):
        first = _payload(doc_name, "ALPHA", 3)
        second = _payload(doc_name, "BETA", 2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    _request, base_urls[0], "PUT", f"/documents/{doc_name}", first
                ),
                pool.submit(
                    _request, base_urls[1], "PUT", f"/documents/{doc_name}", second
                ),
            ]
            responses = [future.result() for future in futures]

        for response in responses:
            assert response.status_code in {200, 503}, (
                f"round {round_index}: unexpected status {response.status_code} "
                f"{response.text}"
            )

        winners = [response for response in responses if response.status_code == 200]
        assert winners, f"round {round_index}: no replica completed the replace"

        time.sleep(sync_wait)
        points = _points(qdrant, collection, doc_name)
        count, versions, markers = _summarize(points)
        assert count in {2, 3}, f"round {round_index}: {count} points left behind"
        assert len(versions) == 1, f"round {round_index}: mixed versions {versions}"
        assert markers in ({"ALPHA"}, {"BETA"}), (
            f"round {round_index}: mixed document versions {markers}"
        )
        expected_count = 3 if markers == {"ALPHA"} else 2
        assert count == expected_count, (
            f"round {round_index}: {markers} should own {expected_count} points, got {count}"
        )

        for base_url in base_urls:
            view = _request(base_url, "GET", f"/documents/{doc_name}", timeout=60.0)
            assert view.status_code == 200, view.text
            assert view.json()["total_chunks"] == count, (
                f"{base_url} serves {view.json()['total_chunks']} chunks but Qdrant holds {count}"
            )
    print(
        f"[ok] {rounds} concurrent replace rounds kept one complete version in Qdrant "
        f"and both catalogs converged"
    )


def check_delete_versus_replace(
    base_urls: Sequence[str],
    qdrant: QdrantClient,
    collection: str,
    doc_name: str,
    sync_wait: float,
) -> None:
    seeded = _request(
        base_urls[0], "PUT", f"/documents/{doc_name}", _payload(doc_name, "GAMMA", 2)
    )
    assert seeded.status_code == 200, seeded.text

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _request,
                base_urls[1],
                "PUT",
                f"/documents/{doc_name}",
                _payload(doc_name, "DELTA", 3),
            ),
            pool.submit(_request, base_urls[0], "DELETE", f"/documents/{doc_name}"),
        ]
        responses = [future.result() for future in futures]

    for response in responses:
        assert response.status_code in {200, 404, 503}, (
            f"unexpected status {response.status_code} {response.text}"
        )

    time.sleep(sync_wait)
    points = _points(qdrant, collection, doc_name)
    count, versions, markers = _summarize(points)
    pdf = _request(base_urls[0], "GET", f"/documents/{doc_name}/pdf", timeout=60.0)
    if count == 0:
        assert pdf.status_code == 404, (
            "Qdrant is empty but the upload still exists, so a later restore would resurrect it"
        )
        print("[ok] delete won the race: no points and no upload file")
        return

    assert markers == {"DELTA"}, f"partial document after delete/replace: {markers}"
    assert count == 3 and len(versions) == 1, (
        f"replace won the race but left {count} points / versions {versions}"
    )
    assert pdf.status_code == 200, pdf.text
    print("[ok] replace won the race: one complete version and its upload file")


def check_expired_lease_is_reclaimed(
    base_url: str,
    broker: redis.Redis,
    acquire_timeout: float,
    probe_name: str,
) -> None:
    assert broker.set(LEASE_KEY, "crashed-holder", px=2_000), (
        "the mutation lease was already held before the crash check started"
    )
    started = time.monotonic()
    response = _request(
        base_url, "DELETE", f"/documents/{probe_name}", timeout=acquire_timeout + 30.0
    )
    waited = time.monotonic() - started
    assert response.status_code == 404, (
        f"a writer should reclaim an expired lease and reach the handler, got "
        f"{response.status_code} {response.text}"
    )
    assert 1.5 <= waited <= acquire_timeout + 5.0, (
        f"waited {waited:.2f}s for a 2s lease with a {acquire_timeout}s acquire budget"
    )
    print(f"[ok] an abandoned lease expired and the next writer proceeded after {waited:.1f}s")


def _discover_collection(
    qdrant: QdrantClient, prefix: str, override: str | None
) -> str:
    if override:
        return override
    names = [
        collection.name
        for collection in qdrant.get_collections().collections
        if collection.name.startswith(prefix)
    ]
    assert len(names) == 1, (
        f"expected exactly one collection starting with {prefix!r}, found {names}; "
        "pass --collection"
    )
    return names[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--qdrant-host", default="127.0.0.1")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection-prefix", default="financebench_hybrid_v1")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sync-wait", type=float, default=3.0)
    parser.add_argument("--acquire-timeout", type=float, default=4.0)
    args = parser.parse_args()

    base_urls = [_server_url(url) for url in args.base_url]
    assert len(base_urls) == 2, "pass exactly two --base-url values"

    broker = redis.Redis.from_url(args.redis_url, socket_timeout=5.0)
    qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, timeout=30.0)
    collection = _discover_collection(
        qdrant, args.collection_prefix, args.collection
    )
    suffix = uuid.uuid4().hex[:8].upper()
    doc_name = f"MRC_MULTI_REPLICA_{suffix}"
    probe_name = f"MRC_MISSING_DOCUMENT_{suffix}"

    print(f"collection={collection} doc={doc_name}")
    try:
        check_ready(base_urls)
        check_held_lease_blocks_every_replica(
            base_urls, broker, args.acquire_timeout, probe_name
        )
        check_concurrent_replaces(
            base_urls,
            qdrant,
            collection,
            doc_name,
            args.rounds,
            args.sync_wait,
        )
        check_delete_versus_replace(
            base_urls, qdrant, collection, doc_name, args.sync_wait
        )
        check_expired_lease_is_reclaimed(
            base_urls[0], broker, args.acquire_timeout, probe_name
        )
    except AssertionError as exc:
        print(f"[fail] {exc}")
        return 1
    finally:
        for base_url in base_urls:
            try:
                _request(base_url, "DELETE", f"/documents/{doc_name}", timeout=60.0)
            except httpx.HTTPError:
                pass
        if broker.get(LEASE_KEY) in {b"external-holder", b"crashed-holder"}:
            broker.delete(LEASE_KEY)
        qdrant.close()

    print("RESULT: multi-replica mutation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
