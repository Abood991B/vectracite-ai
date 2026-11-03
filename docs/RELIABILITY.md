# Reliability

## Integration-Point Audit

Every outbound call on the in-scope flows. "Breaker" means a circuit opens after consecutive failures
and lets a probe through after the recovery window.

| Dependency | Timeout | Circuit breaker | Bulkhead | Retry policy | Status |
|---|---|---|---|---|---|
| Qdrant (vectors) | 5 s client timeout | none | single client shared by all requests | none | partial — a Qdrant outage fails the request and `/ready` reports degraded; no breaker, so a hung Qdrant costs 5 s per call |
| Redis (semantic cache) | 1.5 s connect + read | self-disabling: the client is dropped on the first failure and the cache falls back to its bounded memory store | shared with the index counter | none | good — one timeout, then no further cost |
| Redis (index generation) | 1.0 s connect + read | 2 failures / 30 s (`src/api/index_state.py`) | shared | none | good — fail-fast to the process-local sync path |
| Redis (mutation lease) | 2.0 s connect + read (`LOCK_TIMEOUT_SECONDS`) | 2 failures / 30 s (`_LEASE_BREAKER` in `src/api/distributed_lock.py`) | shared with the cache and index counter | none | good — renewed every TTL/3 while a write is in flight; an unreachable Redis fails closed, and the breaker stops paying the 2 s timeout after two failures |
| Remote LLM (OpenAI-compatible) | connect 5 s, read 30 s | 2 failures / 30 s (`UnifiedLLMClient`) | one shared `httpx.AsyncClient` per event loop | none | good — a dead provider falls back to local synthesis after one connect timeout |
| Remote OCR service | 120 s (`OCR_TIMEOUT_SECONDS`) | none | ingestion slot: one parser at a time per process | none | partial — a slow OCR call holds the ingestion slot until it returns, even after the 120 s request timeout |
| MLflow | client default | none | evaluation path only, not a query path | none | out of scope for request latency |

## Query & Resource Findings

- **Unbounded result sets.** `/documents/{doc_name}` is paginated (`limit`/`offset`). `/documents` and
  `/scenarios` are bounded by corpus size and recorded as known limitations; the threshold to paginate is
  ~100 documents.
- **Blocked threads.** PDF parsing runs on a single-worker executor behind a `BoundedSemaphore(1)`, so
  a second upload receives 429 instead of queueing. A timed-out parse requests cancellation between
  pages; an OCR call already running cannot be interrupted and retains the slot.
- **Cache invalidation.** The semantic cache is cleared on every mutation and keyed by
  scope + top_k + sanitized question, in a namespaced key space. Answer entries carry a TTL.
- **Derived data.** Chunks, embeddings, the BM25 index, vectors, and cached answers are all derived from
  the PDF corpus and rebuild from it; the Qdrant collection name is versioned so a numbering change
  cannot reuse stale vectors.

## Health Checks & Metrics

- `GET /health` — liveness only: the process is up.
- `GET /ready` — deep check: vector backend reachable and not degraded; Redis required only when
  `REQUIRE_REDIS=true`. An empty catalog is ready, because uploading is a valid first action.
- `GET /metrics` — indexed chunks and documents, semantic-cache stats (backend, hit rate, degraded,
  last error), model identifiers, LLM provider status, retriever health.
- **Gap (ticketed):** there are no RED metrics (rate, errors, duration per endpoint) and no symptom
  alerts. The Phase 1 budgets are measured by `scripts/bench_hot_paths.py` on demand, not guarded in
  production. At demo scale this is acceptable; Phase 8 decides whether a budget becomes a CI gate.

## Deploy vs Release

- The image is built once and deployed; the model weights are preloaded in the Docker image.
- Configuration is environment-driven (`AUTH_REQUIRED`, `REQUIRE_REDIS`, `ALLOW_QDRANT_FALLBACK`,
  `LLM_REMOTE_ENABLED`, `OCR_ENABLED`), so a release can be rolled back by restoring the previous
  environment rather than rebuilding.
- Startup fails closed on enabled-but-misconfigured authentication, and refuses to start with a
  process-local index when shared Qdrant is expected.
- Kubernetes runs two replicas behind readiness checks; uploads require `ReadWriteMany` or a
  single-replica deployment. `REQUIRE_REDIS=true` (both Compose and the manifests) enables the
  cross-replica mutation lease.
- **Mutation lease.** Every mutation takes a Redis lease (`MUTATION_LEASE_TTL_SECONDS`, default
  300 s, acquire budget `MUTATION_LEASE_ACQUIRE_TIMEOUT_SECONDS`, default 30 s). A heartbeat renews
  it every third of the TTL while the write is in flight, so a commit that outlives one TTL cannot be
  overtaken by a second replica. A crashed holder stops renewing, and its lease expires for the next
  writer.
- **Evidence, 2026-11-03.** `scripts/multi_replica_check.py` against two uvicorn processes sharing
  Qdrant, Redis, and `data/uploads`, run twice (once before and once after the breaker commit): a
  lease held by an external process made both replicas return 503 (and the handler resumed after
  release); three concurrent replace rounds each left exactly one version in Qdrant and both catalogs
  converged; the concurrent delete-vs-replace check saw the replace win two of three runs and the
  delete one, each leaving one clean state (three points + upload, or no points + no upload); an
  abandoned 2 s lease was reclaimed in 2.0 s, 2.7 s, and 2.9 s across three runs. A real-Redis test
  holds the lease for 2.5× its TTL and proves no intruder can take it
  (`test_heartbeat_keeps_a_real_redis_lease_past_its_ttl`), and an HTTP-level test slows
  `_commit_document` past a 1 s TTL and proves the lease is still held at the end of the route
  (`test_a_slow_commit_keeps_the_lease_across_the_http_route`).
- **Measured breaker effect, 2026-11-03.** With Redis unreachable, four consecutive mutations each
  paid **2.000–2.031 s** before the 503. With the breaker, the first two pay 2.03 s + 2.00 s and the
  next two return in **0.0 s**.
- **Fencing and reconcile.** Acquisition also assigns a monotonic `commit_epoch` (a Redis `INCR`
  beside the lease), and every point carries it. A replace or delete only removes points with
  `commit_epoch <=` its own, so a stale holder cannot destroy a newer commit; hydration keeps the
  maximum epoch per document and deletes the rest, so a crash between the upsert and the stale-point
  delete converges on the next sync. Duplicates written before the epoch existed are kept and logged
  rather than guessed away, and the next replace of that document clears them.
- **Fence and reconcile evidence, 2026-11-03.** Nine deterministic tests in
  `tests/test_commit_fencing.py` cover a stale replace, a stale delete, a current delete, a stale
  clear, a current clear, the crash-mid-commit kill point, a versioned commit over bulk-index points,
  idempotent hydration, and tied legacy versions. A live spot check against server Qdrant (throwaway
  collection) confirmed both payload indexes (`doc_name`, `commit_epoch`), deleted the older version
  during hydration, and kept the epoch-8 points when an epoch-7 replace ran afterwards.
- **Collection-wide clears.** `clear` no longer drops the collection: it deletes points committed
  up to the caller's epoch and keeps newer points, so a stale clear cannot erase a newer restore.
  The routes count survivors and return 409 without deleting upload files. Keeping the collection
  also preserves its payload indexes.
- **Residual.** A stale write can still add points that one replica serves until its next hydrate
  reconciles them; the window is one sync interval. No concurrency path is unfenced at this scale.

### Deferred with triggers

No deferred concurrency gap remains at this scale. The next capacity triggers (replicas, QPS,
chunks) are recorded in `PERFORMANCE.md` `### Capacity triggers`.
