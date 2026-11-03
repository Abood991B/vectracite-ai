# Performance

Measured performance state for the repository's six hot paths.
Tool of record: `scripts/bench_hot_paths.py`, run in two modes — `inproc` (in-memory Qdrant,
lightweight models, fixture corpus; deterministic) and `http` (a live server with real BGE models and
the 830-page corpus). Numbers are only comparable within the same mode.

## Baselines & Budgets

Live-HTTP baseline, 2026-11-04, 10 iterations after 2 warmups, real `BAAI/bge-small-en-v1.5` and
`bge-reranker-base` on CPU, in-memory Qdrant, Redis absent.

| Metric (p50/p95, throughput, cost) | Baseline (date) | Budget | Gate |
|---|---|---|---|
| `/query` p95, cache miss | 194.36 ms (2026-11-04) | ≤ 250 ms | CI budget gate (inproc) + dashboard |
| `/query` p95, cache hit | 2.29 ms (2026-11-04) | ≤ 25 ms | CI budget gate (inproc) |
| `/query/stream` time-to-first-byte p95 | 206.93 ms (2026-11-04) | ≤ 500 ms | CI budget gate (inproc) + dashboard |
| `/ingest/pdf` 5-page p95 | 1875.52 ms (2026-11-04) | ≤ 3500 ms | CI budget gate (inproc) + dashboard |
| Cold start → first answered request | 660 s before cache fix; **20.0 s** (2026-11-04) | ≤ 120 s | manual / dashboard (process start is not timed in CI) |
| `/evaluate/latest` p95 | 17.38 ms (2026-11-04) | ≤ 50 ms | CI budget gate (inproc) |
| Workbench `GET /` p95 | 1.76 ms (2026-11-04) | ≤ 20 ms | CI budget gate (inproc) |
| Default `pytest` suite | ~54 min before cache fix; 149.6 s for 97 tests (2026-11-02); **67.3 s for 146 tests** (2026-11-03); **192 passed, 2 skipped in 62.1 s** (2026-11-04) | ≤ 10 min | CI gate |

In-process baseline, same date, 15 iterations after 3 warmups, lightweight models, fixture corpus.
Kept because it isolates code cost from model and corpus cost:

| Metric | p50 | p95 |
|---|---:|---:|
| Cold start → first answer | 1798.99 ms | — |
| `/query` cache miss | 9.96 ms | 12.49 ms |
| `/query` cache hit | 6.70 ms | 8.46 ms |
| `/query/stream` TTFB | 7.90 ms | 11.21 ms |
| `retrieve_hybrid` | 0.89 ms | 1.39 ms |
| `rerank` | 0.36 ms | 0.41 ms |
| `/ingest/pdf` 5-page | 4159.68 ms | 4215.82 ms |
| Workbench `GET /` | 5.62 ms | 6.79 ms |
| `/evaluate/latest` | 6.79 ms | 14.33 ms |

Reproduce:

```bash
python scripts/bench_hot_paths.py --mode inproc --iterations 15
python scripts/bench_hot_paths.py --mode http --base-url http://127.0.0.1:8080 --iterations 10
```

## Load Reality

Single-user / demo deployment: one operator, a handful of queries per session, the five checked-in SEC
filings (830 pages, 4,154 chunks), and occasional PDF uploads. No sustained QPS and no SLA.
Conflicting writers are exercised only by the two-process lease check
(`scripts/multi_replica_check.py`), not under load. QPS and growth rate are therefore unmeasured by
design — this is the single source for load numbers, and `ARCHITECTURE.md` cites it rather than
repeating them.

Consequence for later phases: an algorithm or query fix can earn its keep; a cache tier, replica, or
shard must clear a high bar because there is no measured load for it to serve.

### Capacity triggers

Phase 5 shipped no infrastructure. These are the measured conditions that would change that, so the
decision is revisited on evidence rather than mood:

| Trigger (measured) | Move |
|---|---|
| Sustained QPS > 5, or `/query` p95 > 250 ms with CPU saturated | Vertical headroom first, then a second API replica |
| More than one concurrent upload sustained (the single-slot semaphore returns 429) | Move ingestion behind a job queue |
| More than 1,000,000 chunks, or Qdrant search p95 > 100 ms | Segment or shard the collection |
| Cache hit rate below 30% after warm-up | Revisit cache keys and threshold before adding a tier |

## Profile Findings

| Hotspot | Evidence (profiler/APM) | Suspected cause | Fix | Status |
|---|---|---|---|---|
| Cold start ≈ 660 s | Server log, 2026-11-02: 3M_2018 221.36 s / 160 pages, 3M_2022 164.38 s / 252 pages, ACTI 108.42 s / 198 pages, ADOBE 54.39 s / 99 pages, AMD 58.79 s / 121 pages | `PARSER_CACHE_VERSION` moved 2 → 4 while the checked-in `data/corpus_index.json` stayed at version 2, so every start re-parsed all 830 pages (~0.73 s/page) | Regenerated the cache; added a freshness gate so a stale cache fails fast. Measured after: 23.3 s | fixed |
| Test suite ≈ 54 min | `--durations`: 619 s × 2 domain integration tests, 1663 s `test_real_corpus.py` | `_cache_path_for_corpus` wrote the domain cache into the per-test temp `DATA_DIR`, so 9 corpus PDFs were re-parsed on every run and never cached | Cache beside the corpus directory, write by default. Measured after: 149.6 s for 97 tests | fixed |
| `/ingest/pdf` ≈ 3–4 s per 5-page file | bench, both modes | Dominated by per-ingest re-parse plus Qdrant upsert; 1 s of it is a Redis connect timeout when Redis is absent | Not yet attacked | open |
| Redis absence costs 1 s per call | `INFO:src.api.index_state:Index generation unavailable: TimeoutError` on every mutation and sync tick | `redis.from_url(socket_connect_timeout=1.0)` per call, no breaker | Phase 7 added a 2 failures / 30 s breaker in `index_state`; the sync path fails fast | fixed |
| Mutation lock while Redis is unreachable costs 2 s per call | Four consecutive acquisitions against a closed port each measured **2.000–2.031 s** (2026-11-03) | `LOCK_TIMEOUT_SECONDS` paid on every attempt; only the index counter had a breaker | Breaker on the lease (2 failures / 30 s). After: first two attempts 2.03 s + 2.00 s, next two **0.0 s** | fixed |
| Retrieval is cheap relative to synthesis | inproc: `retrieve_hybrid` 0.89 ms, `rerank` 0.36 ms vs `/query` 9.96 ms | Query time is dominated by embedding and deterministic synthesis, not fusion | No action | n/a |
| Query path has no infrastructure-shaped bottleneck | Same split: retrieval is ~13% of query time; cache hits land at 2.3 ms live | The remaining cost is model inference and synthesis, which a cache tier, queue, or replica would not reduce | No infrastructure move; capacity triggers recorded under Load Reality | n/a |
| Every live query cost ≈2.1 s | Direct Qdrant client: **2042 ms via `localhost`, 8 ms via `127.0.0.1`** | Windows resolves `localhost` to `::1` first; Qdrant binds IPv4, so each search paid a connection fallback | Defaults changed to `127.0.0.1` in `src/config.py` and `.env.example`; pinned by a test | fixed |
| Load behavior | 30 s at concurrency 4: **6.6 rps, p95 756 ms, 0 errors** uncached; 20 s cached: **685 rps, p95 5.9 ms, 0 errors** (2026-11-04) | CPU-bound embedding and the cross-encoder under concurrency; the in-memory cache absorbs repeated work | Recorded; `scripts/load_test.py` is the tool | measured |

## Optimization Ledger

| Change | Before | After | Verdict (keep/revert) | Date |
|---|---|---|---|---|
| Regenerate `data/corpus_index.json` + `corpus_embeddings.npz` for `PARSER_CACHE_VERSION` 4 | 660 s cold start | 23.3 s cold start | keep | 2026-11-02 |
| Domain corpus cache beside the corpus dir + `write_cache=True` | 1663 s `test_real_corpus.py`, 619 s × 2 domain tests | 22.6 s `test_real_corpus.py`; domain tests 0.10 s + 6.22 s | keep | 2026-11-02 |
| Fix stale workbench assertion (`/ingest/pdf` → `/ingest/pdf/stream`) | 1 red test | suite green | keep | 2026-11-02 |
| Full default suite after both cache fixes | ~54 min (est. 3240 s) | 149.6 s, 97 tests | keep | 2026-11-02 |
| Mutation lease circuit breaker (2 failures / 30 s) | 2.0 s per mutation while Redis is unreachable (4/4 probes at 2.000–2.031 s) | 2.03 s + 2.00 s for the first two probes, 0.0 s for the next two (breaker open) | keep | 2026-11-03 |
