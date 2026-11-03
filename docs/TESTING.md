# Testing

## Test Strategy

Pytest over a four-layer pyramid. The default suite uses the four small synthetic PDFs in
`tests/fixtures/corpus_pdfs` for deterministic unit and API tests; real PDFs, real BGE models, and
sustained evaluation live behind the `integration` marker.

| Layer | What it pins | Files |
|---|---|---|
| Unit | Parsing, table AST, financial numbers, verifier, cache keys, guardrails | `test_pipeline.py`, `test_ocr.py`, `test_regressions.py` |
| API contract | Routes, auth, limits, SSE framing, CRUD, UI asset contract | `test_pipeline.py`, `test_phase3.py` |
| Regression | Owner-boundary behavior: citations, cache namespaces, page numbering, scope inference | `test_regressions.py` |
| Integration | Real corpus, real retrieval quality, domain benchmarks | `test_real_corpus.py`, `test_domain_benchmark.py` (`-m integration`) |

"Green" means the default suite passes and `ruff` is clean. Integration tests gate releases, not the
per-change loop.

Measured feedback cost, 2026-11-02. Before the cache fix the default `pytest -q` took ~54 minutes: two
domain tests cost 619 s each and `test_real_corpus.py` cost 1663 s, all re-parsing the corpus. After the
fix the suite dropped to 149.6 s for 97 tests. Measured again on 2026-11-03: **146 passed in 67.3 s**
(Redis up, so the real-Redis lease test runs; it skips without Redis). The full run on 2026-11-04
was **194 passed, 2 skipped in 61.1 s**. `tests/test_corpus_cache.py` fails in 0.07 s if a checked-in
cache goes stale again.

## Safety Net Map

| Module | Pinned behaviors | Test files | Gaps |
|---|---|---|---|
| `src/api/answer_service.py` | Guarded pipeline end to end: cache lookup and hit marking, retrieval, synthesis, citation filtering, numeric verification, latency field, streamed redaction then final response | `tests/test_answer_service.py`, `test_pipeline.py` (query/stream e2e), `test_regressions.py` | `_build_response` empty-`answer_chunks` branch still pinned only through HTTP |
| `src/api/main.py` — query routes | JSON answer shape, SSE event order, redaction at the wire, verification event | `test_pipeline.py`, `test_phase3.py` | `/metrics` contract untested |
| `src/api/main.py` — ingest | Size/page limits, base64 decode errors, structured ingest, streaming progress, cancellation slot, commit | `test_pipeline.py`, `test_regressions.py`, `test_ocr.py` | Parse-timeout path exercised only through the slot test; parse throughput not asserted |
| `src/api/main.py` — startup | `_ensure_indexed` hydration, demo clearing, rebuild behavior | `test_regressions.py` | No lifespan/cold-start assertion; stale-cache gate added below |
| `src/retrieval/hybrid_retriever.py` | Dense + BM25 + weighted RRF, scope filtering, document CRUD, shared Qdrant catalog, commit-epoch fence, epoch-scoped clear, duplicate-version reconcile | `test_pipeline.py`, `test_regressions.py`, `test_domain_benchmark.py`, `test_commit_fencing.py` | `sync_from_qdrant` generation path untested |
| `src/retrieval/reranker.py` | Adaptive gate preserves ranking; SCAR aligns table evidence first | `test_pipeline.py`, `test_domain_benchmark.py` | Forced-gate mode only in the domain harness; GPU path unvalidated |
| `src/cache_and_guardrails/semantic_cache.py` | Exact + HNSW lookup, entity/year/top-k mismatch rejection, namespace migration, degraded health reporting | `test_pipeline.py`, `test_regressions.py`, `test_phase3.py` | Redis-absence latency cost is measured in `PERFORMANCE.md` but not asserted |
| `src/ingestion/pdf_parser.py` | Table preservation, section chunking, physical-page citations, coverage reporting, OCR limits, cancellation | `test_pipeline.py`, `test_ocr.py`, `test_regressions.py` | No performance assertion on parse cost |
| `src/api/ui.py` — workbench | Asset and element contract, streaming upload endpoint, zoom/inspector controls | `test_pipeline.py` | No browser-level test; validated manually through `agent-browser` |
| `src/api/distributed_lock.py` | Acquire and timeout, renewal while held, lost-lease 503, release only own token, crash expiry, failed-acquire client close, breaker skip after two failures, real-Redis renewal across 2.5 TTLs; HTTP route holds the lease past a 1 s TTL | `tests/test_distributed_lock.py`, `tests/test_pipeline.py`, `scripts/multi_replica_check.py` | The two-process check needs live Qdrant/Redis; the breaker recovery window is covered in `test_resilience.py`, not here |
| `src/usecases/answer_pipeline.py` | Evidence selection shared by JSON and SSE: per-document comparison retrieval, table-candidate augmentation, scope filtering | `tests/test_answer_service.py`, `tests/test_comparison_evidence.py`, `test_pipeline.py` | Rank within evidence stays low for comparison questions; the labelled sets grade citation and value, not rank |
| `src/usecases/synthesis.py` | Table answer selection (statement-line preference and metric family), comparison blocks, narrative scoring, query terms | `tests/test_comparison_evidence.py`, `test_answer_service.py`, `test_regressions.py`, `test_narrative_synthesis.py` | 3M 2022's income statement extracts as plain text, so it is answered from the statement-line path |
| `/evaluate` | Persisted report when the signature matches; domain benchmark reports | `test_pipeline.py`, `test_domain_benchmark.py` | `POST /evaluate` response shape not pinned |

## Characterization Backlog

- [x] `AnswerService` direct unit pins — `tests/test_answer_service.py` covers cache hits, cache stores, citation filtering, streamed redaction, and injection blocking
- [x] `POST /evaluate` response contract — pinned in `tests/test_api_contracts.py`
- [x] `/metrics` payload contract — pinned in `tests/test_api_contracts.py`
- [ ] Cold-start budget assertion in CI (risk: medium, priority: medium — needs a stable machine class)

## CI Gates

| Gate | Command | Budget | Status |
|---|---|---|---|
| Default suite | `pytest -q` | ≤ 10 min | implemented - **194 passed, 2 skipped in 61.1 s** (2026-11-04) |
| Documentation links | `python scripts/check_docs.py` | 0 broken or unpublished targets | implemented - runs in CI |
| Lint | `ruff check .` | 0 findings | implemented - runs in CI |
| Integration suite | `pytest -q -m integration` | informational | implemented — 23.7 s cached |
| Checked-in corpus caches match the current parser and PDF hashes | `pytest tests/test_corpus_cache.py` | ≤ 5 s | implemented — 0.07 s |
| Dependency Rule holds | `python scripts/check_layers.py` | 0 inward violations, 0 cycles | implemented — 0 / 0, enforced in CI |
| Hot-path budgets | `python scripts/bench_hot_paths.py --mode inproc --gate` | generous CI caps in `scripts/bench_hot_paths.py`; real budgets in `PERFORMANCE.md` | implemented — CI runs 3 iterations and fails on a catastrophic regression |
| Cross-replica mutation lease | `python scripts/multi_replica_check.py --base-url … --base-url …` | 4 checks (held lease, concurrent replace, delete vs replace, expired lease) | manual — needs two live processes plus shared Qdrant and Redis; run it after touching the mutation path |
