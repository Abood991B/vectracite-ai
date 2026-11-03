# Architecture

## System Context

VectraCite-AI answers questions over a real SEC-filing corpus. PyMuPDF extracts page-cited text and
tables; a hybrid retriever (dense BGE + BM25 + weighted RRF) proposes evidence; an adaptive SCAR
reranker reorders only when a table-AST signal is strong; a synthesizer produces a cited answer that a
deterministic verifier checks for numeric presence; a semantic cache serves repeat questions; guardrails
block injection and redact PII.

Key integrations: Qdrant (vectors), Redis Stack (cache and index generation), MLflow (evaluation
tracking), an optional OpenAI-compatible LLM endpoint, an optional OCR service, and the checked-in
corpus caches on disk. Load reality is single-user demo scale — see `PERFORMANCE.md` `## Load Reality`;
this document does not repeat those numbers.

## Layer Map & Dependency Rule

Measured with `scripts/check_layers.py` (2026-11-04): **49 modules, 0 inward violations, 0 dependency
cycles**. Increment 1 moved `DocumentChunk` and `RetrievedDocument` into `src/domain/`
(V2). Increment 2 moved the answer policy to `src/usecases/` behind owned ports (V1, V3, V11).
Increment 3 moved scope inference and scenario building out of the controller and put the evaluation
harness behind an adapter (V9, V10 partial). Increment 4 stopped the two entity modules from reading
deployment settings (V4 partial, V12). Increment 5 moved the OCR contract into the domain and the
document lifecycle into `src/usecases/documents.py` behind a file store adapter (V5, V9 remainder).
Increment 7 moved the workbench document to a static asset (V13). Increment 6a moved deterministic
synthesis into `src/usecases/synthesis.py` (V7 partial). Phase 3 removed the last controller-side
vendor code and consolidated duplicated decisions (`citations`, `redis_clients`, `page_images`,
`count_pdf_pages`), closing V10.

**Adapters still read `src.config` directly.** Both sit in the outer ring, so the Dependency Rule
permits the edge; it is a testability smell, not a rule breach, and is ledgered for Phase 4.
**No entity or use-case module depends on anything outward.**

| Layer | Modules | Declared dependencies | Actual dependencies |
|---|---|---|---|
| Entities | `domain/documents.py`, `domain/ocr.py`, `retrieval/financial_numbers.py`, `retrieval/table_ast.py`, `inference/schemas.py` | nothing outward | nothing outward |
| Use cases | `usecases/answer_pipeline.py`, `usecases/synthesis.py`, `usecases/documents.py`, `usecases/scoping.py`, `usecases/scenarios.py`, `cache_and_guardrails/numeric_verifier.py`, `cache_and_guardrails/guardrails.py` | entities + `usecases/ports.py` | nothing outward — ports only |
| Interface adapters | `retrieval/hybrid_retriever.py`, `reranker.py`, `embedding_cache.py`, `cache_and_guardrails/semantic_cache.py`, `inference/llm_client.py`, `ingestion/pdf_parser.py`, `ocr.py`, `corpus_cache.py`, `dataset_loader.py`, `adapters/uploads.py`, `adapters/page_images.py`, `adapters/redis_clients.py`, `adapters/evaluation_runner.py` | entities | `config` (ledgered), sibling adapters |
| Frameworks & drivers | `api/main.py`, `ui.py` (loader), `static/workbench.html`, `middleware.py`, `security.py`, `distributed_lock.py`, `index_state.py`, `config.py` | everything | — |

Nothing points back outward from `config` and `financial_numbers`; those are the stable roots
(fan-in 15 and 3 respectively). The violations are all in the other direction.

| Violation | Location | Fix | Status |
|---|---|---|---|
| V1: the answer use case imports `fastapi.HTTPException`, so blocking a query raises an HTTP error from inside the pipeline | `src/api/answer_service.py` | Move the pipeline to `src/usecases/`; raise a domain `QueryBlocked`; the controller maps it to 400 | fixed 2026-11-02 — `src/usecases/answer_pipeline.py` + `QueryBlockedError`; `src/api/answer_service.py` is a thin 400-mapping adapter |
| V2: a use case imported an adapter type (`RetrievedDocument` lived in `retrieval/hybrid_retriever.py`) | `src/cache_and_guardrails/numeric_verifier.py` | Move the evidence DTO to `src/domain/` and import it from there | fixed 2026-11-02 — `src/domain/documents.py`; violations 12 → 11 |
| V3: the pipeline depends on concrete `HybridRetriever`, `SemanticCache`, and `UnifiedLLMClient`; there are no ports | `src/api/answer_service.py` | Define `RetrieverPort`, `AnswerCachePort`, `SynthesizerPort`, `VerifierPort` inward | fixed 2026-11-02 — `src/usecases/ports.py`; adapters satisfy the protocols structurally |
| V4: 9 modules import the `settings` singleton, including two entity modules | `pdf_parser.py`, `schemas.py`, 7 adapters | Inject settings at the composition root; entities take explicit parameters | partly fixed 2026-11-02 — entities no longer import settings; 8 adapters still do (outer-ring, ledgered as a testability smell) |
| V5: an entity module imports an adapter (`pdf_parser` → `ingestion/ocr.py`) | `src/ingestion/pdf_parser.py` | Inject an `OcrPort`; the OCR module implements it | fixed 2026-11-02 — `src/domain/ocr.py` owns the contract; the parser imports only that |
| V6: `DocumentChunk` and table logic are constructed inside PyMuPDF page handling | `src/ingestion/pdf_parser.py` | Extract pure chunking/tabulation functions over text; keep PyMuPDF in an adapter | partly fixed 2026-11-02 — `_parse_page` is now an orchestrator over `_page_text`, `_manifest_blocks`, and `_compose_content`; native table detection still needs the page object |
| V7: one module is both provider adapter and synthesis use case (httpx transport, prompt building, deterministic synthesis, table formatting) | `src/inference/llm_client.py` | Split into `LocalSynthesizer` (use case), `RemoteLlmSynthesizer` (adapter), provider client | partly fixed 2026-11-02 — deterministic synthesis moved to `src/usecases/synthesis.py`; the remaining module is the provider adapter plus the local/remote choice |
| V8: one adapter owns BM25 state, Qdrant transport, embedding, and the document catalog | `src/retrieval/hybrid_retriever.py` | Split into `Bm25Index`, `QdrantVectorStore`, `Embedder`, `DocumentCatalog` behind `RetrieverPort` | partly fixed 2026-11-02 — `LexicalIndex` owns tokenization, BM25 state, and ranking; Qdrant store and catalog extraction ticketed |
| V9: controller holds domain rules (`_build_scenarios`, `_commit_document`, `_load_persisted_chunks`, scope inference) | `src/api/main.py` | Move to `src/usecases/`; `main.py` becomes wiring plus route mapping | fixed 2026-11-02 — scope/scenarios in increment 3; document commit and upload precedence in `src/usecases/documents.py` with `src/adapters/uploads.py` |
| V10: the composition root imports `pymupdf` for page rendering and `evaluation` for the report runner | `src/api/main.py` | Render in an adapter; inject the evaluation runner | fixed 2026-11-02 — `src/adapters/page_images.py` renders and validates PDFs; `src/adapters/evaluation_runner.py` owns the harness |
| V11: the cache adapter owns its Redis payload format and memory fallback with no inward port | `src/cache_and_guardrails/semantic_cache.py` | Covered by V3's `AnswerCachePort`; internals may stay | fixed 2026-11-02 — the pipeline depends on `AnswerCachePort` only |
| V12: request/response DTOs import deployment settings | `src/inference/schemas.py` | Pass limits explicitly from the controller | fixed 2026-11-02 — the DTO carries a structural cap; the configured limit is enforced by the ingestion controller |
| V13: a 1,687-line HTML payload lives in Python | `src/api/ui.py` | Move to a static asset loaded by the route | fixed 2026-11-02 — `src/api/static/workbench.html`; `ui.py` is a cached loader |

## Bounded Contexts & Context Map

One bounded context: document question answering. Internally the contexts are ingestion (PDF → chunks),
retrieval (query → ranked evidence), synthesis (evidence → cited answer), and governance (guardrails,
verification, caching). They share the `DocumentChunk` and evidence DTOs; the shared kernel is exactly
what V2 moves into `src/domain/`. No anti-corruption layer exists yet between the vendor SDKs (Qdrant,
Redis, PyMuPDF, httpx) and the rest of the code — that is what the ports in V3/V5/V7/V8 introduce.

## Domain Glossary (Ubiquitous Language)

| Term | Meaning | Code name |
|---|---|---|
| Chunk | A page-anchored unit of extracted text or table content | `DocumentChunk` |
| Evidence | A retrieved chunk scored by fusion and optionally reranking | `RetrievedDocument` |
| Scope | The document filter applied to a query; `ALL` means no filter | `doc_name`, `_normalize_scope` |
| Physical page | The 1-based page index in the PDF, used for every citation | `page_number` |
| Contributor | An evidence page the synthesized answer actually cited | `answer_contributors` |
| Verification | Numeric-token presence check against cited evidence, not semantic entailment | `DeterministicNumericVerifier` |
| Gate | Whether the cross-encoder reranker runs for a query | `RERANKER_GATE`, `adaptive` |
| Generation | A monotonic counter that tells replicas the shared index changed | `INDEX_GENERATION_KEY` |

## Data & Storage Decisions

- **Qdrant** holds dense vectors and payloads; the collection is scoped to
  `…__physical_pages_v4__…` so page-numbering changes cannot reuse stale vectors. Old collections are
  not deleted automatically. A keyword payload index on `doc_name` is created (and verified) at
  startup: every scoped search filters on that field, so without it Qdrant scans the collection. An
  integer index on `commit_epoch` is created beside it for the stale-delete fence and the reconcile.
- **Redis Stack** holds the semantic cache (exact hash + HNSW) and the index-generation counter.
  Cache keys are namespaced `vectracite:semcache:physical-pages-v4:`; old entries expire by TTL.
- **Filesystem** holds the parsed-corpus caches (`data/*_index.json`), embeddings
  (`data/corpus_embeddings.npz`), and uploads. Parsed caches are content-addressed by PDF SHA256 plus
  the parser signature and gated by `tests/test_corpus_cache.py`. Embedding vectors are cached per
  chunk id, so a user upload reuses every corpus vector and only its own chunks are encoded; startup
  never writes that file, which belongs to `scripts/build_corpus_index.py`.
- **System of record** is the PDF corpus plus uploads; chunks, vectors, and cache entries are derived
  data rebuilt from it.
- **Concurrency and isolation.** There is no relational database and therefore no database isolation
  level to rely on. Within a process, mutations are serialized by an asyncio lock; across processes by a
  Redis mutation lease whose TTL is renewed every third of its lifetime while a write is in flight
  (`src/api/distributed_lock.py`). Every acquisition also returns a monotonic `commit_epoch`: points
  carry it, and a replace or delete only removes points with an older or equal epoch, so a stale holder
  cannot destroy a newer commit. A holder that notices the lease went to another replica fails its own
  remaining writes with 503 instead of writing through. Hydration reconciles to the maximum epoch per
  document, so a crash between the Qdrant upsert and the stale-point delete converges on the next sync.
  A document commit is a multi-step write (file + catalog + Qdrant) with an application-level
  compensation: snapshot, replace, restore on failure. There are still no cross-store transactions:
  the PDF file and Qdrant can disagree after a crash until the next successful commit. Collection-wide
  clears are fenced too: they delete points by epoch instead of dropping the collection, so newer
  points survive a stale clear.
- **Pagination.** `/documents/{doc_name}` is paginated (`limit`/`offset`); `/documents` and
  `/scenarios` return bounded-by-corpus lists, recorded as known limitations rather than paginated at this
  scale.

## Decision Log

| Date | Decision | Why | Alternatives rejected |
|---|---|---|---|
| 2026-11-02 | Physical PDF pages are the only citation coordinate; `SOURCE PAGE` labels remain text | Removes a whole class of mapping bugs and keeps citations verifiable against the file | Keeping dual numbering with a translation table |
| 2026-11-02 | Corrections land as one reviewed commit before Phase 2 code moves | The owner asked for the in-flight edits to be verified first; the suite is the gate that makes the re-layering safe | Re-layering on top of unverified changes |
| 2026-11-02 | Phase 2 scope is full codebase re-layering, executed in seven reviewable increments | Owner chose full scope; sequencing keeps each increment test-green instead of one big-bang rewrite | Hot-path-only pass |
| 2026-11-02 | Ports are structural `typing.Protocol` contracts, and adapters keep their existing method names | Avoids rewriting every adapter just to satisfy an ABC, while the pipeline is still testable with plain fakes | Abstract base classes plus adapter renames |
| 2026-11-02 | Scenario and scope policy moved now; document commit and upload precedence wait for increment 5 | The commit path needs the parser/uploads boundary to move without faking a port; splitting it now would produce a port shaped like the controller | Moving all controller rules in one step |
| 2026-11-02 | Adapter → `config` edges are accepted and ledgered; only entity/use-case → outer edges count as violations | Both adapter and config sit in the outer ring, so the Dependency Rule permits the edge; the testability cost is real but is a smell, not a rule breach | Injecting settings into all 8 adapters in the same pass |
| 2026-11-02 | The document lifecycle is a use case with a file-store port and catalog port; rollback stays inside it | Commit is policy (snapshot, replace, restore, invalidate) and was previously untestable outside HTTP; the ports let it run against fakes | Leaving commit in the controller |
| 2026-11-02 | Phase 5 ships no cache tier, ingestion queue, replica, or shard | Measured load is single-user, retrieval is ~13% of query time, and the only profile-named bottleneck (cold start) is fixed. Infrastructure here would be sized by fear, not evidence | Replica and queue proposals, both rejected as premature |
| 2026-11-02 | Capacity triggers recorded in `PERFORMANCE.md` instead of relying on memory | A deferred decision without a number becomes a forgotten one; each move now has a measured condition that reopens it | Leaving the deferral implicit |
| 2026-11-02 | Index `doc_name` in Qdrant and keep `/documents/{doc}` paginated; leave the small list endpoints unpaginated | Scoped search filters on `doc_name`, so the missing index was a real data-layer defect. The unpaginated lists are bounded by corpus size and are ledgered rather than paying pagination complexity now | Adding pagination everywhere pre-emptively |
| 2026-11-03 | The mutation lease renews every TTL/3 and a lost lease fails the next write, instead of a fixed TTL | Two replicas sharing one Qdrant can interleave a replace's upsert with the other replica's stale-point delete; renewal keeps exclusivity without a long crash-recovery wait. Verified against two live processes; the epoch fence and reconcile landed later the same day (next row) | A fixed 300 s TTL (a longer commit could be overtaken) or a long TTL without renewal (a crash would block writers for minutes) |
| 2026-11-03 | Fence commits with a shared epoch and reconcile duplicate versions during hydration | Qdrant cannot reject a write, so the fence makes stale deletes conditional on `commit_epoch` and the reconcile converges to the max epoch; together they close the stale-writer and crash-mid-commit gaps without a new datastore | A heartbeat-only warning (the stale delete still destroys newer data) or tombstone/outbox machinery (out of proportion for two replicas) |
| 2026-11-03 | Fence collection-wide clears by deleting points by epoch instead of dropping the collection | A collection drop cannot be conditioned on a token; an epoch-scoped delete keeps newer points and the payload indexes, and the routes report 409 without deleting uploads when survivors remain | Keeping the drop with a pre-drop lease check (a time-of-check race) |
| 2026-11-03 | Table and comparison questions search the document's own table chunks instead of only the reranked top 6 | Retrieval ranked narrative segment sales above 3M 2018 page 14, and the comparison path fell back to one narrative block; adding term-matched tables and preferring statement rows fixed the measured case | Raising `top_k` (does not guarantee the table is in the window) or a query-time finance word list |
