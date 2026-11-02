# Cache, sync, and streaming

## Semantic cache

`SemanticCache` now has three lookup paths:

1. Exact SHA256 key lookup, O(1).
2. Redis Stack HNSW vector search over the 256-dimensional fast query vector.
3. Legacy JSON scan and bounded in-memory fallback for Redis deployments without
   the search module.

Redis Stack is already the compose/Kubernetes Redis image. The cache reports
`vector_index: true` in `/health` when the HNSW index is ready.

Validation on a real `redis/redis-stack-server` container:

```text
backend=redis vector_index=True
exact hit similarity=1.0
near-query semantic hit similarity=0.9648
clear removes exact and semantic entries
```

Semantic compatibility still requires matching numeric tokens and sufficient
lexical overlap; a paraphrase with different numbers is intentionally rejected.

## Index synchronization

API mutations and the KFP index component increment `vectracite:index:generation`.
The periodic task reads the counter every `INDEX_SYNC_SECONDS` and hydrates the
local BM25 catalog only when it changes. If Redis is unavailable, it falls back
to comparing the authoritative point count with the local catalog.

## Page images

Rendered page PNGs are cached in process with a bounded 128-entry LRU. The cache
key includes the PDF mtime/size, page number, and normalized highlight terms. The
response carries an ETag and `Cache-Control: private, max-age=3600`; matching
`If-None-Match` requests return `304`.

## Streaming

`/query/stream` now streams `llm_client.stream_answer` tokens instead of producing
the full answer first. A `StreamingRedactor` holds a bounded tail so PII patterns
split across SSE chunks cannot escape redaction. The final event carries the
answer, matched cells, deltas, citations, verification, guardrail flags, latency,
and provider metadata.

The workbench uses `fetch` with a readable stream, updates the answer box as tokens
arrive, and renders the final verification event. If streaming fails, it falls back
to the non-streaming `/query` route.

The remote HTTP provider reuses a single `httpx.AsyncClient` per event loop and
closes it during application shutdown.
