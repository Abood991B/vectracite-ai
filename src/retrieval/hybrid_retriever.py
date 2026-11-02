import hashlib
import logging
import re
import threading
import time
import uuid
from typing import Any, Dict, List
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.domain.documents import DocumentChunk, RetrievedDocument
from src.retrieval.embedding_cache import load_embeddings, save_embeddings
from src.retrieval.table_ast import KNOWN_ENTITY_ALIASES

logger = logging.getLogger(__name__)

BM25_STOPWORDS = frozenset(
    {
        "about",
        "according",
        "also",
        "and",
        "answer",
        "are",
        "based",
        "been",
        "both",
        "but",
        "calculate",
        "can",
        "could",
        "did",
        "does",
        "following",
        "for",
        "from",
        "give",
        "has",
        "have",
        "how",
        "include",
        "including",
        "into",
        "its",
        "may",
        "not",
        "please",
        "provided",
        "question",
        "rely",
        "relying",
        "response",
        "should",
        "show",
        "shown",
        "state",
        "such",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "those",
        "use",
        "using",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "your",
    }
)

_SHARED_EMBEDDERS: dict[tuple[str, int], SentenceTransformer] = {}
_SHARED_LIGHTWEIGHT_EMBEDDERS: dict[int, "LightweightEmbedder"] = {}


class LightweightEmbedder:
    """Deterministic hashing embedder used only for fast tests and local CI."""

    def __init__(self, dimension: int = settings.EMBEDDING_DIM) -> None:
        self.dimension = dimension

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del show_progress_bar
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-zA-Z0-9_$\.%]+", text.lower())
            features = tokens + [
                f"{left}:{right}" for left, right in zip(tokens, tokens[1:])
            ]
            for feature in features:
                digest = hashlib.sha1(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        return matrix


def get_embedder(
    model_name: str,
    dimension: int,
    use_lightweight: bool | None = None,
) -> Any:
    """Load and cache one embedder per model/dimension pair."""
    lightweight = settings.USE_LIGHTWEIGHT_MODELS if use_lightweight is None else use_lightweight
    if lightweight:
        if dimension not in _SHARED_LIGHTWEIGHT_EMBEDDERS:
            _SHARED_LIGHTWEIGHT_EMBEDDERS[dimension] = LightweightEmbedder(dimension)
        return _SHARED_LIGHTWEIGHT_EMBEDDERS[dimension]
    key = (model_name, dimension)
    if key not in _SHARED_EMBEDDERS:
        model = SentenceTransformer(model_name)
        dimension_getter = getattr(model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = model.get_sentence_embedding_dimension
        actual_dimension = dimension_getter()
        if actual_dimension is not None and int(actual_dimension) != dimension:
            raise ValueError(
                f"embedding model {model_name} produces {actual_dimension} dimensions, "
                f"not {dimension}"
            )
        _SHARED_EMBEDDERS[key] = model
    return _SHARED_EMBEDDERS[key]


def get_shared_embedder(use_lightweight: bool | None = None) -> Any:
    """Load either the production BGE embedder or the isolated CI substitute."""
    return get_embedder(
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_DIM,
        use_lightweight=use_lightweight,
    )


def clear_embedder_cache() -> None:
    """Release cached embedding models between independent experiment runs."""
    _SHARED_EMBEDDERS.clear()
    _SHARED_LIGHTWEIGHT_EMBEDDERS.clear()


class HybridRetriever:
    """
    Production Hybrid Retrieval Engine combining:
      1. Dense Semantic Vector Search stored in Qdrant.
      2. Sparse Lexical BM25 Search (BM25Okapi) for exact financial ticker, date, and metric matches.
      3. Reciprocal Rank Fusion (RRF) to merge dense and lexical rankings.

    Mutations are serialized and the in-process BM25/read catalog is rebuilt from
    Qdrant, so multiple API replicas converge on a shared authoritative store.
    """

    def __init__(
        self,
        collection_name: str = settings.QDRANT_COLLECTION,
        in_memory: bool = settings.QDRANT_IN_MEMORY,
        rrf_k: int = 60,
        use_lightweight_embedder: bool | None = None,
        client: QdrantClient | None = None,
        allow_fallback: bool = settings.ALLOW_QDRANT_FALLBACK,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        dense_weight: float | None = None,
        bm25_weight: float | None = None,
        candidate_multiplier: int = 4,
    ) -> None:
        self.rrf_k = rrf_k
        self.use_lightweight_embedder = (
            settings.USE_LIGHTWEIGHT_MODELS
            if use_lightweight_embedder is None
            else use_lightweight_embedder
        )
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.embedding_dim = embedding_dim or settings.EMBEDDING_DIM
        self.dense_weight = (
            settings.HYBRID_DENSE_WEIGHT if dense_weight is None else dense_weight
        )
        self.bm25_weight = (
            settings.HYBRID_BM25_WEIGHT if bm25_weight is None else bm25_weight
        )
        self.candidate_multiplier = max(1, candidate_multiplier)
        model_tag = (
            "lightweight"
            if self.use_lightweight_embedder
            else re.sub(r"[^a-z0-9]+", "_", self.embedding_model.casefold()).strip("_")
        )
        self.collection_name = (
            f"{collection_name}__physical_pages_v4__{model_tag}__d{self.embedding_dim}"[:200]
        )
        self.requested_backend = "memory" if in_memory else "qdrant"
        self.shared_client = client is not None
        self.degraded = False
        self.last_error: str | None = None
        self.last_sync_at = 0.0
        self._mutation_lock = threading.RLock()

        if client is not None:
            self.qdrant = client
            self.backend = "shared_client"
        elif in_memory:
            self.qdrant = QdrantClient(":memory:")
            self.backend = "memory"
        else:
            try:
                self.qdrant = QdrantClient(
                    host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=5.0
                )
                self.qdrant.get_collections()
                self.backend = "qdrant"
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: shared Qdrant is unreachable"
                if not allow_fallback:
                    raise RuntimeError(
                        "Shared Qdrant is unavailable; refusing to start with a "
                        "process-local index. Set ALLOW_QDRANT_FALLBACK=true only for "
                        "explicit single-process deployments."
                    ) from exc
                logger.warning(
                    "Qdrant server unreachable (%s); falling back to :memory%.",
                    type(exc).__name__,
                )
                self.qdrant = QdrantClient(":memory:")
                self.backend = "memory_fallback"
                self.degraded = True

        self.chunks_store: List[DocumentChunk] = []
        self.bm25: BM25Okapi | None = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def _search_dense(
        self,
        query_vector: List[float],
        limit: int,
        doc_name: str | None = None,
    ) -> List[Any]:
        """Run dense search across legacy and current Qdrant client APIs."""
        query_filter = None
        if doc_name:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_name",
                        match=qmodels.MatchValue(value=doc_name),
                    )
                ]
            )
        if hasattr(self.qdrant, "search"):
            return self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
            )

        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
        return response.points

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode text with the configured production or test-only embedder."""
        embedder = get_embedder(
            self.embedding_model,
            self.embedding_dim,
            use_lightweight=self.use_lightweight_embedder,
        )
        embeddings = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"['�?T]s\b", "", text.lower())
        tokens: List[str] = []
        for token in re.findall(r"[\w$\.%]+", cleaned, flags=re.UNICODE):
            if len(token) < 2 and not token.isdigit():
                continue
            if token in BM25_STOPWORDS:
                continue
            tokens.append(token)
            parts = re.findall(r"[a-z]+|\d+", token)
            if len(parts) > 1:
                tokens.extend(
                    part for part in parts if len(part) >= 2 or part.isdigit()
                )
        return tokens

    @classmethod
    def _query_entities(cls, query: str) -> List[set[str]]:
        tokens = set(cls._tokenize(query))
        folded = query.casefold()
        return [
            aliases
            for aliases in KNOWN_ENTITY_ALIASES.values()
            if any(alias in tokens or alias in folded for alias in aliases)
        ]

    @classmethod
    def _chunk_matches_entity(
        cls, chunk: DocumentChunk, entity_aliases: set[str]
    ) -> bool:
        doc_name = chunk.doc_name.casefold()
        text_tokens = set(cls._tokenize(chunk.text))
        return any(alias in text_tokens or alias in doc_name for alias in entity_aliases)

    @staticmethod
    def _point_uuid(chunk: DocumentChunk) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

    @staticmethod
    def _chunk_sort_key(chunk: DocumentChunk) -> tuple[Any, ...]:
        return (
            chunk.doc_name,
            chunk.page_number,
            int(chunk.metadata.get("chunk_index", 0)),
            chunk.chunk_id,
        )

    def _build_points(
        self,
        chunks: List[DocumentChunk],
        vectors: np.ndarray,
        *,
        version: str | None = None,
    ) -> List[qmodels.PointStruct]:
        points: List[qmodels.PointStruct] = []
        for idx, chunk in enumerate(chunks):
            payload: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "contains_table": chunk.contains_table,
                "metadata": chunk.metadata,
            }
            if version is not None:
                payload["version"] = version
            points.append(
                qmodels.PointStruct(
                    id=self._point_uuid(chunk),
                    vector=vectors[idx].tolist(),
                    payload=payload,
                )
            )
        return points

    def _upsert_points(self, points: List[qmodels.PointStruct]) -> None:
        """Upsert in bounded batches so large corpora stay below Qdrant's body limit."""
        batch_size = max(1, settings.QDRANT_UPSERT_BATCH_SIZE)
        for start in range(0, len(points), batch_size):
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
            )

    def _rebuild_bm25(self) -> None:
        if self.chunks_store:
            tokenized_corpus = [self._tokenize(chunk.text) for chunk in self.chunks_store]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def _replace_local_chunks(
        self, doc_name: str, chunks: List[DocumentChunk]
    ) -> None:
        remaining = [chunk for chunk in self.chunks_store if chunk.doc_name != doc_name]
        self.chunks_store = sorted(remaining + chunks, key=self._chunk_sort_key)
        self._rebuild_bm25()

    @staticmethod
    def _chunk_from_payload(payload: Dict[str, Any]) -> DocumentChunk | None:
        chunk_id = str(payload.get("chunk_id", "")).strip()
        doc_name = str(payload.get("doc_name", "")).strip()
        text = str(payload.get("text", ""))
        if not chunk_id or not doc_name or not text:
            return None
        page_number = int(payload.get("page_number", 1))
        contains_table = bool(payload.get("contains_table", False))
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {
                "source": f"{doc_name}.pdf",
                "page": page_number,
                "chunk_index": 0,
                "has_financial_table": contains_table,
            }
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_name=doc_name,
            page_number=page_number,
            text=text,
            contains_table=contains_table,
            metadata=metadata,
        )

    def index_chunks(
        self,
        chunks: List[DocumentChunk],
        *,
        vectors: np.ndarray | None = None,
        use_embedding_cache: bool = True,
        write_embedding_cache: bool = False,
    ) -> int:
        """Indexes DocumentChunks into both Qdrant and the in-process BM25 catalog."""
        if not chunks:
            return 0

        if vectors is None:
            uses_default_embedder = (
                self.embedding_model == settings.EMBEDDING_MODEL
                and self.embedding_dim == settings.EMBEDDING_DIM
            )
            if (
                use_embedding_cache
                and not self.use_lightweight_embedder
                and uses_default_embedder
            ):
                vectors = load_embeddings(chunks)
            if vectors is None:
                texts = [chunk.text for chunk in chunks]
                vectors = self._encode_texts(texts)
                if (
                    write_embedding_cache
                    and not self.use_lightweight_embedder
                    and uses_default_embedder
                ):
                    save_embeddings(chunks, vectors)
        else:
            vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors) != len(chunks):
            raise ValueError("embedding count does not match chunk count")
        points = self._build_points(chunks, vectors)

        with self._mutation_lock:
            self._upsert_points(points)
            existing = {chunk.chunk_id: chunk for chunk in self.chunks_store}
            for chunk in chunks:
                existing[chunk.chunk_id] = chunk
            self.chunks_store = sorted(existing.values(), key=self._chunk_sort_key)
            self._rebuild_bm25()
        return len(chunks)

    def replace_document(self, doc_name: str, chunks: List[DocumentChunk]) -> int:
        """Stage a new document version, then remove stale points for that document."""
        if not chunks:
            return self.delete_document(doc_name)

        version = uuid.uuid4().hex
        vectors = self._encode_texts([chunk.text for chunk in chunks])
        points = self._build_points(chunks, vectors, version=version)
        stale_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_name", match=qmodels.MatchValue(value=doc_name)
                )
            ],
            must_not=[
                qmodels.FieldCondition(
                    key="version", match=qmodels.MatchValue(value=version)
                )
            ],
        )

        with self._mutation_lock:
            old_chunks = [c for c in self.chunks_store if c.doc_name == doc_name]
            try:
                self._upsert_points(points)
                self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=stale_filter,
                )
            except Exception:
                try:
                    self.qdrant.delete(
                        collection_name=self.collection_name,
                        points_selector=qmodels.Filter(
                            must=[
                                qmodels.FieldCondition(
                                    key="version",
                                    match=qmodels.MatchValue(value=version),
                                )
                            ]
                        ),
                    )
                    if old_chunks:
                        old_vectors = self._encode_texts([c.text for c in old_chunks])
                        self._upsert_points(
                            self._build_points(old_chunks, old_vectors)
                        )
                except Exception:
                    logger.exception("Unable to roll back failed document replacement")
                raise

            self._replace_local_chunks(doc_name, chunks)
        return len(old_chunks)

    def snapshot_document(self, doc_name: str) -> List[DocumentChunk]:
        """Return a local copy of a document's indexed chunks for rollback."""
        with self._mutation_lock:
            return [
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    doc_name=chunk.doc_name,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    contains_table=chunk.contains_table,
                    metadata=dict(chunk.metadata),
                )
                for chunk in self.chunks_store
                if chunk.doc_name == doc_name
            ]

    def restore_document(self, doc_name: str, chunks: List[DocumentChunk]) -> None:
        """Restore a previously snapshotted document after an API-level failure."""
        if chunks:
            self.replace_document(doc_name, chunks)
        else:
            self.delete_document(doc_name)

    def retrieve_dense_only(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        """Executes standard dense-only vector search in Qdrant (used as baseline comparison)."""
        with self._mutation_lock:
            has_chunks = bool(self.chunks_store)
        if not has_chunks:
            return []

        q_vec = self._encode_texts([query])[0].tolist()
        hits = self._search_dense(q_vec, top_k)

        results: List[RetrievedDocument] = []
        for rank, hit in enumerate(hits):
            payload: Dict[str, Any] = hit.payload or {}
            results.append(
                RetrievedDocument(
                    chunk_id=str(payload.get("chunk_id", "")),
                    doc_name=str(payload.get("doc_name", "")),
                    page_number=int(payload.get("page_number", 1)),
                    text=str(payload.get("text", "")),
                    contains_table=bool(payload.get("contains_table", False)),
                    dense_score=float(hit.score),
                    bm25_score=0.0,
                    rrf_score=1.0 / (self.rrf_k + rank + 1),
                )
            )
        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 10,
        doc_name: str | None = None,
    ) -> List[RetrievedDocument]:
        """
        Executes Hybrid Search by fusing Qdrant Dense Rankings and BM25 Lexical Rankings
        via Reciprocal Rank Fusion (RRF):
            RRF_Score(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_bm25(d))
        """
        with self._mutation_lock:
            chunks_store = list(self.chunks_store)
            bm25 = self.bm25
        if not chunks_store or bm25 is None:
            return []

        scope = doc_name.strip() if doc_name and doc_name.strip().upper() != "ALL" else None
        candidate_limit = max(top_k * self.candidate_multiplier, 40)
        q_vec = self._encode_texts([query])[0].tolist()
        dense_hits = self._search_dense(q_vec, candidate_limit, doc_name=scope)
        if scope:
            dense_hits = [
                hit
                for hit in dense_hits
                if (hit.payload or {}).get("doc_name") == scope
            ]

        dense_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, DocumentChunk] = {c.chunk_id: c for c in chunks_store}

        for rank, hit in enumerate(dense_hits):
            cid = str((hit.payload or {}).get("chunk_id", ""))
            if cid:
                dense_ranks[cid] = rank + 1
                dense_scores[cid] = float(hit.score)

        query_tokens = self._tokenize(query)
        raw_bm25_scores = bm25.get_scores(query_tokens)
        candidate_indices = np.arange(len(chunks_store))
        if scope:
            candidate_indices = np.asarray(
                [idx for idx, chunk in enumerate(chunks_store) if chunk.doc_name == scope],
                dtype=int,
            )
        ranked_indices = candidate_indices[np.argsort(raw_bm25_scores[candidate_indices])[::-1]]
        sorted_bm25_indices = ranked_indices[:candidate_limit]

        bm25_ranks: Dict[str, int] = {}
        bm25_scores: Dict[str, float] = {}
        for rank, idx in enumerate(sorted_bm25_indices):
            cid = chunks_store[int(idx)].chunk_id
            bm25_ranks[cid] = rank + 1
            bm25_scores[cid] = float(raw_bm25_scores[int(idx)])

        all_candidate_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        fused_results: List[RetrievedDocument] = []
        query_entities = self._query_entities(query) if scope is None else []

        for cid in all_candidate_ids:
            chunk = chunk_lookup.get(cid)
            if not chunk:
                continue

            table_bonus = 0.0003 if chunk.contains_table else 0.0
            base_rrf_score = table_bonus
            if cid in dense_ranks:
                base_rrf_score += self.dense_weight / (
                    self.rrf_k + dense_ranks[cid]
                )
            if cid in bm25_ranks:
                base_rrf_score += self.bm25_weight / (
                    self.rrf_k + bm25_ranks[cid]
                )
            anchor_multiplier = (
                1.35
                if any(
                    self._chunk_matches_entity(chunk, entity_aliases)
                    for entity_aliases in query_entities
                )
                else 1.0
            )
            rrf_score = base_rrf_score * anchor_multiplier

            fused_results.append(
                RetrievedDocument(
                    chunk_id=chunk.chunk_id,
                    doc_name=chunk.doc_name,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    contains_table=chunk.contains_table,
                    dense_score=dense_scores.get(cid, 0.0),
                    bm25_score=bm25_scores.get(cid, 0.0),
                    rrf_score=rrf_score,
                )
            )

        fused_results.sort(
            key=lambda x: (x.rrf_score, x.contains_table),
            reverse=True,
        )
        return fused_results[:top_k]

    def clear(self) -> None:
        """Remove every indexed chunk and recreate the empty vector collection."""
        with self._mutation_lock:
            self.qdrant.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            self.chunks_store = []
            self.bm25 = None
            self.last_sync_at = time.monotonic()

    def delete_document(self, doc_name: str) -> int:
        """Remove every point belonging to `doc_name` without dropping the collection."""
        selector = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_name", match=qmodels.MatchValue(value=doc_name)
                )
            ]
        )
        with self._mutation_lock:
            local_removed = sum(1 for c in self.chunks_store if c.doc_name == doc_name)
            try:
                remote_removed = self.qdrant.count(
                    collection_name=self.collection_name,
                    count_filter=selector,
                    exact=True,
                ).count
            except Exception:
                remote_removed = local_removed
            if remote_removed:
                self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=selector,
                )
            self.chunks_store = [
                chunk for chunk in self.chunks_store if chunk.doc_name != doc_name
            ]
            self._rebuild_bm25()
        return max(local_removed, remote_removed)

    def hydrate_from_qdrant(self) -> int:
        """Rebuild BM25 and the read catalog from the shared Qdrant payload store."""
        records: List[Any] = []
        offset: Any = None
        while True:
            page, offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(page)
            if offset is None:
                break

        chunks = [
            chunk
            for record in records
            if (chunk := self._chunk_from_payload(record.payload or {})) is not None
        ]
        chunks.sort(key=self._chunk_sort_key)
        with self._mutation_lock:
            self.chunks_store = chunks
            self._rebuild_bm25()
            self.last_sync_at = time.monotonic()
        return len(chunks)

    def sync_from_qdrant(self, *, force: bool = False) -> int:
        """Refresh the local catalog when a shared backend is in use."""
        if self.backend == "memory" and not self.shared_client:
            return len(self.chunks_store)
        if (
            not force
            and time.monotonic() - self.last_sync_at < settings.INDEX_SYNC_SECONDS
        ):
            return len(self.chunks_store)
        return self.hydrate_from_qdrant()

    def count_points(self) -> int:
        """Return the authoritative point count when the backend supports it."""
        try:
            return self.qdrant.count(
                collection_name=self.collection_name, exact=True
            ).count
        except Exception:
            return len(self.chunks_store)

    def health(self) -> Dict[str, Any]:
        reachable = True
        last_error = self.last_error
        if self.backend in {"qdrant", "shared_client"}:
            try:
                self.qdrant.get_collections()
            except Exception as exc:
                reachable = False
                last_error = f"{type(exc).__name__}: shared Qdrant is unreachable"
        with self._mutation_lock:
            chunks = len(self.chunks_store)
        return {
            "requested_backend": self.requested_backend,
            "backend": self.backend,
            "reachable": reachable,
            "degraded": self.degraded or not reachable,
            "last_error": last_error,
            "indexed_chunks": chunks,
            "last_sync_at": self.last_sync_at,
        }

    def indexed_document_names(self) -> List[str]:
        """Return the sorted document names in the local catalog."""
        return sorted({chunk.doc_name for chunk in self.chunks_store})

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns summary metadata for all currently indexed documents."""
        summary: Dict[str, Dict[str, Any]] = {}
        for chunk in self.chunks_store:
            if chunk.doc_name not in summary:
                summary[chunk.doc_name] = {
                    "doc_name": chunk.doc_name,
                    "total_chunks": 0,
                    "table_chunks": 0,
                    "pages": set(),
                }
            summary[chunk.doc_name]["total_chunks"] += 1
            if chunk.contains_table:
                summary[chunk.doc_name]["table_chunks"] += 1
            summary[chunk.doc_name]["pages"].add(chunk.page_number)

        result = []
        for doc_name, info in sorted(summary.items()):
            result.append(
                {
                    "doc_name": doc_name,
                    "total_chunks": info["total_chunks"],
                    "table_chunks": info["table_chunks"],
                    "pages": sorted(list(info["pages"])),
                }
            )
        return result

    def get_document_chunks(self, doc_name: str) -> List[Dict[str, Any]]:
        """Returns all indexed chunks and tables for a specific document."""
        matched = [c.to_dict() for c in self.chunks_store if c.doc_name == doc_name]
        matched.sort(key=lambda item: (item["page_number"], item["metadata"].get("chunk_index", 0)))
        return matched
