import hashlib
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.domain.attributes import PERSON_ATTRIBUTE_TERMS, person_count_value
from src.domain.documents import DocumentChunk, RetrievedDocument
from src.domain.financial_statements import is_statement_heading
from src.retrieval.embedding_cache import load_embeddings, save_embeddings
from src.retrieval.lexical_index import LexicalIndex, tokenize
from src.usecases.query_normalization import vocabulary_from_texts
from src.retrieval.table_ast import named_entities

logger = logging.getLogger(__name__)

_SHARED_EMBEDDERS: dict[tuple[str, int], SentenceTransformer] = {}
_SHARED_LIGHTWEIGHT_EMBEDDERS: dict[int, "LightweightEmbedder"] = {}


def _chunk_index(chunk: DocumentChunk) -> int:
    return int(chunk.metadata.get("chunk_index", 0) or 0)


def _opening_chunk(chunks: List[DocumentChunk]) -> DocumentChunk:
    return min(chunks, key=_chunk_index)


@dataclass
class DocumentIndex:
    """Per-document answers to questions that only change when the document changes.

    Scanning the whole store for statement headings and people-count sentences took ~10-20 ms
    per query on a 4,000-chunk corpus. Both answers depend only on the document's chunks, so
    they are built once and dropped whenever the store is replaced. Each is built on first use,
    so a table question never pays for the people-count scan and the reverse.
    """

    chunks: List[DocumentChunk] = field(default_factory=list)
    statement_pages: List[List[DocumentChunk]] | None = None
    person_count_chunks: List[DocumentChunk] | None = None


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
        self._lexical = LexicalIndex()
        self._vocab_cache: tuple[tuple[int, str], Dict[str, int]] | None = None
        self._by_document: Dict[str, List[DocumentChunk]] = {}
        self._document_indexes: Dict[str, DocumentIndex] = {}
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
        if self.backend in {"qdrant", "shared_client"}:
            self._ensure_payload_index()

    def _ensure_payload_index(self) -> None:
        """Index the fields scoped search and the commit fence filter on."""
        required = {
            "doc_name": qmodels.PayloadSchemaType.KEYWORD,
            "commit_epoch": qmodels.PayloadSchemaType.INTEGER,
        }
        try:
            info = self.qdrant.get_collection(self.collection_name)
            existing = info.payload_schema or {}
            for field_name, schema in required.items():
                if field_name in existing:
                    continue
                self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )
        except Exception as exc:
            logger.warning(
                "Payload index unavailable (%s); scoped search and fencing will scan",
                type(exc).__name__,
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
        return tokenize(text)

    def _query_entities(self, query: str) -> set[str]:
        """The corpus entities this query names, derived from the indexed document names."""
        return named_entities(query, self._by_document)

    @classmethod
    def _chunk_matches_entity(cls, chunk: DocumentChunk, entity: str) -> bool:
        doc_name = chunk.doc_name.casefold()
        text_tokens = set(cls._tokenize(chunk.text))
        return entity in text_tokens or entity in doc_name

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
        epoch: int | None = None,
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
            if epoch is not None:
                payload["commit_epoch"] = epoch
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

    def _commit_chunks_store(self, chunks: List[DocumentChunk]) -> None:
        """Replace the in-memory store and drop everything derived from the old one.

        Every write to ``chunks_store`` goes through here. That is the whole point: the derived
        per-document indexes cannot go stale because there is no other way to change the store.
        """
        self.chunks_store = chunks
        by_document: Dict[str, List[DocumentChunk]] = {}
        for chunk in chunks:
            by_document.setdefault(chunk.doc_name, []).append(chunk)
        self._by_document = by_document
        self._document_indexes.clear()
        self._lexical.rebuild(chunks)

    def _document_index(self, doc_name: str) -> DocumentIndex:
        """The document's derived index container. Call under the mutation lock."""
        cached = self._document_indexes.get(doc_name)
        if cached is not None:
            return cached
        index = DocumentIndex(chunks=self._by_document.get(doc_name, []))
        self._document_indexes[doc_name] = index
        return index

    def _statement_pages(self, doc_name: str) -> List[List[DocumentChunk]]:
        index = self._document_index(doc_name)
        if index.statement_pages is None:
            pages: Dict[int, List[DocumentChunk]] = {}
            for chunk in index.chunks:
                pages.setdefault(chunk.page_number, []).append(chunk)
            index.statement_pages = [
                sorted(pages[page], key=_chunk_index)
                for page in sorted(pages)
                if is_statement_heading(_opening_chunk(pages[page]).text)
            ]
        return index.statement_pages

    def _person_count_chunks(self, doc_name: str) -> List[DocumentChunk]:
        index = self._document_index(doc_name)
        if index.person_count_chunks is None:
            index.person_count_chunks = [
                chunk for chunk in index.chunks if person_count_value(chunk.text)
            ]
        return index.person_count_chunks

    def _replace_local_chunks(
        self, doc_name: str, chunks: List[DocumentChunk]
    ) -> None:
        remaining = [chunk for chunk in self.chunks_store if chunk.doc_name != doc_name]
        self._commit_chunks_store(
            sorted(remaining + chunks, key=self._chunk_sort_key)
        )

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
        epoch: int | None = None,
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
        points = self._build_points(chunks, vectors, epoch=epoch)

        with self._mutation_lock:
            self._upsert_points(points)
            existing = {chunk.chunk_id: chunk for chunk in self.chunks_store}
            for chunk in chunks:
                existing[chunk.chunk_id] = chunk
            self._commit_chunks_store(
                sorted(existing.values(), key=self._chunk_sort_key)
            )
        return len(chunks)

    def replace_document(
        self, doc_name: str, chunks: List[DocumentChunk], *, epoch: int | None = None
    ) -> int:
        """Stage a new document version, then remove versions older than this commit."""
        if not chunks:
            return self.delete_document(doc_name, epoch=epoch)

        version = uuid.uuid4().hex
        vectors = self._encode_texts([chunk.text for chunk in chunks])
        points = self._build_points(chunks, vectors, version=version, epoch=epoch)
        stale_must_not = [
            qmodels.FieldCondition(
                key="version", match=qmodels.MatchValue(value=version)
            )
        ]
        if epoch is not None:
            # The fence: never delete a version committed by a newer lease epoch.
            stale_must_not.append(
                qmodels.FieldCondition(
                    key="commit_epoch", range=qmodels.Range(gt=epoch)
                )
            )
        stale_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_name", match=qmodels.MatchValue(value=doc_name)
                )
            ],
            must_not=stale_must_not,
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
                            self._build_points(
                                old_chunks,
                                old_vectors,
                                version=uuid.uuid4().hex,
                                epoch=epoch,
                            )
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

    def restore_document(
        self,
        doc_name: str,
        chunks: List[DocumentChunk],
        *,
        epoch: int | None = None,
    ) -> None:
        """Restore a previously snapshotted document after an API-level failure."""
        if chunks:
            self.replace_document(doc_name, chunks, epoch=epoch)
        else:
            self.delete_document(doc_name, epoch=epoch)

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
                    chunk_index=int(payload.get("chunk_index", 0) or 0),
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
        if not chunks_store or not self._lexical.ready:
            return []

        scope = doc_name.strip() if doc_name and doc_name.strip().upper() != "ALL" else None
        candidate_limit = max(top_k * self.candidate_multiplier, 40)
        _t0 = time.perf_counter()
        q_vec = self._encode_texts([query])[0].tolist()
        _t0 = time.perf_counter()
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

        _t0 = time.perf_counter()
        bm25_ranks: Dict[str, int] = {}
        bm25_scores: Dict[str, float] = {}
        for rank, (chunk, score) in enumerate(
            self._lexical.ranked(query, limit=candidate_limit, scope=scope)
        ):
            bm25_ranks[chunk.chunk_id] = rank + 1
            bm25_scores[chunk.chunk_id] = score

        all_candidate_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        fused_results: List[RetrievedDocument] = []
        query_entities = self._query_entities(query) if scope is None else set()

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
                    self._chunk_matches_entity(chunk, entity)
                    for entity in query_entities
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
                    chunk_index=int(chunk.metadata.get("chunk_index", 0) or 0),
                )
            )

        fused_results.sort(
            key=lambda x: (x.rrf_score, x.contains_table),
            reverse=True,
        )
        return fused_results[:top_k]

    def clear(self, *, epoch: int | None = None) -> None:
        """Delete indexed points without dropping the collection.

        A collection drop cannot be conditioned on a commit token, so a fenced clear deletes only
        points committed up to ``epoch``. A stale holder therefore cannot erase a newer commit;
        the caller can detect the survivors with ``has_newer_commit`` and retry.
        """
        must_not = []
        if epoch is not None:
            must_not.append(
                qmodels.FieldCondition(
                    key="commit_epoch", range=qmodels.Range(gt=epoch)
                )
            )
        with self._mutation_lock:
            self._ensure_collection()
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.Filter(must_not=must_not or None),
            )
            self._commit_chunks_store([])
            self.last_sync_at = time.monotonic()

    def has_newer_commit(self, epoch: int) -> bool:
        """Return whether any point was committed by a lease epoch newer than ``epoch``."""
        selector = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="commit_epoch", range=qmodels.Range(gt=epoch)
                )
            ]
        )
        try:
            return (
                self.qdrant.count(
                    collection_name=self.collection_name,
                    count_filter=selector,
                    exact=True,
                ).count
                > 0
            )
        except Exception:
            return False

    def delete_document(self, doc_name: str, *, epoch: int | None = None) -> int:
        """Remove the document's points, never those committed by a newer lease epoch."""
        must_not = []
        if epoch is not None:
            must_not.append(
                qmodels.FieldCondition(
                    key="commit_epoch", range=qmodels.Range(gt=epoch)
                )
            )
        selector = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_name", match=qmodels.MatchValue(value=doc_name)
                )
            ],
            must_not=must_not or None,
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
            if epoch is not None and remote_removed == 0:
                # A newer commit owns this document now; leave the local catalog alone and let
                # the caller report that the delete did not apply.
                return 0
            if remote_removed:
                self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=selector,
                )
            self._commit_chunks_store(
                [
                    chunk
                    for chunk in self.chunks_store
                    if chunk.doc_name != doc_name
                ]
            )
        return max(local_removed, remote_removed)

    def has_document(self, doc_name: str) -> bool:
        """Return whether any point for `doc_name` remains, preferring the shared store."""
        selector = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_name", match=qmodels.MatchValue(value=doc_name)
                )
            ]
        )
        try:
            return (
                self.qdrant.count(
                    collection_name=self.collection_name,
                    count_filter=selector,
                    exact=True,
                ).count
                > 0
            )
        except Exception:
            with self._mutation_lock:
                return any(chunk.doc_name == doc_name for chunk in self.chunks_store)

    def table_candidates(
        self, doc_name: str, terms: set[str], limit: int = 40
    ) -> List[DocumentChunk]:
        """Return the document's table chunks that mention at least one query term.

        Retrieval's top-k can miss an income statement the document contains (3M 2018 keeps it
        on page 14 while narrative segment sales rank higher), so table synthesis searches the
        document's own table chunks instead of only the reranked candidates.
        """
        normalized = {term.casefold() for term in terms if term}
        matches: List[DocumentChunk] = []
        with self._mutation_lock:
            for chunk in self.chunks_store:
                if chunk.doc_name != doc_name or not chunk.contains_table:
                    continue
                if normalized and not normalized.intersection(tokenize(chunk.text)):
                    continue
                matches.append(chunk)
                if len(matches) >= limit:
                    break
        return matches

    def attribute_candidates(
        self, doc_name: str, terms: set[str], limit: int = 5
    ) -> List[DocumentChunk]:
        """Return the document's chunks that state the asked-for people count.

        Reranked top-k can miss the human-capital page ("How many employees does AMD have?" ranks
        compensation and stock-plan pages first), so a headcount question searches the document's
        own chunks for a sentence that states a number of people.
        """
        normalized = {term.casefold() for term in terms if term}
        attribute_question = bool(PERSON_ATTRIBUTE_TERMS & normalized)
        matches: List[DocumentChunk] = []
        with self._mutation_lock:
            for chunk in self._person_count_chunks(doc_name):
                if not attribute_question and not normalized.intersection(
                    tokenize(chunk.text)
                ):
                    continue
                matches.append(chunk)
                if len(matches) >= limit:
                    break
        return matches

    def statement_candidates(
        self, doc_name: str, limit_pages: int = 6
    ) -> List[DocumentChunk]:
        """Return every chunk of the document's consolidated primary statement pages.

        An annual report restates each figure in MD&A tables, notes, and selected-financial-data
        summaries, so retrieval alone can hand synthesis a page that repeats the number without
        being its source. The statement page is identified by its opening chunk, and all of its
        chunks are returned so synthesis can read the reporting line even when the page is
        printed as plain text rather than as an extracted table.
        """
        with self._mutation_lock:
            pages = self._statement_pages(doc_name)[:limit_pages]
        return [chunk for page in pages for chunk in page]

    @staticmethod
    def _payload_epoch(payload: Dict[str, Any]) -> int:
        try:
            return int(payload.get("commit_epoch", 0))
        except (TypeError, ValueError):
            return 0

    def _reconcile_versions(self, records: List[Any]) -> List[Any]:
        """Keep the newest commit epoch per document and delete versions a crash left behind."""
        groups: Dict[str, Dict[str | None, Dict[str, Any]]] = {}
        for record in records:
            payload = record.payload or {}
            doc_name = str(payload.get("doc_name", "")).strip()
            if not doc_name:
                continue
            version = payload.get("version")
            version_key = str(version) if version is not None else None
            epoch = self._payload_epoch(payload)
            group = groups.setdefault(doc_name, {}).setdefault(
                version_key, {"epoch": epoch, "records": []}
            )
            group["records"].append(record)
            group["epoch"] = max(group["epoch"], epoch)

        kept: List[Any] = []
        for doc_name, versions in groups.items():
            if len(versions) <= 1:
                kept.extend(next(iter(versions.values()))["records"])
                continue
            newest_epoch = max(group["epoch"] for group in versions.values())
            winners = [
                version
                for version, group in versions.items()
                if group["epoch"] == newest_epoch
            ]
            if len(winners) > 1:
                logger.warning(
                    "Document %s has %d versions without an ordering epoch; keeping all of them",
                    doc_name,
                    len(winners),
                )
                for version in winners:
                    kept.extend(versions[version]["records"])
                continue
            winner = winners[0]
            kept.extend(versions[winner]["records"])
            for version, group in versions.items():
                if version == winner:
                    continue
                if self._delete_version(doc_name, version):
                    logger.warning(
                        "Reconciled document %s: removed version %s (%d points) in favour "
                        "of epoch %s",
                        doc_name,
                        version,
                        len(group["records"]),
                        newest_epoch,
                    )
        return kept

    def _delete_version(self, doc_name: str, version: str | None) -> bool:
        must = [
            qmodels.FieldCondition(
                key="doc_name", match=qmodels.MatchValue(value=doc_name)
            )
        ]
        if version is None:
            must.append(
                qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="version"))
            )
        else:
            must.append(
                qmodels.FieldCondition(
                    key="version", match=qmodels.MatchValue(value=version)
                )
            )
        try:
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.Filter(must=must),
            )
            return True
        except Exception:
            logger.exception(
                "Unable to remove stale version %s of %s during reconcile",
                version,
                doc_name,
            )
            return False

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

        records = self._reconcile_versions(records)
        chunks = [
            chunk
            for record in records
            if (chunk := self._chunk_from_payload(record.payload or {})) is not None
        ]
        chunks.sort(key=self._chunk_sort_key)
        with self._mutation_lock:
            self._commit_chunks_store(chunks)
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

    def corpus_vocabulary(self) -> Dict[str, int]:
        """Term frequencies from the indexed chunks, cached per catalog generation."""
        with self._mutation_lock:
            chunks = list(self.chunks_store)
        signature = (len(chunks), chunks[0].chunk_id if chunks else "")
        if self._vocab_cache is not None and self._vocab_cache[0] == signature:
            return self._vocab_cache[1]
        counts = vocabulary_from_texts(chunk.text for chunk in chunks)
        self._vocab_cache = (signature, counts)
        return counts

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
