import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.ingestion.pdf_parser import DocumentChunk

logger = logging.getLogger(__name__)

_SHARED_EMBEDDER: SentenceTransformer | None = None


def get_shared_embedder() -> SentenceTransformer:
    """Load the production BGE embedder once per process."""
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _SHARED_EMBEDDER


@dataclass
class RetrievedDocument:
    """Represents a retrieved chunk scored by Hybrid Search (Dense + BM25 RRF) and/or Reranker."""

    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    contains_table: bool
    dense_score: float
    bm25_score: float
    rrf_score: float
    rerank_score: float | None = None


class HybridRetriever:
    """
    Production Hybrid Retrieval Engine combining:
      1. Dense Semantic Vector Search stored in Qdrant.
      2. Sparse Lexical BM25 Search (BM25Okapi) for exact financial ticker, date, and metric matches.
      3. Reciprocal Rank Fusion (RRF) to merge dense and lexical rankings.
    """

    def __init__(
        self,
        collection_name: str = settings.QDRANT_COLLECTION,
        in_memory: bool = settings.QDRANT_IN_MEMORY,
        rrf_k: int = 60,
    ) -> None:
        self.collection_name = collection_name
        self.rrf_k = rrf_k

        if in_memory:
            self.qdrant = QdrantClient(":memory:")
        else:
            try:
                self.qdrant = QdrantClient(
                    host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=5.0
                )
                self.qdrant.get_collections()
            except Exception as exc:
                logger.warning("Qdrant server unreachable (%s); falling back to :memory:.", exc)
                self.qdrant = QdrantClient(":memory:")

        self._ensure_collection()
        self.chunks_store: List[DocumentChunk] = []
        self.bm25: BM25Okapi | None = None

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def _search_dense(self, query_vector: List[float], limit: int) -> List[Any]:
        """Run dense search across legacy and current Qdrant client APIs."""
        if hasattr(self.qdrant, "search"):
            return self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )

        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )
        return response.points

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode text with the shared production BGE-small model."""
        embedder = get_shared_embedder()
        embeddings = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_$\.%]+", text.lower())

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Indexes DocumentChunks into both Qdrant (dense vectors) and BM25 (sparse lexical)."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self._encode_texts(texts)

        points: List[qmodels.PointStruct] = []
        for idx, chunk in enumerate(chunks):
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                qmodels.PointStruct(
                    id=point_uuid,
                    vector=vectors[idx].tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_name": chunk.doc_name,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                        "contains_table": chunk.contains_table,
                    },
                )
            )

        self.qdrant.upsert(collection_name=self.collection_name, points=points)

        # Rebuild BM25 corpus across all indexed chunks
        existing_ids = {c.chunk_id for c in self.chunks_store}
        for c in chunks:
            if c.chunk_id not in existing_ids:
                self.chunks_store.append(c)
                existing_ids.add(c.chunk_id)

        tokenized_corpus = [self._tokenize(c.text) for c in self.chunks_store]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return len(chunks)

    def retrieve_dense_only(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        """Executes standard dense-only vector search in Qdrant (used as baseline comparison)."""
        if not self.chunks_store:
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

    def retrieve_hybrid(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        """
        Executes Hybrid Search by fusing Qdrant Dense Rankings and BM25 Lexical Rankings
        via Reciprocal Rank Fusion (RRF):
            RRF_Score(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_bm25(d))
        """
        if not self.chunks_store or self.bm25 is None:
            return []

        candidate_limit = max(top_k * 2, 15)
        q_vec = self._encode_texts([query])[0].tolist()
        dense_hits = self._search_dense(q_vec, candidate_limit)

        dense_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, DocumentChunk] = {c.chunk_id: c for c in self.chunks_store}

        for rank, hit in enumerate(dense_hits):
            cid = str((hit.payload or {}).get("chunk_id", ""))
            if cid:
                dense_ranks[cid] = rank + 1
                dense_scores[cid] = float(hit.score)

        # Compute BM25 lexical scores
        query_tokens = self._tokenize(query)
        raw_bm25_scores = self.bm25.get_scores(query_tokens)
        sorted_bm25_indices = np.argsort(raw_bm25_scores)[::-1][:candidate_limit]

        bm25_ranks: Dict[str, int] = {}
        bm25_scores: Dict[str, float] = {}
        for rank, idx in enumerate(sorted_bm25_indices):
            cid = self.chunks_store[int(idx)].chunk_id
            bm25_ranks[cid] = rank + 1
            bm25_scores[cid] = float(raw_bm25_scores[int(idx)])

        # Reciprocal Rank Fusion (RRF)
        all_candidate_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        fused_results: List[RetrievedDocument] = []

        for cid in all_candidate_ids:
            chunk = chunk_lookup.get(cid)
            if not chunk:
                continue

            d_rank = dense_ranks.get(cid, candidate_limit + 50)
            b_rank = bm25_ranks.get(cid, candidate_limit + 50)
            rrf_score = (0.45 / (self.rrf_k + d_rank)) + (0.55 / (self.rrf_k + b_rank))

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

        fused_results.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused_results[:top_k]

    def delete_document(self, doc_name: str) -> int:
        """Removes all chunks belonging to `doc_name` and rebuilds the Qdrant + BM25 index."""
        before_count = len(self.chunks_store)
        remaining = [c for c in self.chunks_store if c.doc_name != doc_name]
        removed_count = before_count - len(remaining)

        if removed_count > 0:
            self.qdrant.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            self.chunks_store = []
            self.bm25 = None
            if remaining:
                self.index_chunks(remaining)

        return removed_count

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
