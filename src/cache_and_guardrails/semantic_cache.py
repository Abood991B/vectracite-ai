import hashlib
import json
import logging
from typing import Any, Dict, List
import numpy as np
import redis
from src.config import settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Redis-backed Semantic Cache that stores query embeddings and structured RAG responses.
    When an incoming query has cosine similarity >= threshold (0.92) with a cached query,
    the cached response is served in <20ms with $0 LLM token cost.
    Automatically falls back to an in-memory store if Redis is offline during CI tests.
    """

    def __init__(
        self,
        redis_url: str = settings.REDIS_URL,
        similarity_threshold: float = settings.CACHE_SIMILARITY_THRESHOLD,
        ttl_seconds: int = settings.CACHE_TTL_SECONDS,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._memory_cache: List[Dict[str, Any]] = []
        self._redis_client: redis.Redis | None = None
        self.hits: int = 0
        self.misses: int = 0

        try:
            client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1.5)
            client.ping()
            self._redis_client = client
        except Exception:
            self._redis_client = None

    @staticmethod
    def _embed_query_fast(query: str, dim: int = 256) -> np.ndarray:
        """Computes a normalized character & word n-gram vector for sub-millisecond cache lookup."""
        vec = np.zeros(dim, dtype=np.float32)
        normalized = " ".join(query.lower().strip().split())
        tokens = normalized.split(" ")
        features = tokens + [normalized[i : i + 3] for i in range(max(0, len(normalized) - 2))]
        for feat in features:
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def lookup(self, query: str) -> Dict[str, Any] | None:
        """Checks if a semantically equivalent query exists in Redis or memory cache."""
        q_vec = self._embed_query_fast(query)
        entries = self._load_entries()

        best_score = -1.0
        best_response: Dict[str, Any] | None = None

        for entry in entries:
            cached_vec = np.asarray(entry["vector"], dtype=np.float32)
            similarity = float(np.dot(q_vec, cached_vec))
            if similarity > best_score:
                best_score = similarity
                best_response = entry["response"]

        if best_score >= self.similarity_threshold and best_response is not None:
            self.hits += 1
            result = dict(best_response)
            result["cache_hit"] = True
            result["cache_similarity"] = round(best_score, 4)
            return result

        self.misses += 1
        return None

    def store(self, query: str, response: Dict[str, Any]) -> None:
        """Stores a query embedding and its verified RAG response in Redis/memory."""
        q_vec = self._embed_query_fast(query).tolist()
        entry = {"query": query, "vector": q_vec, "response": response}

        if self._redis_client is not None:
            try:
                key = f"vectracite:semcache:{hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]}"
                self._redis_client.setex(key, self.ttl_seconds, json.dumps(entry))
                return
            except Exception:
                pass

        self._memory_cache.append(entry)

    def _load_entries(self) -> List[Dict[str, Any]]:
        if self._redis_client is not None:
            try:
                keys = self._redis_client.keys("vectracite:semcache:*")
                if not keys:
                    return []
                raw_vals = self._redis_client.mget(keys)
                return [json.loads(v) for v in raw_vals if v]
            except Exception:
                pass
        return self._memory_cache

    def clear(self) -> None:
        """Clear cached responses in both Redis and the local fallback store."""
        if self._redis_client is not None:
            try:
                keys = self._redis_client.keys("vectracite:semcache:*")
                if keys:
                    self._redis_client.delete(*keys)
            except Exception:
                logger.debug("Unable to clear Redis semantic cache", exc_info=True)
        self._memory_cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": "redis" if self._redis_client is not None else "in_memory_fallback",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
        }
