import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Set

import numpy as np
import redis

from src.config import settings
from src.retrieval.financial_numbers import (
    canonical_number_values,
    extract_number_tokens,
)

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
        self._degraded = False
        self._last_error: str | None = None
        self.hits: int = 0
        self.misses: int = 0

        try:
            client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1.5,
                socket_timeout=1.5,
            )
            client.ping()
            self._redis_client = client
        except Exception as exc:
            self._mark_redis_unavailable(exc)

    def _mark_redis_unavailable(self, exc: Exception) -> None:
        """Disable Redis after an outage so readiness exposes the degraded state."""
        self._redis_client = None
        self._degraded = True
        self._last_error = f"{type(exc).__name__}: {exc}"[:500]
        logger.warning("Semantic cache Redis backend unavailable: %s", type(exc).__name__)

    def check_health(self) -> bool:
        """Probe Redis before readiness decisions, without hiding an outage."""
        if self._redis_client is None:
            return False
        try:
            self._redis_client.ping()
            return True
        except Exception as exc:
            self._mark_redis_unavailable(exc)
            return False

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

    @staticmethod
    def _query_signature(query: str) -> tuple[Set[str], Set[str]]:
        """Return numeric and lexical signatures used to reject near-miss cache hits."""
        numbers: Set[str] = set()
        for token in extract_number_tokens(query, ignore_page_references=True):
            numbers.update(canonical_number_values(token))
        stopwords = {
            "about",
            "according",
            "and",
            "are",
            "does",
            "for",
            "from",
            "how",
            "in",
            "is",
            "of",
            "the",
            "was",
            "what",
            "were",
            "with",
        }
        words = {
            word
            for word in re.findall(r"[a-z0-9]+", query.casefold())
            if len(word) >= 3 and word not in stopwords
        }
        return numbers, words

    @classmethod
    def _compatible_queries(cls, left: str, right: str) -> bool:
        left_numbers, left_words = cls._query_signature(left)
        right_numbers, right_words = cls._query_signature(right)
        if left_numbers != right_numbers:
            return False
        if not left_words or not right_words:
            return left_words == right_words
        overlap = len(left_words & right_words) / min(len(left_words), len(right_words))
        return overlap >= 0.65

    def lookup(self, query: str) -> Dict[str, Any] | None:
        """Checks if a semantically equivalent query exists in Redis or memory cache."""
        q_vec = self._embed_query_fast(query)
        entries = self._load_entries()

        best_score = -1.0
        best_response: Dict[str, Any] | None = None

        for entry in entries:
            cached_query = str(entry.get("query", ""))
            if not self._compatible_queries(query, cached_query):
                continue
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
                self._redis_client.set(key, json.dumps(entry), ex=self.ttl_seconds)
                self._redis_client.sadd("vectracite:semcache:index", key)
                return
            except Exception as exc:
                self._mark_redis_unavailable(exc)

        self._memory_cache.append(entry)
        if len(self._memory_cache) > 512:
            del self._memory_cache[: len(self._memory_cache) - 512]

    def _load_entries(self) -> List[Dict[str, Any]]:
        if self._redis_client is not None:
            try:
                index_key = "vectracite:semcache:index"
                keys = list(self._redis_client.smembers(index_key))
                if not keys:
                    keys = list(
                        self._redis_client.scan_iter(
                            match="vectracite:semcache:*", count=200
                        )
                    )
                    keys = [
                        key for key in keys if key != index_key
                    ]
                    if keys:
                        self._redis_client.sadd(index_key, *keys)
                if not keys:
                    return []
                raw_vals = self._redis_client.mget(keys)
                entries: List[Dict[str, Any]] = []
                missing: List[str] = []
                for key, raw_value in zip(keys, raw_vals):
                    if raw_value is None:
                        missing.append(key)
                        continue
                    entries.append(json.loads(raw_value))
                if missing:
                    self._redis_client.srem(index_key, *missing)
                return entries
            except Exception as exc:
                self._mark_redis_unavailable(exc)
        return self._memory_cache

    def clear(self) -> None:
        """Clear cached responses in both Redis and the local fallback store."""
        if self._redis_client is not None:
            try:
                index_key = "vectracite:semcache:index"
                keys = set(self._redis_client.smembers(index_key))
                keys.update(
                    key
                    for key in self._redis_client.scan_iter(
                        match="vectracite:semcache:*", count=200
                    )
                    if key != index_key
                )
                if keys:
                    self._redis_client.delete(*keys)
                self._redis_client.delete(index_key)
            except Exception as exc:
                self._mark_redis_unavailable(exc)
        self._memory_cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": "redis" if self._redis_client is not None else "in_memory_fallback",
            "degraded": self._degraded,
            "last_error": self._last_error,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
        }
