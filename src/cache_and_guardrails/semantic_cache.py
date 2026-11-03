import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Set

import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from src.adapters.redis_clients import CACHE_TIMEOUT_SECONDS, build_redis_client
from src.config import settings
from src.retrieval.financial_numbers import (
    canonical_number_values,
    extract_number_tokens,
)

logger = logging.getLogger(__name__)

CACHE_NAMESPACE = "vectracite:semcache:physical-pages-v4:"
EXACT_PREFIX = CACHE_NAMESPACE + "exact:"
ENTRY_PREFIX = CACHE_NAMESPACE + "entry:"
LEGACY_PREFIX = CACHE_NAMESPACE + "legacy:"
LEGACY_INDEX_KEY = CACHE_NAMESPACE + "index"
VECTOR_INDEX_NAME = CACHE_NAMESPACE + "vector"
VECTOR_DIMENSION = 256


class SemanticCache:
    """Redis-backed exact and semantic cache with a bounded local fallback."""

    def __init__(
        self,
        redis_url: str = settings.REDIS_URL,
        similarity_threshold: float = settings.CACHE_SIMILARITY_THRESHOLD,
        ttl_seconds: int = settings.CACHE_TTL_SECONDS,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.redis_url = redis_url
        self._last_reconnect_attempt = 0.0
        self._memory_cache: List[Dict[str, Any]] = []
        self._redis_client: redis.Redis | None = None
        self._redis_vector_client: redis.Redis | None = None
        self._vector_index_ready = False
        self._degraded = False
        self._last_error: str | None = None
        self.hits = 0
        self.misses = 0
        self.exact_hits = 0
        self.semantic_hits = 0

        try:
            self._connect_clients()
        except Exception as exc:
            self._mark_redis_unavailable(exc)

    def _connect_clients(self) -> None:
        """Open the cache and vector connections and rebuild the vector index."""
        client = build_redis_client(self.redis_url, timeout=CACHE_TIMEOUT_SECONDS)
        client.ping()
        self._redis_client = client
        self._degraded = False
        self._last_error = None
        try:
            vector_client = build_redis_client(
                self.redis_url,
                timeout=CACHE_TIMEOUT_SECONDS,
                decode_responses=False,
            )
            vector_client.ping()
            self._redis_vector_client = vector_client
            self._ensure_vector_index()
        except Exception as exc:
            logger.info("Redis vector cache unavailable: %s", type(exc).__name__)
            self._redis_vector_client = None

    def _mark_redis_unavailable(self, exc: Exception) -> None:
        """Disable Redis after an outage so readiness exposes the degraded state."""
        self._redis_client = None
        self._redis_vector_client = None
        self._vector_index_ready = False
        self._degraded = True
        self._last_error = f"{type(exc).__name__}: {exc}"[:500]
        log_unavailable = logger.warning if settings.REQUIRE_REDIS else logger.info
        log_unavailable("Semantic cache Redis backend unavailable: %s", type(exc).__name__)

    def _ensure_vector_index(self) -> None:
        if self._redis_vector_client is None:
            return
        search = self._redis_vector_client.ft(VECTOR_INDEX_NAME)
        try:
            search.info()
            self._vector_index_ready = True
            return
        except Exception:
            pass
        try:
            search.create_index(
                fields=[
                    TextField("query"),
                    VectorField(
                        "vector",
                        "HNSW",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": VECTOR_DIMENSION,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                ],
                definition=IndexDefinition(
                    prefix=[ENTRY_PREFIX],
                    index_type=IndexType.HASH,
                ),
            )
            self._vector_index_ready = True
        except Exception as exc:
            logger.info("Redis vector index creation failed: %s", type(exc).__name__)
            self._vector_index_ready = False

    def check_health(self) -> bool:
        """Probe Redis before readiness decisions, and re-establish a lost connection."""
        if self._redis_client is None:
            return self._try_reconnect()
        try:
            self._redis_client.ping()
            self._degraded = False
            self._last_error = None
            return True
        except Exception as exc:
            self._mark_redis_unavailable(exc)
            return False

    def _try_reconnect(self) -> bool:
        """Retry a lost Redis connection at most once every five seconds."""
        now = time.monotonic()
        if now - self._last_reconnect_attempt < 5.0:
            return False
        self._last_reconnect_attempt = now
        try:
            self._connect_clients()
        except Exception as exc:
            self._mark_redis_unavailable(exc)
            return False
        logger.info("Semantic cache Redis backend reconnected")
        return self._redis_client is not None

    @staticmethod
    def _embed_query_fast(query: str, dim: int = VECTOR_DIMENSION) -> np.ndarray:
        """Compute a normalized character/word n-gram vector for fast lookup."""
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
        """Return numeric and lexical signatures used to reject near-miss hits."""
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
        left_key = cls._split_cache_key(left)
        right_key = cls._split_cache_key(right)
        if left_key and right_key:
            left_scope, left_top_k, left_question = left_key
            right_scope, right_top_k, right_question = right_key
            if left_scope != right_scope or left_top_k != right_top_k:
                return False
            left, right = left_question, right_question
        left_numbers, left_words = cls._query_signature(left)
        right_numbers, right_words = cls._query_signature(right)
        if left_numbers != right_numbers:
            return False
        if not left_words or not right_words:
            return left_words == right_words
        overlap = len(left_words & right_words) / min(len(left_words), len(right_words))
        return overlap >= 0.65

    @staticmethod
    def _split_cache_key(key: str) -> tuple[str, str, str] | None:
        """Split ``scope::top_k=N::question``; legacy entries without the shape return None."""
        parts = key.split("::", 2)
        if len(parts) != 3:
            return None
        return parts[0], parts[1], parts[2]

    @staticmethod
    def _exact_key(query: str) -> str:
        return EXACT_PREFIX + hashlib.sha256(query.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _entry_key(query: str) -> str:
        return ENTRY_PREFIX + hashlib.sha256(query.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _cache_result(
        self,
        response: Dict[str, Any],
        similarity: float,
        *,
        exact: bool,
    ) -> Dict[str, Any]:
        self.hits += 1
        if exact:
            self.exact_hits += 1
        else:
            self.semantic_hits += 1
        result = dict(response)
        result["cache_hit"] = True
        result["cache_similarity"] = round(similarity, 4)
        return result

    def _exact_lookup(self, query: str) -> Dict[str, Any] | None:
        exact_key = self._exact_key(query)
        if self._redis_client is not None:
            try:
                raw = self._redis_client.get(exact_key)
            except Exception as exc:
                self._mark_redis_unavailable(exc)
            else:
                if raw:
                    try:
                        return self._cache_result(json.loads(raw), 1.0, exact=True)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        logger.warning("Ignoring malformed exact cache entry")
        for entry in reversed(self._memory_cache):
            if entry.get("query") == query:
                return self._cache_result(entry["response"], 1.0, exact=True)
        return None

    def _vector_lookup(self, query: str, q_vec: np.ndarray) -> Dict[str, Any] | None:
        if not self._vector_index_ready or self._redis_vector_client is None:
            return None
        try:
            search_query = (
                Query("*=>[KNN 12 @vector $vec AS distance]")
                .return_fields("query", "response", "distance")
                .dialect(2)
            )
            results = self._redis_vector_client.ft(VECTOR_INDEX_NAME).search(
                search_query,
                query_params={"vec": q_vec.astype(np.float32).tobytes()},
            )
        except Exception as exc:
            logger.info("Redis vector cache lookup failed: %s", type(exc).__name__)
            return None
        best_score = -1.0
        best_response: Dict[str, Any] | None = None
        for document in results.docs:
            cached_query = self._decode(getattr(document, "query", ""))
            if not self._compatible_queries(query, cached_query):
                continue
            distance = float(self._decode(getattr(document, "distance", "1")))
            similarity = 1.0 - distance
            if similarity > best_score:
                best_score = similarity
                best_response = json.loads(self._decode(document.response))
        if best_score >= self.similarity_threshold and best_response is not None:
            return self._cache_result(best_response, best_score, exact=False)
        return None

    def _legacy_entries(self) -> List[Dict[str, Any]]:
        if self._redis_client is None:
            return list(self._memory_cache)
        try:
            keys = [
                key
                for key in self._redis_client.scan_iter(
                    match=f"{LEGACY_PREFIX}*", count=200
                )
                if not key.startswith((EXACT_PREFIX, ENTRY_PREFIX, LEGACY_INDEX_KEY))
            ]
            if not keys:
                return []
            raw_vals = self._redis_client.mget(keys)
            entries: List[Dict[str, Any]] = []
            for raw in raw_vals:
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(entry, dict) and "query" in entry and "response" in entry:
                    entries.append(entry)
            return entries
        except Exception as exc:
            self._mark_redis_unavailable(exc)
            return list(self._memory_cache)

    def _legacy_lookup(self, query: str, q_vec: np.ndarray) -> Dict[str, Any] | None:
        best_score = -1.0
        best_response: Dict[str, Any] | None = None
        for entry in self._legacy_entries():
            cached_query = str(entry.get("query", ""))
            if not self._compatible_queries(query, cached_query):
                continue
            cached_vec = np.asarray(entry["vector"], dtype=np.float32)
            similarity = float(np.dot(q_vec, cached_vec))
            if similarity > best_score:
                best_score = similarity
                best_response = entry["response"]
        if best_score >= self.similarity_threshold and best_response is not None:
            return self._cache_result(best_response, best_score, exact=False)
        return None

    def lookup(self, query: str) -> Dict[str, Any] | None:
        """Check exact, Redis vector, legacy, then in-memory cache paths."""
        exact = self._exact_lookup(query)
        if exact is not None:
            return exact
        q_vec = self._embed_query_fast(query)
        vector_hit = self._vector_lookup(query, q_vec)
        if vector_hit is not None:
            return vector_hit
        legacy_hit = self._legacy_lookup(query, q_vec)
        if legacy_hit is not None:
            return legacy_hit
        self.misses += 1
        return None

    def _memory_store(self, query: str, entry: Dict[str, Any]) -> None:
        self._memory_cache.append(entry)
        if len(self._memory_cache) > 512:
            del self._memory_cache[: len(self._memory_cache) - 512]

    def store(self, query: str, response: Dict[str, Any]) -> None:
        """Store exact and semantic entries in Redis, falling back to memory."""
        q_vec = self._embed_query_fast(query).astype(np.float32)
        entry = {"query": query, "vector": q_vec.tolist(), "response": response}
        raw_response = json.dumps(response, ensure_ascii=False)
        if self._redis_client is not None:
            try:
                exact_key = self._exact_key(query)
                entry_key = self._entry_key(query)
                pipeline = self._redis_client.pipeline()
                pipeline.set(exact_key, raw_response, ex=self.ttl_seconds)
                pipeline.execute()
                if (
                    self._redis_vector_client is not None
                    and self._vector_index_ready
                ):
                    self._redis_vector_client.hset(
                        entry_key,
                        mapping={
                            "query": query,
                            "vector": q_vec.tobytes(),
                            "response": raw_response,
                        },
                    )
                    self._redis_vector_client.expire(entry_key, self.ttl_seconds)
                else:
                    legacy_key = (
                        LEGACY_PREFIX
                        + hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
                    )
                    pipeline.set(
                        legacy_key,
                        json.dumps(entry, ensure_ascii=False),
                        ex=self.ttl_seconds,
                    )
                    pipeline.execute()
                return
            except Exception as exc:
                self._mark_redis_unavailable(exc)
        self._memory_store(query, entry)

    def clear(self) -> None:
        """Clear exact, semantic, legacy, and local fallback entries."""
        if self._redis_client is not None:
            try:
                keys = [
                    key
                    for key in self._redis_client.scan_iter(
                        match=f"{CACHE_NAMESPACE}*", count=200
                    )
                    if key != LEGACY_INDEX_KEY
                ]
                if keys:
                    self._redis_client.delete(*keys)
                self._redis_client.delete(LEGACY_INDEX_KEY)
            except Exception as exc:
                self._mark_redis_unavailable(exc)
        if self._redis_vector_client is not None:
            try:
                vector_keys = [
                    self._decode(key)
                    for key in self._redis_vector_client.scan_iter(
                        match=f"{ENTRY_PREFIX}*", count=200
                    )
                ]
                if vector_keys:
                    self._redis_vector_client.delete(*vector_keys)
            except Exception as exc:
                logger.info("Redis vector cache clear failed: %s", type(exc).__name__)
        self._memory_cache.clear()
        self.hits = 0
        self.misses = 0
        self.exact_hits = 0
        self.semantic_hits = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": "redis" if self._redis_client is not None else "in_memory_fallback",
            "vector_index": self._vector_index_ready,
            "degraded": self._degraded,
            "last_error": self._last_error,
            "hits": self.hits,
            "misses": self.misses,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
        }
