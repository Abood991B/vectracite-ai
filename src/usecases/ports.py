"""Ports owned by the application layer.

The use cases define what they need; adapters implement it. Nothing here imports a framework, a
database driver, or a vendor SDK, so a use case can be tested with plain fakes and a vendor can be
swapped without touching the policy that uses it.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Protocol, Sequence

from src.domain.documents import DocumentChunk, RetrievedDocument


class QueryBlockedError(Exception):
    """Raised when input guardrails refuse a question. Delivery layers decide the response code."""

    def __init__(self, rules: Sequence[str]) -> None:
        self.rules = list(rules)
        super().__init__("Query blocked by Enterprise AI Guardrails.")


class RetrieverPort(Protocol):
    """Evidence lookup over the indexed corpus."""

    def retrieve_hybrid(
        self, query: str, top_k: int, doc_name: str | None = None
    ) -> List[RetrievedDocument]: ...

    def indexed_document_names(self) -> List[str]: ...

    def list_documents(self) -> List[Dict[str, Any]]: ...

    def get_document_chunks(self, doc_name: str) -> List[Dict[str, Any]]: ...

    def corpus_vocabulary(self) -> Dict[str, int]: ...

    def table_candidates(
        self, doc_name: str, terms: set[str], limit: int = 40
    ) -> List[DocumentChunk]: ...


class RerankerPort(Protocol):
    """Optional reordering of retrieved evidence."""

    def rerank(
        self, query: str, candidates: List[RetrievedDocument], top_n: int
    ) -> List[RetrievedDocument]: ...


class AnswerCachePort(Protocol):
    """Response cache keyed by scope, top-k, and sanitized question."""

    def lookup(self, key: str) -> Dict[str, Any] | None: ...

    def store(self, key: str, response: Dict[str, Any]) -> None: ...


class SynthesizerPort(Protocol):
    """Turns evidence and a question into an answer, locally or through a provider."""

    last_provider: str
    last_error: str | None

    async def generate_answer(
        self, question: str, chunks: List[RetrievedDocument], indexed_docs: List[str] | None = None
    ) -> str: ...

    def stream_answer(
        self, question: str, chunks: List[RetrievedDocument], indexed_docs: List[str] | None = None
    ) -> AsyncIterator[str]: ...
