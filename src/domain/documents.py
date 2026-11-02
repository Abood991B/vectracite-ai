"""Domain objects for document evidence.

These types are deliberately dependency-free: no settings, no PDF library, no vector store, no web
framework. Adapters construct them; use cases consume them. Keeping them here is what lets the answer
pipeline and the verifier be tested without any of those details.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class DocumentChunk:
    """A layout- and section-aware document chunk with an exact page citation."""

    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    contains_table: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedDocument:
    """A retrieved chunk scored by hybrid search and, optionally, the reranker."""

    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    contains_table: bool
    dense_score: float
    bm25_score: float
    rrf_score: float
    rerank_score: float | None = None


__all__ = ["DocumentChunk", "RetrievedDocument"]
