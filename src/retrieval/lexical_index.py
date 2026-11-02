"""BM25 lexical ranking over the indexed chunk catalog.

Owns the tokenizer, the stopword list, and the BM25 state, so the retriever can ask for ranked
lexical candidates without knowing how lexical scoring works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.domain.documents import DocumentChunk

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


def tokenize(text: str) -> List[str]:
    """Split text into BM25 terms, dropping stopwords and stripping possessives."""
    cleaned = re.sub(r"['\u2019]s\b", "", text.lower())
    tokens: List[str] = []
    for token in re.findall(r"[\w$\.%]+", cleaned, flags=re.UNICODE):
        if len(token) < 2 and not token.isdigit():
            continue
        if token in BM25_STOPWORDS:
            continue
        tokens.append(token)
        parts = re.findall(r"[a-z]+|\d+", token)
        if len(parts) > 1:
            tokens.extend(part for part in parts if len(part) >= 2 or part.isdigit())
    return tokens


@dataclass(frozen=True)
class _LexicalState:
    chunks: Tuple[DocumentChunk, ...]
    bm25: BM25Okapi


class LexicalIndex:
    """A rebuildable BM25 index that scores against one immutable chunk snapshot."""

    def __init__(self) -> None:
        self._state: _LexicalState | None = None

    @property
    def ready(self) -> bool:
        return self._state is not None

    def rebuild(self, chunks: Sequence[DocumentChunk]) -> None:
        """Replace the index with one built over ``chunks``."""
        if not chunks:
            self._state = None
            return
        corpus = [tokenize(chunk.text) for chunk in chunks]
        self._state = _LexicalState(tuple(chunks), BM25Okapi(corpus))

    def ranked(
        self, query: str, *, limit: int, scope: str | None = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Return up to ``limit`` chunks ordered by BM25 score, optionally scoped to a document."""
        state = self._state
        if state is None:
            return []
        scores = state.bm25.get_scores(tokenize(query))
        indices = np.arange(len(state.chunks))
        if scope:
            indices = np.asarray(
                [index for index, chunk in enumerate(state.chunks) if chunk.doc_name == scope],
                dtype=int,
            )
        ordered = indices[np.argsort(scores[indices])[::-1]][:limit]
        return [(state.chunks[int(index)], float(scores[int(index)])) for index in ordered]
