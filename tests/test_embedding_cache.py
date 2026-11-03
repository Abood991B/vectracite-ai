"""Pin the per-chunk embedding cache.

A cache built from the checked-in filings must keep serving its chunks after a user upload joins the
index. Without that, every restart re-embeds the whole corpus and can overwrite the tracked cache file
with the user's own text.
"""

import numpy as np

from qdrant_client import QdrantClient

from src.config import settings
from src.domain.documents import DocumentChunk
from src.retrieval.embedding_cache import load_embeddings, save_embeddings
from src.retrieval.hybrid_retriever import HybridRetriever

DIM = settings.EMBEDDING_DIM


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_name="CACHE_DOC",
        page_number=1,
        text=text,
        contains_table=False,
        metadata={"chunk_index": 0},
    )


def _vectors(count: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, DIM)).astype(np.float32)


def test_a_cache_with_extra_entries_still_serves_the_corpus_chunks(tmp_path) -> None:
    cache = tmp_path / "embeddings.npz"
    corpus = [_chunk("a", "alpha"), _chunk("b", "beta")]
    save_embeddings(corpus + [_chunk("upload", "mine")], _vectors(3), cache_path=cache)

    loaded = load_embeddings(corpus, cache_path=cache)

    assert loaded is not None
    assert loaded.shape == (2, DIM)


def test_indexing_corpus_plus_an_upload_embeds_only_the_upload(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "embeddings.npz"
    monkeypatch.setattr(settings, "EMBEDDING_CACHE_PATH", cache)
    corpus = [_chunk("a", "alpha"), _chunk("b", "beta")]
    save_embeddings(corpus, _vectors(2), cache_path=cache)
    retriever = HybridRetriever(
        collection_name="embedding_cache_test",
        in_memory=True,
        client=QdrantClient(":memory:"),
        use_lightweight_embedder=False,
    )
    encoded: list[str] = []

    def fake_encode(texts):
        encoded.extend(texts)
        return _vectors(len(texts), seed=7)

    monkeypatch.setattr(retriever, "_encode_texts", fake_encode)

    retriever.index_chunks(corpus + [_chunk("upload", "my private paragraph")])

    assert encoded == ["my private paragraph"]


def test_startup_indexes_without_writing_the_tracked_cache(monkeypatch) -> None:
    from src.api import main as api_main

    captured: dict[str, object] = {}
    monkeypatch.setattr(api_main.retriever, "chunks_store", [])
    monkeypatch.setattr(api_main, "_load_persisted_chunks", lambda *args, **kwargs: [_chunk("a", "x")])

    def fake_index(chunks, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(api_main.retriever, "index_chunks", fake_index)

    api_main._ensure_indexed()

    assert captured["use_embedding_cache"] is True
    assert captured["write_embedding_cache"] is False
