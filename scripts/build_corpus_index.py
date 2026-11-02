"""Build the checked-in parsed-chunk cache for the real PDF corpus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.corpus_cache import load_corpus_chunks
from src.ingestion.pdf_parser import MultimodalFinancialParser
from src.retrieval.embedding_cache import load_embeddings, save_embeddings
from src.retrieval.hybrid_retriever import get_shared_embedder


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Also build the BGE embedding cache (slow on CPU, fast at startup).",
    )
    argument_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the committed cache without reparsing the PDFs.",
    )
    args = argument_parser.parse_args()
    manifest_path = settings.CORPUS_PDF_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing {manifest_path}; run scripts/fetch_real_pdf_corpus.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.verify:
        cached = json.loads(settings.CORPUS_INDEX_PATH.read_text(encoding="utf-8"))
        cached_docs = cached.get("documents") or {}
        manifest_docs = set(manifest.get("documents", {}))
        if set(cached_docs) != manifest_docs:
            raise SystemExit("Cached corpus documents do not match the manifest.")
        for doc_name, document in cached_docs.items():
            if not document.get("chunks"):
                raise SystemExit(f"{doc_name}: cached corpus has no chunks.")
            if document.get("sha256") != manifest["documents"][doc_name]["sha256"]:
                raise SystemExit(f"{doc_name}: cached corpus SHA256 does not match.")
        if not settings.EMBEDDING_CACHE_PATH.exists():
            raise SystemExit(f"Missing {settings.EMBEDDING_CACHE_PATH}.")
        print(f"OK {settings.CORPUS_INDEX_PATH} and {settings.EMBEDDING_CACHE_PATH}")
        return
    parser = MultimodalFinancialParser(
        chunk_size=settings.PARSER_CHUNK_SIZE,
        chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
    )

    started_at = time.perf_counter()
    chunks = load_corpus_chunks(
        settings.CORPUS_PDF_DIR,
        parser,
        cache_path=settings.CORPUS_INDEX_PATH,
        force=True,
        write=True,
    )
    elapsed = time.perf_counter() - started_at

    documents = sorted({chunk.doc_name for chunk in chunks})
    tables = sum(1 for chunk in chunks if chunk.contains_table)
    print(
        f"Built {settings.CORPUS_INDEX_PATH} for {len(documents)} documents, "
        f"{len(chunks)} chunks ({tables} table chunks) in {elapsed:.2f}s"
    )
    for doc_name in documents:
        doc_chunks = [chunk for chunk in chunks if chunk.doc_name == doc_name]
        doc_tables = sum(1 for chunk in doc_chunks if chunk.contains_table)
        source_pages = max((chunk.page_number for chunk in doc_chunks), default=0)
        print(
            f"  {doc_name}: {len(doc_chunks)} chunks, {doc_tables} table chunks, "
            f"max page {source_pages}"
        )

    cached = json.loads(settings.CORPUS_INDEX_PATH.read_text(encoding="utf-8"))
    cached_docs = set(cached.get("documents", {}))
    manifest_docs = set(manifest.get("documents", {}))
    if cached_docs != manifest_docs:
        raise SystemExit(
            f"Cache/manifest mismatch: {sorted(cached_docs)} != {sorted(manifest_docs)}"
        )
    print("Cache/manifest document sets match.")

    if args.embeddings:
        cached_vectors = load_embeddings(chunks)
        if cached_vectors is not None:
            print("Embedding cache already valid; nothing to do.")
            return
        embedder = get_shared_embedder(use_lightweight=False)
        embed_started_at = time.perf_counter()
        vectors = np.asarray(
            embedder.encode(
                [chunk.text for chunk in chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        save_embeddings(chunks, vectors)
        print(
            f"Built {settings.EMBEDDING_CACHE_PATH} with {len(vectors)} vectors "
            f"in {time.perf_counter() - embed_started_at:.2f}s"
        )


if __name__ == "__main__":
    main()
