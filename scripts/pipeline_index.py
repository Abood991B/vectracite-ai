"""KFP component: rebuild the shared Qdrant index from a chunks artifact."""

import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from src.config import settings
from src.ingestion.pdf_parser import DocumentChunk
from src.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--qdrant-host", default=settings.QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=settings.QDRANT_PORT)
    parser.add_argument("--collection", default=settings.QDRANT_COLLECTION)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    chunks = [DocumentChunk(**item) for item in payload]
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, timeout=30.0)
    retriever = HybridRetriever(
        collection_name=args.collection,
        in_memory=False,
        client=client,
        allow_fallback=False,
    )
    retriever.clear()
    grouped: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_name, []).append(chunk)
    for doc_name in sorted(grouped):
        retriever.replace_document(doc_name, grouped[doc_name])

    report = {
        "collection": args.collection,
        "documents": len(grouped),
        "chunks": len(chunks),
        "qdrant_url": f"{args.qdrant_host}:{args.qdrant_port}",
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
