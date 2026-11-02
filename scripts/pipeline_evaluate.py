"""KFP component: evaluate the shared index and enforce a retrieval quality gate."""

import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from evaluation.run_ablation_benchmark import run_evaluation_suite
from src.config import settings
from src.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-host", default=settings.QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=settings.QDRANT_PORT)
    parser.add_argument("--collection", default=settings.QDRANT_COLLECTION)
    parser.add_argument("--min-recall-at-3", type=float, default=0.90)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, timeout=30.0)
    retriever = HybridRetriever(
        collection_name=args.collection,
        in_memory=False,
        client=client,
        allow_fallback=False,
    )
    retriever.hydrate_from_qdrant()
    report = run_evaluation_suite(retriever.chunks_store)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    recall = float(report["hybrid_plus_reranker"]["recall_at_3"])
    print(f"Recall@3={recall:.4f}; gate={args.min_recall_at_3:.4f}")
    if recall < args.min_recall_at_3:
        raise SystemExit(
            f"Quality gate failed: recall_at_3 {recall:.4f} < {args.min_recall_at_3:.4f}"
        )


if __name__ == "__main__":
    main()
