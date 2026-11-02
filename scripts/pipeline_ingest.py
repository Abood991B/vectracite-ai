"""KFP component: parse the corpus into a portable chunks artifact."""

import argparse
import json
from pathlib import Path

from src.ingestion.pdf_parser import MultimodalFinancialParser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pdf_parser = MultimodalFinancialParser(chunk_size=900, chunk_overlap=150)
    chunks = []
    for pdf_path in sorted(Path(args.corpus_dir).glob("*.pdf")):
        chunks.extend(pdf_parser.parse_pdf(pdf_path))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Ingested {len(chunks)} chunks from {args.corpus_dir}")


if __name__ == "__main__":
    main()
