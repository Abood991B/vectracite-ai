"""Build benchmark artifacts for the checked-in real PDF corpus.

This script never creates or rewrites PDFs. It filters the verified FinanceBench
ground truth to the demo documents, calibrates FinanceBench's page numbering
against the physical PDF pages, and writes ``data/benchmark_queries.json`` plus a
normalized ``data/financebench_ground_truth.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pymupdf as fitz

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.dataset_loader import FinanceBenchLoader
from src.retrieval.financial_numbers import (
    display_financial_token,
    extract_number_tokens,
    parse_financial_number,
)

SELECTED_DOCS = {
    "AMD_2022_10K",
    "3M_2022_10K",
    "3M_2018_10K",
    "ADOBE_2022_10K",
    "ACTIVISIONBLIZZARD_2019_10K",
}

GROUND_TRUTH_PATH = settings.DATA_DIR / "financebench_ground_truth.json"
QUERIES_PATH = settings.DATA_DIR / "benchmark_queries.json"
def _gold_keywords(answer: str) -> List[str]:
    """Keep verifiable financial figures, excluding years and bare single digits."""
    keywords: set[str] = set()
    for token in extract_number_tokens(answer):
        parsed = parse_financial_number(token)
        if parsed is None:
            continue
        raw = token.strip(" ,")
        if (
            not parsed.is_percentage
            and parsed.scale == 1
            and parsed.value == parsed.value.to_integral_value()
            and 1900 <= abs(parsed.value) <= 2100
        ):
            continue
        if (
            not parsed.is_percentage
            and parsed.scale == 1
            and parsed.value == parsed.value.to_integral_value()
            and abs(parsed.value) < 10
        ):
            continue
        keywords.add(display_financial_token(raw))
    return sorted(keywords)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if len(token) >= 2}


def _containment(evidence_text: str, page_text: str) -> float:
    evidence = _tokens(evidence_text)
    if not evidence:
        return 0.0
    return len(evidence & _tokens(page_text)) / len(evidence)


def _detect_page_offset(pdf_path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Return 1 when FinanceBench page numbers are 0-based, else 0."""
    document = fitz.open(pdf_path)
    try:
        scores: Dict[int, List[float]] = {0: [], 1: []}
        for row in rows:
            for item in row.get("gold_evidence") or []:
                evidence_page = item.get("evidence_page_num")
                if not isinstance(evidence_page, int):
                    continue
                for offset in (0, 1):
                    index = evidence_page + offset - 1
                    if 0 <= index < document.page_count:
                        scores[offset].append(
                            _containment(
                                str(item.get("evidence_text", "")),
                                document[index].get_text("text", sort=True),
                            )
                        )
        means = {
            offset: (sum(values) / len(values) if values else 0.0)
            for offset, values in scores.items()
        }
        selected = max(means, key=means.get)
        if means[selected] < 0.35:
            raise RuntimeError(
                f"could not calibrate page numbering for {pdf_path.name}: {means}"
            )
        print(
            f"{pdf_path.name}: page offset {selected:+d} "
            f"(containment 1-based={means[0]:.3f}, 0-based={means[1]:.3f})"
        )
        return selected
    finally:
        document.close()


def load_ground_truth(refresh: bool) -> List[Dict[str, Any]]:
    if GROUND_TRUTH_PATH.exists() and not refresh:
        rows = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
        if {row["doc_name"] for row in rows} >= SELECTED_DOCS:
            return [row for row in rows if row["doc_name"] in SELECTED_DOCS]

    print("Fetching FinanceBench ground truth from Hugging Face...")
    rows = FinanceBenchLoader().load_ground_truth_samples(max_samples=10_000)
    filtered = [row for row in rows if row.get("doc_name") in SELECTED_DOCS]
    missing = SELECTED_DOCS - {row["doc_name"] for row in filtered}
    if missing:
        raise RuntimeError(f"FinanceBench is missing selected documents: {sorted(missing)}")
    return filtered


def normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    offsets: Dict[str, int] = {}
    for doc_name in sorted(SELECTED_DOCS):
        doc_rows = [row for row in rows if row["doc_name"] == doc_name]
        pdf_path = settings.CORPUS_PDF_DIR / f"{doc_name}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        offsets[doc_name] = _detect_page_offset(pdf_path, doc_rows)

    for row in rows:
        if row["doc_name"] not in SELECTED_DOCS:
            continue
        offset = offsets[row["doc_name"]]
        evidence = []
        for item in row.get("gold_evidence") or []:
            page = item.get("evidence_page_num")
            evidence.append(
                {
                    "doc_name": item.get("doc_name", row["doc_name"]),
                    "evidence_page_num": page,
                    "physical_page_num": page + offset if isinstance(page, int) else None,
                    "evidence_text": item.get("evidence_text", ""),
                    "evidence_text_full_page": item.get("evidence_text_full_page", ""),
                }
            )
        normalized.append(
            {
                "financebench_id": row.get("financebench_id"),
                "company": row.get("company"),
                "doc_name": row["doc_name"],
                "question": row["question"],
                "gold_answer": row.get("gold_answer"),
                "gold_evidence": evidence,
                "gold_page_num": row.get("gold_page_num"),
            }
        )
    return sorted(normalized, key=lambda item: (item["doc_name"], item["financebench_id"] or ""))


def build_queries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    for row in rows:
        pages = [
            item["physical_page_num"]
            for item in row["gold_evidence"]
            if isinstance(item.get("physical_page_num"), int)
        ]
        if not pages:
            continue
        answer = str(row.get("gold_answer") or "")
        queries.append(
            {
                "doc_name": row["doc_name"],
                "page_number": pages[0],
                "evidence_pages": sorted(set(pages)),
                "question": row["question"],
                "gold_answer": answer,
                "gold_keywords": _gold_keywords(answer),
                "financebench_id": row.get("financebench_id"),
            }
        )
    return queries


def build(refresh: bool) -> None:
    rows = load_ground_truth(refresh=refresh)
    normalized = normalize_rows(rows)
    queries = build_queries(normalized)

    GROUND_TRUTH_PATH.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    QUERIES_PATH.write_text(
        json.dumps(queries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    docs = sorted({row["doc_name"] for row in normalized})
    print(f"wrote {len(normalized)} ground-truth rows for {len(docs)} documents")
    print(f"wrote {len(queries)} benchmark queries to {QUERIES_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch FinanceBench ground truth from Hugging Face before filtering.",
    )
    args = parser.parse_args()
    build(refresh=args.refresh)


if __name__ == "__main__":
    main()
