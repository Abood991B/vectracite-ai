import json
import logging
from typing import Any, Dict, List
from datasets import load_dataset
from src.config import settings

logger = logging.getLogger(__name__)


class FinanceBenchLoader:
    """
    Loads verified ground-truth financial Q&A pairs from Hugging Face (PatronusAI/financebench).

    Fetching the filing PDFs themselves lives in ``scripts/fetch_real_pdf_corpus.py``, which
    verifies checksums against the corpus manifest.
    """

    def __init__(self) -> None:
        self.ground_truth_path = settings.DATA_DIR / "financebench_ground_truth.json"

    def load_ground_truth_samples(self, max_samples: int = 15) -> List[Dict[str, Any]]:
        """Fetches verified FinanceBench benchmark rows from Hugging Face."""
        logger.info("Loading PatronusAI/financebench dataset from Hugging Face...")
        ds = load_dataset("PatronusAI/financebench", split="train")

        samples: List[Dict[str, Any]] = []
        for row in ds:
            doc_name = row.get("doc_name", "")
            if not doc_name:
                continue

            samples.append(
                {
                    "financebench_id": row.get("financebench_id"),
                    "company": row.get("company"),
                    "doc_name": doc_name,
                    "question": row.get("question"),
                    "gold_answer": row.get("answer"),
                    "gold_evidence": row.get("evidence"),
                    "gold_page_num": row.get("page_number"),
                }
            )
            if len(samples) >= max_samples:
                break

        self.ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ground_truth_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d verified ground-truth Q&A pairs.", len(samples))
        return samples
