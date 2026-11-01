import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import httpx
from datasets import load_dataset
from src.config import settings

logger = logging.getLogger(__name__)

PATRONUS_GITHUB_PDF_BASE = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
)


class FinanceBenchLoader:
    """
    Loads verified ground-truth financial Q&A pairs from Hugging Face (PatronusAI/financebench)
    and downloads the corresponding official SEC 10-K/10-Q PDF filings.
    """

    def __init__(self, pdf_dir: Path = settings.PDF_DIR) -> None:
        self.pdf_dir = pdf_dir
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
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

    def download_sec_pdfs(self, doc_names: List[str]) -> List[Path]:
        """Downloads official SEC filing PDFs from the Patronus AI FinanceBench corpus."""
        downloaded_paths: List[Path] = []
        unique_docs = sorted(list(set(doc_names)))

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            for doc_name in unique_docs:
                filename = f"{doc_name}.pdf" if not doc_name.endswith(".pdf") else doc_name
                local_path = self.pdf_dir / filename

                if local_path.exists() and local_path.stat().st_size > 10_000:
                    downloaded_paths.append(local_path)
                    continue

                url = f"{PATRONUS_GITHUB_PDF_BASE}/{filename}"
                logger.info("Downloading SEC filing from %s", url)
                response = client.get(url)

                if response.status_code == 200:
                    local_path.write_bytes(response.content)
                    downloaded_paths.append(local_path)
                else:
                    logger.warning("Failed to fetch %s (HTTP %d)", filename, response.status_code)

        return downloaded_paths
