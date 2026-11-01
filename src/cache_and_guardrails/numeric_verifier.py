import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List
from src.retrieval.hybrid_retriever import RetrievedDocument


@dataclass
class NumericVerificationResult:
    """Deterministic audit report verifying numerical claims against retrieved table cells."""

    is_fully_grounded: bool
    numeric_grounding_score: float
    verified_figures: List[str]
    unverified_figures: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeterministicNumericVerifier:
    """
    Symbolic & Numerical Grounding Engine that extracts quantitative tokens (currencies,
    percentages, ratios, and multi-digit figures) from generated answers and verifies their
    exact presence inside the retrieved source chunks and Markdown tables.
    """

    NUMERIC_TOKEN_REGEX = re.compile(
        r"(?:\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?%|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b)"
    )

    @staticmethod
    def _normalize_number(token: str) -> str:
        return re.sub(r"[\s$,]", "", token.strip()).lower()

    def verify(
        self, answer: str, retrieved_chunks: List[RetrievedDocument], doc_names: List[str] | None = None
    ) -> NumericVerificationResult:
        """Cross-checks every quantitative figure in `answer` against `retrieved_chunks`."""
        raw_Matches = self.NUMERIC_TOKEN_REGEX.findall(answer)
        if not raw_Matches:
            return NumericVerificationResult(
                is_fully_grounded=True,
                numeric_grounding_score=1.0,
                verified_figures=[],
                unverified_figures=[],
            )

        corpus_text = " ".join(c.text for c in retrieved_chunks)
        normalized_corpus = self._normalize_number(corpus_text)

        verified: List[str] = []
        unverified: List[str] = []
        seen = set()

        for raw_token in raw_Matches:
            clean_tok = raw_token.strip()
            norm_tok = self._normalize_number(clean_tok)
            if not norm_tok or norm_tok in seen:
                continue
            seen.add(norm_tok)

            if norm_tok in normalized_corpus:
                verified.append(clean_tok)
            else:
                unverified.append(clean_tok)

        total = len(verified) + len(unverified)
        score = round(len(verified) / total, 4) if total > 0 else 1.0
        return NumericVerificationResult(
            is_fully_grounded=len(unverified) == 0,
            numeric_grounding_score=score,
            verified_figures=verified,
            unverified_figures=unverified,
        )
