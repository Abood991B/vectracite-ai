from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from src.retrieval.financial_numbers import (
    canonical_number_values,
    canonical_numbers_in_text,
    display_financial_token,
    extract_number_tokens,
)
from src.domain.documents import RetrievedDocument


@dataclass
class NumericVerificationResult:
    """Deterministic audit report verifying numerical claims against retrieved table cells."""

    is_fully_grounded: bool
    numeric_grounding_score: float
    verified_figures: List[str]
    unverified_figures: List[str]
    verification_applicable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeterministicNumericVerifier:
    """
    Symbolic & Numerical Grounding Engine that extracts quantitative tokens (currencies,
    percentages, ratios, and multi-digit figures) from generated answers and verifies their
    exact presence inside the retrieved source chunks and Markdown tables.
    """

    @classmethod
    def _extract_tokens(cls, text: str, *, ignore_page_references: bool = False) -> List[str]:
        """Extract quantitative claims while optionally excluding citation page metadata.

        Standalone years are intentionally treated as claims and must be present in a source.
        Explicit ``p. 12``/``page 12`` references are citation metadata, so answer-side page
        numbers are ignored rather than diluting the grounding score.
        """
        return extract_number_tokens(text, ignore_page_references=ignore_page_references)

    def verify(
        self, answer: str, retrieved_chunks: List[RetrievedDocument], doc_names: List[str] | None = None
    ) -> NumericVerificationResult:
        """Cross-checks every quantitative figure in `answer` against `retrieved_chunks`."""
        raw_matches = self._extract_tokens(answer, ignore_page_references=True)
        if not retrieved_chunks:
            return NumericVerificationResult(
                is_fully_grounded=False,
                numeric_grounding_score=0.0,
                verified_figures=[],
                unverified_figures=[],
                verification_applicable=False,
            )
        if not raw_matches:
            return NumericVerificationResult(
                is_fully_grounded=False,
                numeric_grounding_score=0.0,
                verified_figures=[],
                unverified_figures=[],
                verification_applicable=False,
            )

        corpus_numbers = canonical_numbers_in_text(
            "\n".join(chunk.text for chunk in retrieved_chunks)
        )
        derived_numbers: set[str] = set()
        from src.retrieval.table_ast import TableASTParser

        for chunk in retrieved_chunks:
            if not chunk.contains_table:
                continue
            cells = TableASTParser.extract_cells(
                chunk.text,
                doc_name=chunk.doc_name,
                page_number=chunk.page_number,
            )
            for delta in TableASTParser.compute_row_deltas(cells):
                for key in ("absolute_change", "pct_change"):
                    derived_numbers.update(canonical_number_values(delta[key]))

        known_numbers = corpus_numbers | derived_numbers
        verified: List[str] = []
        unverified: List[str] = []
        seen = set()

        for raw_token in raw_matches:
            clean_tok = raw_token.strip()
            candidates = canonical_number_values(clean_tok)
            if not candidates:
                continue
            signature = tuple(sorted(candidates))
            if signature in seen:
                continue
            seen.add(signature)
            display_token = display_financial_token(clean_tok)
            if candidates & known_numbers:
                verified.append(display_token)
            else:
                unverified.append(display_token)

        total = len(verified) + len(unverified)
        score = round(len(verified) / total, 4) if total > 0 else 1.0
        return NumericVerificationResult(
            is_fully_grounded=len(unverified) == 0,
            numeric_grounding_score=score,
            verified_figures=verified,
            unverified_figures=unverified,
            verification_applicable=True,
        )
