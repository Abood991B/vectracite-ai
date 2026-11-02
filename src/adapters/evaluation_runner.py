"""Adapter for the offline evaluation harness.

The harness lives outside `src/` and is a development tool, not application policy. Confining its
imports here keeps the API layer from depending on `evaluation` directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluation.run_ablation_benchmark import (
    build_dynamic_eval_queries,
    compute_evaluation_signature,
    run_evaluation_suite,
)
from src.domain.documents import DocumentChunk

__all__ = ["build_live_cases", "compute_signature", "run_suite"]


def build_live_cases(chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
    """Return the scenario cases the harness can derive from indexed chunks."""
    return build_dynamic_eval_queries(chunks)


def compute_signature(chunks: List[DocumentChunk]) -> str:
    """Identify the indexed content an evaluation report was produced from."""
    return compute_evaluation_signature(chunks)


def run_suite(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Run the ablation benchmark over the current index."""
    return run_evaluation_suite(chunks)
