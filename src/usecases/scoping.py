"""Query and document scoping policy.

Pure functions: no settings, no retriever, no HTTP. The controller supplies the indexed document names
and maps `InvalidDocumentNameError` to a status code.
"""

from __future__ import annotations

import re
from typing import List, Sequence

COMPARISON_RE = re.compile(
    r"\b(compare|comparison|between|across|versus|vs\.?|both|each)\b",
    flags=re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:fy\s?)?((?:19|20)\d{2}|\d{2})\b", re.IGNORECASE)


class InvalidDocumentNameError(ValueError):
    """Raised when a document identifier cannot be used as a corpus filename."""


def clean_document_name(doc_name: str) -> str:
    """Normalize a document identifier before using it as a corpus filename."""
    clean_name = doc_name.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    if clean_name.lower().endswith(".pdf"):
        clean_name = clean_name[:-4]
    if not clean_name or clean_name in {".", ".."}:
        raise InvalidDocumentNameError("Document name cannot be empty.")
    if any(character in '<>:"/\\|?*' for character in clean_name):
        raise InvalidDocumentNameError("Document name contains invalid filename characters.")
    return clean_name


def normalize_scope(doc_name: str | None) -> str | None:
    """Normalize the optional query scope while treating ALL as global search."""
    if not doc_name or not doc_name.strip() or doc_name.strip().upper() == "ALL":
        return None
    return clean_document_name(doc_name)


def infer_scope_from_question(question: str, indexed_names: Sequence[str]) -> str | None:
    """Honor an explicit document reference only when it is unambiguous."""
    question_folded = question.casefold()
    question_compact = re.sub(r"[^a-z0-9]", "", question_folded)
    ordered_names: List[str] = sorted(set(indexed_names), key=len, reverse=True)
    if len(ordered_names) <= 1:
        return ordered_names[0] if ordered_names else None

    exact_matches = [
        doc_name
        for doc_name in ordered_names
        if re.search(rf"\b{re.escape(doc_name.casefold())}\b", question_folded)
    ]
    if exact_matches:
        return exact_matches[0] if len(exact_matches) == 1 else None

    matches: List[str] = []
    for doc_name in ordered_names:
        aliases = {
            doc_name.split("_", maxsplit=1)[0].casefold(),
        }
        compact_aliases = {
            re.sub(r"[^a-z0-9]", "", alias) for alias in aliases if len(alias) >= 2
        }
        if any(
            len(alias) >= 2
            and re.search(rf"\b{re.escape(alias)}\b", question_folded)
            for alias in aliases
        ) or any(alias and alias in question_compact for alias in compact_aliases):
            matches.append(doc_name)

    if len(matches) > 1:
        years = {
            year if len(year) == 4 else f"20{year}" for year in YEAR_RE.findall(question)
        }
        year_matches = [
            doc_name for doc_name in matches if any(year in doc_name for year in years)
        ]
        if len(year_matches) == 1:
            matches = year_matches

    if len(matches) != 1:
        return None
    if COMPARISON_RE.search(question):
        return None
    return matches[0]
