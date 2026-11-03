"""Correct likely typos against the indexed corpus vocabulary.

Generic by construction: the vocabulary comes from whatever documents are indexed, so this works for
uploaded PDFs in any domain rather than a hand-written finance word list.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9$%.-]*")
MIN_LENGTH = 5
MIN_CORPUS_COUNT = 5


def _within_one_edit(left: str, right: str) -> bool:
    """True when the edit distance is at most one (substitution, insertion, deletion)."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        differences = [
            index
            for index, (a, b) in enumerate(zip(left, right))
            if a != b
        ]
        if len(differences) == 1:
            return True
        if len(differences) == 2 and differences[1] == differences[0] + 1:
            first, second = differences
            return left[first] == right[second] and left[second] == right[first]
        return False
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    index = 0
    while index < len(shorter) and shorter[index] == longer[index]:
        index += 1
    return shorter[index:] == longer[index + 1 :]


def correct_query_terms(question: str, vocabulary: Mapping[str, int]) -> str:
    """Replace out-of-vocabulary lowercase words with a close, frequent corpus term."""
    if not vocabulary:
        return question
    buckets: Dict[tuple[str, int], list[str]] = {}
    known_terms: list[str] = []
    for term, count in vocabulary.items():
        if count >= MIN_CORPUS_COUNT and len(term) >= MIN_LENGTH:
            buckets.setdefault((term[:1], len(term)), []).append(term)
            known_terms.append(term)

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        if len(word) < MIN_LENGTH or word.lower() in vocabulary:
            return word
        if word[:1].isupper():
            return word
        lowered = word.lower()
        # A word the corpus builds longer terms from is a fragment of the language, not a typo:
        # "count" sits inside "account", so correcting it to the nearest edit neighbour ("court")
        # would turn "employee count" into "employee court".
        if any(lowered in term and len(term) > len(lowered) for term in known_terms):
            return word
        candidates: list[str] = []
        for length in (len(lowered) - 1, len(lowered), len(lowered) + 1):
            candidates.extend(buckets.get((lowered[:1], length), []))
        best = next(
            (candidate for candidate in candidates if _within_one_edit(lowered, candidate)),
            None,
        )
        return best or word

    return WORD_PATTERN.sub(replace, question)


def vocabulary_from_texts(texts: Iterable[str]) -> Dict[str, int]:
    """Build a term-frequency map; the retriever supplies one per index generation."""
    counts: Dict[str, int] = {}
    for text in texts:
        for word in WORD_PATTERN.findall(text):
            lowered = word.lower()
            if len(lowered) >= MIN_LENGTH:
                counts[lowered] = counts.get(lowered, 0) + 1
    return counts
