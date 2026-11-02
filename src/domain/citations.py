"""The citation grammar, owned in one place.

Answers cite evidence as ``[DOC_NAME, Page N]``. That grammar is consumed by two very different
callers — the pipeline that filters cited evidence and the adapter that validates a remote provider's
answer — so it belongs to the domain rather than to either of them.
"""

from __future__ import annotations

import re
from typing import Set, Tuple

CITATION_PATTERN = re.compile(
    r"\[([^,\]]+),\s*(?:page|p)\.?\s*(\d+)\]",
    flags=re.IGNORECASE,
)


def parse_citations(text: str) -> Set[Tuple[str, int]]:
    """Return every ``(document, page)`` pair cited in ``text``."""
    return {
        (match.group(1).strip(), int(match.group(2)))
        for match in CITATION_PATTERN.finditer(text)
    }
