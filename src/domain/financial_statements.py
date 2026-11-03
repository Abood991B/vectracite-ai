"""Recognise the page that carries a company's consolidated primary financial statements.

An annual report states the same figure several times: once in the audited statement, and again
in MD&A tables, notes, and selected-financial-data summaries. For a question about a reported
figure the statement is the source, so evidence selection needs to tell the page roles apart.

Page role is taken from the page's own opening chunk, because a heading is a title at the top of
the page. Testing every chunk would also match prose that mentions the statement ("expense was
allocated in the consolidated statements of operations as follows"), audit reports that describe
the statements they cover, and tables of contents that list several statements. Callers therefore
pass the opening chunk only.
"""

from __future__ import annotations

import re

__all__ = ["asked_statement_families", "is_statement_heading", "statement_family", "statement_family_count"]

# Matching runs on letters only, because extraction breaks words: 3M's FY2018 heading extracts
# as "Consolidated Statement of Incom e". Digits and punctuation are dropped with them. Every
# family accepts the plural, because filings title the statements in both numbers.
STATEMENT_FAMILIES = (
    r"statements?ofoperations",
    r"statements?ofincome",
    r"statements?ofearnings",
    r"statements?ofcomprehensiveincome",
    r"statements?ofcashflows",
    r"balancesheets?",
    r"statements?offinancialposition",
    r"statements?ofchangesin",
)
STATEMENT_HEADING_RE = re.compile(
    r"consolidated(?:" + "|".join(STATEMENT_FAMILIES) + r")"
)
FAMILY_RES = tuple(re.compile(family) for family in STATEMENT_FAMILIES)
STATEMENT_HEADING_WINDOW = 400

# Words a title may carry in lower case. Anything else in lower case means the heading was
# reached through a sentence, as in an audit report or a note, and the page is not a statement.
TITLE_CONNECTIVES = frozenset(
    {"and", "of", "the", "for", "inc", "corp", "corporation", "company", "plc", "ltd", "co"}
)


def _letter_indexed(text: str) -> tuple[str, list[int]]:
    """The text's letters, plus each letter's position in the original string."""
    letters: list[str] = []
    positions: list[int] = []
    for position, character in enumerate(text):
        if character.isalpha():
            letters.append(character.lower())
            positions.append(position)
    return "".join(letters), positions


def _is_title_prefix(prefix: str) -> bool:
    """True when the text before a heading reads as a page title, not as part of a sentence."""
    for word in prefix.split():
        if not any(character.isalpha() for character in word):
            continue
        if word.isupper() or word[:1].isupper():
            continue
        if word.casefold() in TITLE_CONNECTIVES:
            continue
        return False
    return True


def statement_family_count(text: str) -> int:
    """How many different consolidated statements the text's opening names."""
    flattened, _ = _letter_indexed(text[:STATEMENT_HEADING_WINDOW])
    return sum(1 for family in FAMILY_RES if family.search(flattened))


def statement_family(text: str) -> str | None:
    """The one consolidated statement this page opens, or None when it opens none.

    Pass the page's opening chunk. The family name is the canonical regex fragment, so a caller can
    match a question that names a statement ("the cash flow statement") against the page that is
    that statement.
    """
    head = text[:STATEMENT_HEADING_WINDOW]
    if statement_family_count(head) > 1:
        return None
    flattened, positions = _letter_indexed(head)
    match = STATEMENT_HEADING_RE.search(flattened)
    if match is None:
        return None
    if not _is_title_prefix(head[: positions[match.start()]]):
        return None
    return next((family for family in STATEMENT_FAMILIES if re.search(family, flattened)), None)


def is_statement_heading(text: str) -> bool:
    """True when this chunk opens a page that is itself a consolidated statement.

    Pass the page's opening chunk. A page that lists several statements is a table of contents,
    and a heading reached through a sentence belongs to prose about the statements.
    """
    return statement_family(text) is not None


# What a question says when it names the statement it wants answered from. A question that names
# none is answered from whichever statement carries the figure.
ASKED_STATEMENT_PATTERNS = (
    (r"cash ?flows?", ("statements?ofcashflows",)),
    (
        r"balance ?sheets?|financial position",
        ("balancesheets?", "statements?offinancialposition"),
    ),
    (
        r"income statements?|statements? of (?:operations|income|earnings)|\boperations\b",
        ("statements?ofoperations", "statements?ofincome", "statements?ofearnings"),
    ),
    (r"comprehensive income", ("statements?ofcomprehensiveincome",)),
    (
        r"changes in equity|equity statement|stockholders'? equity|shareholders'? equity",
        ("statements?ofchangesin",),
    ),
)


def asked_statement_families(question: str) -> frozenset[str]:
    """The statement families a question names, empty when it names none."""
    folded = question.casefold()
    return frozenset(
        family
        for pattern, families in ASKED_STATEMENT_PATTERNS
        if re.search(pattern, folded)
        for family in families
    )
