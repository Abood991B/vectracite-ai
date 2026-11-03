"""Recognise the scalar attributes a question asks for.

A headcount sentence states a whole number of people: "employed 93,516 people",
"approximately 9,200 total full-time and part-time employees". Retrieval ranks those pages
badly because "employees" appears in every compensation and stock-plan page, so the retriever's
attribute scan and narrative synthesis share one definition here instead of each keeping its own
regex.
"""

from __future__ import annotations

import re
from typing import Optional

COUNT_QUESTION_RE = re.compile(
    r"\b(how many|how much|number of|count|headcount)\b",
    flags=re.IGNORECASE,
)
# A count question is only a table question when it also names a statement metric. Otherwise a bare
# filing year ("employee count in FY2022") routes a headcount question into table synthesis, which
# then answers from an equity-plan or retirement-plan row.
FINANCIAL_METRIC_RE = re.compile(
    r"\b(revenue|sales|income|profit|margins?|capex|capital expenditure|expenses?|cash|"
    r"cashflow|assets?|liabilit\w*|dividends?|debt|earnings|eps|shares?|turnover)\b",
    flags=re.IGNORECASE,
)
PERSON_COUNT_NOUNS = (
    r"employees|people|staff|workforce|headcount|full-time equivalents"
)
PERSON_COUNT_RE = re.compile(
    rf"\b(\d[\d,]*(?:\.\d+)?)\s+(?:[\w\-']+\s+){{0,4}}?(?:{PERSON_COUNT_NOUNS})\b",
    flags=re.IGNORECASE,
)
PERSON_ATTRIBUTE_TERMS = frozenset(
    {
        "employee",
        "employees",
        "employed",
        "people",
        "staff",
        "workforce",
        "headcount",
        "equivalents",
    }
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def is_count_question(question: str) -> bool:
    """True when the question asks for a whole-number quantity."""
    return bool(COUNT_QUESTION_RE.search(question))


def asks_for_people_count(question: str) -> bool:
    """True when the question asks how many people the document reports."""
    if not is_count_question(question):
        return False
    words = {word.casefold() for word in re.findall(r"[\w']+", question)}
    return bool(PERSON_ATTRIBUTE_TERMS & words)


def person_count_value(line: str) -> Optional[str]:
    """Return the people count a line states, or None when it does not state one.

    Three digits or more separates a headcount ("9,200 employees") from executive ages
    ("58 Chief People Officer") and plan percentages ("3 percent ... for employees").
    """
    for match in PERSON_COUNT_RE.finditer(line):
        number = match.group(1).rstrip(".,")
        digits = re.sub(r"\D", "", number)
        if len(digits) < 3 or _YEAR_RE.fullmatch(digits):
            continue
        return number
    return None
