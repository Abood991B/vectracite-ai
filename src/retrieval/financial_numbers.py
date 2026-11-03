"""Shared structural parsing for financial figures.

The API, table arithmetic, and numeric verifier must agree on what a number
means. This module is the single source of truth for signs, accounting
parentheses, percentages, thousands separators, currency symbols, and scale
suffixes. It intentionally performs structural comparison rather than substring
matching so ``123`` can never be treated as grounded by ``1234``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Iterable, List, Set


PAGE_REFERENCE_REGEX = re.compile(
    r"(?i)\b(?:p(?:age)?s?\.?)\s*\d+(?:\s*[-\u2013\u2014]\s*\d+)?"
)

_SCALE_ALIASES = (
    (r"trillion", Decimal("1000000000000")),
    (r"billion", Decimal("1000000000")),
    (r"million", Decimal("1000000")),
    (r"thousand", Decimal("1000")),
    (r"tn", Decimal("1000000000000")),
    (r"bn", Decimal("1000000000")),
    (r"mn", Decimal("1000000")),
    (r"t", Decimal("1000000000000")),
    (r"b", Decimal("1000000000")),
    (r"m", Decimal("1000000")),
    (r"k", Decimal("1000")),
)
_SCALE_PATTERN = r"(?:" + "|".join(alias for alias, _ in _SCALE_ALIASES) + r")"

# Currency symbols and common ISO codes are accepted but not used for equality.
_CURRENCY_SYMBOLS = "".join(chr(codepoint) for codepoint in (0x20AC, 0x00A3, 0x00A5))
_CURRENCY_PREFIX = (
    "(?:US\\$|USD|EUR|GBP|CAD|AUD|JPY|CNY|CHF|[$" + _CURRENCY_SYMBOLS + "])"
)
_CORE_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_TOKEN_PATTERN = (
    r"(?<![\w.])"
    r"(?:"
    rf"\(\s*[+-]?\s*{_CURRENCY_PREFIX}?\s*{_CORE_NUMBER}\s*(?:%|percent)?"
    rf"(?:\s*{_SCALE_PATTERN})?\s*\)"
    rf"|[+-]?\s*{_CURRENCY_PREFIX}?\s*{_CORE_NUMBER}\s*(?:%|percent)?"
    rf"(?:\s*{_SCALE_PATTERN})?"
    r")"
    r"(?![\w]|\.\d|-[A-Za-z])"
)
NUMERIC_TOKEN_REGEX = re.compile(_TOKEN_PATTERN, flags=re.IGNORECASE)
_MISSING_MARKERS = frozenset(
    {
        "",
        "-",
        "--",
        "\u2014",
        "\u2013",
        "n/a",
        "na",
        "n.m.",
        "nm",
        "not available",
        "not applicable",
        "*",
    }
)


@dataclass(frozen=True)
class FinancialNumber:
    """A parsed financial figure with enough metadata for display and comparison."""

    raw: str
    value: Decimal
    scale: Decimal
    is_percentage: bool
    accounting_negative: bool
    explicit_sign: bool

    @property
    def scaled_value(self) -> Decimal:
        return self.value * self.scale

    def canonical_values(self) -> Set[str]:
        """Return every structurally legitimate canonical form of this figure.

        A token such as ``$23,601M`` is ambiguous without its table header: it
        can mean 23,601 (already expressed in millions) or 23,601,000,000. Both
        interpretations are accepted for grounding, while arithmetic always uses
        the explicit scale.
        """
        if self.is_percentage:
            return {f"{_format_decimal(self.value)}%"}
        values = {_format_decimal(self.value)}
        if self.scale != 1:
            values.add(_format_decimal(self.scaled_value))
        return values


def _format_decimal(value: Decimal) -> str:
    if value == 0:
        value = abs(value)
    return format(value.normalize(), "f")


def _scale_and_strip(value: str) -> tuple[str, Decimal]:
    for alias, factor in _SCALE_ALIASES:
        match = re.search(rf"(?i){alias}\.?\s*$", value)
        if match:
            return value[: match.start()].strip(), factor
    return value, Decimal("1")


def parse_financial_number(token: str) -> FinancialNumber | None:
    """Parse one financial token, returning ``None`` for prose or missing cells."""
    raw = token.strip()
    if raw.casefold() in _MISSING_MARKERS:
        return None

    accounting_negative = raw.startswith("(") and raw.endswith(")")
    if accounting_negative:
        raw = raw[1:-1].strip()

    is_percentage = bool(re.search(r"(?i)(?:%|percent)\s*$", raw))
    raw = re.sub(r"(?i)\s*(?:%|percent)\s*$", "", raw).strip()
    raw, scale = _scale_and_strip(raw)
    raw = re.sub(
        rf"^([+-]?)\s*{_CURRENCY_PREFIX}\s*",
        r"\1",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    raw = raw.replace(",", "").replace(" ", "")

    explicit_sign = raw.startswith(("+", "-"))
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None

    if accounting_negative and not explicit_sign:
        value = -abs(value)

    return FinancialNumber(
        raw=token.strip(),
        value=value,
        scale=scale,
        is_percentage=is_percentage,
        accounting_negative=accounting_negative,
        explicit_sign=explicit_sign,
    )


def _has_claim_shape(token: str) -> bool:
    """Reject single bare digits while retaining years, money, and percentages."""
    stripped = token.strip("() ")
    if "%" in stripped or re.search(r"(?i)(?:percent|thousand|million|billion|trillion)", stripped):
        return True
    if re.search(r"[$" + _CURRENCY_SYMBOLS + r"]|USD|EUR|GBP", stripped, flags=re.IGNORECASE):
        return True
    if re.search(r"[.,]", stripped):
        return True
    digits = re.sub(r"\D", "", stripped)
    return len(digits) >= 2


def extract_number_tokens(text: str, *, ignore_page_references: bool = False) -> List[str]:
    """Extract candidate quantitative claims from free text in reading order."""
    if ignore_page_references:
        text = PAGE_REFERENCE_REGEX.sub("", text)
    return [
        match.group(0).strip()
        for match in NUMERIC_TOKEN_REGEX.finditer(text)
        if _has_claim_shape(match.group(0))
    ]


def display_financial_token(token: str) -> str:
    """Return the figure as written without a trailing scale word or suffix."""
    parsed = parse_financial_number(token)
    if parsed is None or parsed.is_percentage or parsed.scale == 1:
        return token.strip()
    stripped = re.sub(rf"(?i)\s*{_SCALE_PATTERN}\.?\s*$", "", token).strip()
    return stripped or token.strip()


def canonical_number_values(token: str) -> Set[str]:
    """Return canonical comparison values for one token, or an empty set."""
    parsed = parse_financial_number(token)
    if parsed is None:
        return set()
    return parsed.canonical_values()


def canonical_numbers_in_text(text: str, *, ignore_page_references: bool = False) -> Set[str]:
    """Return all canonical numeric values present in text."""
    values: Set[str] = set()
    for token in extract_number_tokens(text, ignore_page_references=ignore_page_references):
        values.update(canonical_number_values(token))
    return values


def format_financial_value(
    value: Decimal,
    *,
    scale: Decimal = Decimal("1"),
    is_percentage: bool = False,
    signed: bool = True,
) -> str:
    """Format a deterministic value for the answer payload."""
    sign = ""
    if signed and value > 0:
        sign = "+"
    suffix = "%" if is_percentage else _scale_suffix(scale)
    return f"{sign}{value:,.2f}{suffix}"


def _scale_suffix(scale: Decimal) -> str:
    if scale == Decimal("1000000000000"):
        return "T"
    if scale == Decimal("1000000000"):
        return "B"
    if scale == Decimal("1000000"):
        return "M"
    if scale == Decimal("1000"):
        return "K"
    return ""


def common_scale(numbers: Iterable[FinancialNumber]) -> Decimal:
    """Return a shared display scale when all figures use the same scale."""
    scales = {number.scale for number in numbers if not number.is_percentage}
    return scales.pop() if len(scales) == 1 else Decimal("1")
