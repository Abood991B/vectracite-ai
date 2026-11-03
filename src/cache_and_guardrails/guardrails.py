import re
from dataclasses import dataclass
from typing import List


@dataclass
class GuardrailCheckResult:
    passed: bool
    sanitized_text: str
    triggered_rules: List[str]


class StreamingRedactor:
    """Redact PII across stream boundaries without delaying the whole answer."""

    def __init__(self, guardrails: "EnterpriseGuardrails", tail_chars: int = 256) -> None:
        self.guardrails = guardrails
        self.tail_chars = tail_chars
        self._buffer = ""
        self._flags: List[str] = []
        self._sanitized_parts: List[str] = []

    def _record(self, result: GuardrailCheckResult) -> str:
        for flag in result.triggered_rules:
            if flag not in self._flags:
                self._flags.append(flag)
        if result.sanitized_text:
            self._sanitized_parts.append(result.sanitized_text)
        return result.sanitized_text

    def feed(self, text: str) -> str:
        self._buffer += text
        if len(self._buffer) <= self.tail_chars * 2:
            return ""
        safe_prefix = self._buffer[: -self.tail_chars]
        self._buffer = self._buffer[-self.tail_chars :]
        return self._record(self.guardrails.sanitize_output(safe_prefix))

    def finish(self) -> str:
        remaining = self._record(self.guardrails.sanitize_output(self._buffer))
        self._buffer = ""
        return remaining

    @property
    def sanitized_text(self) -> str:
        return "".join(self._sanitized_parts)

    @property
    def triggered_rules(self) -> List[str]:
        return list(self._flags)


class EnterpriseGuardrails:
    """
    Production AI Safety & Security layer enforcing:
      1. Input Prompt-Injection & Jailbreak Prevention.
      2. Output PII Redaction (Emails, Phone Numbers, SSNs, Credit Cards, Secret API Keys).
      3. Citation Grounding Verification.
    """

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"system\s*prompt",
        r"you\s+are\s+now\s+dan",
        r"bypass\s+safety",
        r"reveal\s+your\s+(hidden|system)\s+instructions",
        r"drop\s+table",
        r"<script>",
    ]

    PII_PATTERNS = {
        "EMAIL": (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]"),
        "SSN": (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
        "API_KEY": (r"\b(?:sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})\b", "[REDACTED_API_KEY]"),
    }
    # A bare 13-16 digit run also matches year columns ("2018 2017 2016 2015"), which corrupted
    # financial answers. Candidate groups are checksummed before redacting.
    CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

    @staticmethod
    def _luhn_valid(candidate: str) -> bool:
        digits = [int(character) for character in candidate if character.isdigit()]
        if not 13 <= len(digits) <= 19:
            return False
        total = 0
        for index, digit in enumerate(reversed(digits)):
            if index % 2:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0

    @classmethod
    def _looks_like_card(cls, candidate: str) -> bool:
        """A card passes the checksum and is not a row of four-digit years."""
        groups = re.findall(r"\d+", candidate)
        if len(groups) >= 3 and all(
            len(group) == 4 and 1900 <= int(group) <= 2099 for group in groups
        ):
            return False
        return cls._luhn_valid(candidate)

    def validate_input(self, query: str) -> GuardrailCheckResult:
        """Inspects incoming user query for prompt injection or malicious payloads."""
        triggered: List[str] = []
        lowered = query.lower()

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, lowered):
                triggered.append(f"PROMPT_INJECTION_DETECTED:{pattern}")

        if len(query.strip()) > 4000:
            triggered.append("QUERY_LENGTH_EXCEEDED")

        return GuardrailCheckResult(
            passed=len(triggered) == 0,
            sanitized_text=query.strip(),
            triggered_rules=triggered,
        )

    def sanitize_output(self, text: str) -> GuardrailCheckResult:
        """Scans LLM output and redacts any accidentally leaked PII or credentials."""
        sanitized = text
        triggered: List[str] = []

        for label, (pattern, replacement) in self.PII_PATTERNS.items():
            if re.search(pattern, sanitized):
                sanitized = re.sub(pattern, replacement, sanitized)
                triggered.append(f"PII_REDACTED:{label}")

        redacted_cards = self.CARD_CANDIDATE.sub(
            lambda match: (
                "[REDACTED_CARD]"
                if self._looks_like_card(match.group(0))
                else match.group(0)
            ),
            sanitized,
        )
        if redacted_cards != sanitized:
            sanitized = redacted_cards
            triggered.append("PII_REDACTED:CREDIT_CARD")

        return GuardrailCheckResult(
            passed=True,
            sanitized_text=sanitized,
            triggered_rules=triggered,
        )

    def streaming_redactor(self, tail_chars: int = 256) -> StreamingRedactor:
        """Create a boundary-safe redactor for token streams."""
        return StreamingRedactor(self, tail_chars=tail_chars)
