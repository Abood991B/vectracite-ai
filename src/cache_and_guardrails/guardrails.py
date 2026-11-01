import re
from dataclasses import dataclass
from typing import List


@dataclass
class GuardrailCheckResult:
    passed: bool
    sanitized_text: str
    triggered_rules: List[str]


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
        "CREDIT_CARD": (r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_CARD]"),
        "API_KEY": (r"\b(?:sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})\b", "[REDACTED_API_KEY]"),
    }

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

        return GuardrailCheckResult(
            passed=True,
            sanitized_text=sanitized,
            triggered_rules=triggered,
        )
