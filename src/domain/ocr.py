"""OCR domain types.

The parser needs to ask "give me the text in this image"; it should not know whether that is answered
by PaddleOCR, a remote service, or a test double. This module owns that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class OCRResult:
    """Text recognised from one page image."""

    text: str
    engine: str
    confidence: float | None = None


class OCRBackend(Protocol):
    """Port for an OCR provider."""

    name: str

    def recognize(self, image_png: bytes) -> OCRResult: ...
