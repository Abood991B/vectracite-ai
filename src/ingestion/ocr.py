"""Optional OCR engines for image-only PDF pages."""

from __future__ import annotations

import base64
import io
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRResult:
    text: str
    engine: str
    confidence: float | None = None


class OCRBackend(Protocol):
    name: str

    def recognize(self, image_png: bytes) -> OCRResult:
        ...


class PaddleOCRBackend:
    """Local PaddleOCR PP-OCRv5 pipeline.

    The mobile detector/recognizer are used by default because the server models
    are substantially slower on CPU. GPU deployments can override the model names.
    """

    name = "paddleocr"

    def __init__(
        self,
        *,
        language: str = settings.OCR_LANGUAGES,
        det_model: str = settings.OCR_PADDLE_DET_MODEL,
        rec_model: str = settings.OCR_PADDLE_REC_MODEL,
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "OCR_ENGINE=paddleocr requires the optional OCR dependencies. "
                "Install requirements-ocr.txt."
            ) from exc
        kwargs: dict[str, Any] = {
            "text_detection_model_name": det_model or None,
            "text_recognition_model_name": rec_model or None,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "enable_mkldnn": False,
        }
        if not det_model and not rec_model:
            kwargs["lang"] = language
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No ccache found.*")
            self._pipeline = PaddleOCR(**kwargs)

    def recognize(self, image_png: bytes) -> OCRResult:
        from PIL import Image

        image = Image.open(io.BytesIO(image_png)).convert("RGB")
        results = list(self._pipeline.predict(np.asarray(image)))
        if not results:
            return OCRResult(text="", engine=self.name)
        first = results[0]
        payload: Any = getattr(first, "json", first)
        if isinstance(payload, dict) and "res" in payload:
            payload = payload["res"]
        if not isinstance(payload, dict):
            return OCRResult(text="", engine=self.name)
        texts = [str(text).strip() for text in payload.get("rec_texts") or []]
        scores = [float(score) for score in payload.get("rec_scores") or []]
        text = "\n".join(item for item in texts if item)
        confidence = round(sum(scores) / len(scores), 4) if scores else None
        return OCRResult(text=text, engine=self.name, confidence=confidence)


class RemoteOCRBackend:
    """Call a separate OCR service with a PNG payload."""

    name = "remote"

    def __init__(
        self,
        url: str = settings.OCR_REMOTE_URL,
        timeout_seconds: float = settings.OCR_TIMEOUT_SECONDS,
    ) -> None:
        if not url.strip():
            raise RuntimeError("OCR_ENGINE=remote requires OCR_REMOTE_URL.")
        self.url = url.strip()
        self.timeout_seconds = timeout_seconds

    def recognize(self, image_png: bytes) -> OCRResult:
        response = httpx.post(
            self.url,
            json={
                "image_base64": base64.b64encode(image_png).decode("ascii"),
                "format": "png",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("text") or payload.get("markdown") or "").strip()
        confidence = payload.get("confidence")
        return OCRResult(
            text=text,
            engine=str(payload.get("engine") or self.name),
            confidence=float(confidence) if confidence is not None else None,
        )


def build_ocr_backend() -> OCRBackend | None:
    """Build the configured engine without making OCR mandatory for text PDFs."""
    if not settings.OCR_ENABLED:
        return None
    engine = settings.OCR_ENGINE.strip().lower()
    if engine == "paddleocr":
        return PaddleOCRBackend()
    if engine == "remote":
        return RemoteOCRBackend()
    if engine == "auto":
        if settings.OCR_REMOTE_URL.strip():
            return RemoteOCRBackend()
        try:
            return PaddleOCRBackend()
        except RuntimeError as exc:
            logger.warning("OCR auto-detection disabled: %s", exc)
            return None
    raise ValueError(f"Unsupported OCR_ENGINE: {settings.OCR_ENGINE}")
