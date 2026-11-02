"""OCR adapter and image-only PDF fallback tests."""

from __future__ import annotations

import pymupdf as fitz

from src.config import settings
from src.ingestion.corpus_cache import _parser_signature
from src.ingestion.ocr import OCRResult, build_ocr_backend
from src.ingestion.pdf_parser import MultimodalFinancialParser


class FakeOCR:
    name = "fake"

    def recognize(self, image_png: bytes) -> OCRResult:
        assert image_png.startswith(b"\x89PNG")
        return OCRResult(
            text="FAKE OCR TEXT\nRevenue was 123 million.",
            engine=self.name,
            confidence=0.99,
        )


def _image_only_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page(width=400, height=400)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 400))
    pixmap.clear_with(255)
    page.insert_image(fitz.Rect(0, 0, 400, 400), stream=pixmap.tobytes("png"))
    payload = document.tobytes()
    document.close()
    return payload


def test_image_only_pdf_uses_configured_ocr_backend() -> None:
    parser = MultimodalFinancialParser(
        chunk_size=400,
        chunk_overlap=50,
        ocr_backend=FakeOCR(),
        ocr_min_chars=10,
    )
    chunks = parser.parse_pdf_bytes(_image_only_pdf(), "SCANNED_TEST")
    assert chunks
    assert "FAKE OCR TEXT" in chunks[0].text
    assert chunks[0].metadata["ocr_used"] is True
    assert chunks[0].metadata["ocr_engine"] == "fake"


def test_parser_signature_includes_ocr_configuration() -> None:
    disabled = MultimodalFinancialParser(chunk_size=400, chunk_overlap=50)
    enabled = MultimodalFinancialParser(
        chunk_size=400,
        chunk_overlap=50,
        ocr_backend=FakeOCR(),
        ocr_min_chars=10,
    )
    assert _parser_signature(disabled)["ocr_engine"] is None
    assert _parser_signature(enabled)["ocr_engine"] == "fake"
    assert _parser_signature(disabled) != _parser_signature(enabled)


def test_ocr_backend_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OCR_ENABLED", False)
    assert build_ocr_backend() is None


def test_remote_ocr_requires_url(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OCR_ENABLED", True)
    monkeypatch.setattr(settings, "OCR_ENGINE", "remote")
    monkeypatch.setattr(settings, "OCR_REMOTE_URL", "")
    try:
        build_ocr_backend()
    except RuntimeError as exc:
        assert "OCR_REMOTE_URL" in str(exc)
    else:
        raise AssertionError("remote OCR should require a URL")
