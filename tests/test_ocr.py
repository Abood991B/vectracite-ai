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


def _image_only_pdf(*, blank: bool = False) -> bytes:
    document = fitz.open()
    page = document.new_page(width=400, height=400)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 400))
    pixmap.clear_with(255)
    page.insert_image(fitz.Rect(0, 0, 400, 400), stream=pixmap.tobytes("png"))
    if not blank:
        page.draw_rect(fitz.Rect(10, 10, 30, 30), fill=(0, 0, 0))
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


def test_ocr_page_limit_rejects_incomplete_document() -> None:
    import pytest

    image_pdf = fitz.open(stream=_image_only_pdf(), filetype="pdf")
    document = fitz.open()
    document.insert_pdf(image_pdf)
    document.insert_pdf(image_pdf)
    payload = document.tobytes()
    document.close()
    image_pdf.close()
    parser = MultimodalFinancialParser(ocr_backend=FakeOCR(), ocr_max_pages=1)
    with pytest.raises(ValueError, match="OCR.*limit.*page 2"):
        parser.parse_pdf_bytes(payload, "SCANNED_TEST")


def test_short_ocr_result_still_counts_toward_page_limit() -> None:
    import pytest

    class ShortOCR:
        name = "short"

        def recognize(self, image_png):
            return OCRResult(text="123", engine=self.name)

    source = fitz.open(stream=_image_only_pdf(), filetype="pdf")
    document = fitz.open()
    document.insert_pdf(source)
    document.insert_pdf(source)
    payload = document.tobytes()
    document.close()
    source.close()
    with pytest.raises(ValueError, match="OCR.*limit.*page 2"):
        MultimodalFinancialParser(ocr_backend=ShortOCR(), ocr_max_pages=1).parse_pdf_bytes(
            payload, "SCAN"
        )


def test_blank_image_is_distinguished_from_ocr_failure() -> None:
    import pytest

    parser = MultimodalFinancialParser(ocr_backend=FakeOCR())
    with pytest.raises(ValueError, match="only blank image pages"):
        parser.parse_pdf_bytes(_image_only_pdf(blank=True), "BLANK")


def test_nonblank_scan_requires_ocr_when_disabled() -> None:
    import pytest

    parser = MultimodalFinancialParser()
    with pytest.raises(ValueError, match="Scanned or image-only"):
        parser.parse_pdf_bytes(_image_only_pdf(), "SCAN")


def test_parser_honors_cancellation_before_processing_pages() -> None:
    import threading
    import pytest

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ValueError, match="processing cancelled"):
        MultimodalFinancialParser().parse_pdf_bytes(
            _image_only_pdf(), "SCAN", cancel_event=cancelled
        )


def test_mixed_document_reports_image_pages_without_text() -> None:
    from fastapi.testclient import TestClient
    import base64
    from src.api.main import app

    source = fitz.open(stream=_image_only_pdf(blank=True), filetype="pdf")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 80), "A readable page with enough native text for indexing.")
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20))
    pixmap.clear_with(0)
    page.insert_image(fitz.Rect(50, 100, 70, 120), stream=pixmap.tobytes("png"))
    document.insert_pdf(source)
    payload = document.tobytes()
    document.close()
    source.close()
    client = TestClient(app)
    try:
        response = client.post("/ingest/pdf", json={
            "doc_name": "MIXED_COVERAGE_TEST",
            "pdf_base64": base64.b64encode(payload).decode("ascii"),
        })
        assert response.status_code == 200
        pages = response.json()["coverage"]["pages"]
        assert pages[0]["status"] == "native_text"
        assert pages[1]["status"] == "blank_image"
        assert response.json()["coverage"]["pages_without_text"] == []
        assert response.json()["coverage"]["pages_with_unrecognized_images"] == [1]
    finally:
        client.delete("/documents/MIXED_COVERAGE_TEST")
