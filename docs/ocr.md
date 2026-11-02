# OCR status and measurements

OCR is optional and disabled by default. Digital PDFs use PyMuPDF's native text
layer. Extraction can miss or misorder content. OCR is only attempted for pages with images and
less than `OCR_MIN_CHARS` extractable text.

When OCR is enabled, a page limit, OCR error, or empty OCR result rejects the
document. Exactly white image pages are skipped and reported as blank; other
blank or near-blank image pages can still trigger rejection. OCR-disabled mixed PDFs
and pages with enough native text can still contain unrecognized image content.
Check the source pages before relying on the index.
The 120-second ingestion timeout may be too short for local OCR, especially
for multiple pages. Adjust `INGEST_TIMEOUT_SECONDS` after measuring your own
documents. A timed-out OCR call may continue until the call returns.

## Measured candidates on this workstation

| Engine | Hardware | Crop load | Crop inference | Result |
|---|---|---:|---:|---|
| PaddleOCR PP-OCRv5 mobile (det+rec) | Ryzen 7 4800H CPU | 5.42 s | 16.40 s | 1,135 chars, mean confidence 0.9518 |
| PaddleOCR-VL-1.6 Transformers | GTX 1660 Ti 6 GB | 6.97 s | 415.66 s | readable Markdown, not interactive |
| OvisOCR2 vLLM 0.22.1 | GTX 1660 Ti 6 GB | model loaded | did not finish in 15 min | not viable on 6 GB |
| TeleOCR 1.2B | GTX 1660 Ti 6 GB | not executed | not executed | vLLM/layout stack exceeds practical budget |

The crop is 1157x772 px from a real CDC MMWR page. A full page is larger, so the
CPU OCR path is only suitable for low-volume fallback. The VLM results do not meet
an interactive ingestion target on this GPU.

## Recommendation

- Keep `OCR_ENABLED=false` for the 6 GB workstation.
- Use the native text path for normal PDFs.
- For scanned production workloads, deploy a remote GPU OCR service and set:
  `OCR_ENABLED=true`, `OCR_ENGINE=remote`, `OCR_REMOTE_URL=...`.
- For low-volume local fallback, install `requirements-ocr.txt` and set
  `OCR_ENGINE=paddleocr`. The PP-OCRv5 mobile pipeline returns text but does not
  preserve table structure.
- Do not enable PaddleOCR-VL or OvisOCR2 on a 6 GB card for interactive ingestion.

## Adapter

`src/ingestion/ocr.py` provides:

- `PaddleOCRBackend`: optional local PP-OCRv5 pipeline.
- `RemoteOCRBackend`: posts a PNG to a service and accepts `{text}` or `{markdown}`.
- `build_ocr_backend()`: disabled, `auto`, `paddleocr`, or `remote`.

Parser metadata records `ocr_used` and `ocr_engine`. The corpus cache signature
includes the OCR engine, DPI, and minimum character threshold.
