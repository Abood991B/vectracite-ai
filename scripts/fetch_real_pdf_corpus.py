"""Download and verify the checked-in real SEC filing corpus.

The demo corpus is a small slice of the public FinanceBench PDF collection. This
script is the only supported way to populate ``data/corpus_pdfs``. It verifies
file size, page count, PDF signature, and SHA256 before replacing a document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pymupdf as fitz
import httpx

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings

PATRONUS_PDF_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
MANIFEST_SCHEMA_VERSION = 1

DEMO_DOCUMENTS: Dict[str, Dict[str, Any]] = {
    "AMD_2022_10K": {"pages": 121, "bytes": 5_122_473},
    "3M_2022_10K": {"pages": 252, "bytes": 1_655_681},
    "3M_2018_10K": {"pages": 160, "bytes": 1_252_586},
    "ADOBE_2022_10K": {"pages": 99, "bytes": 620_288},
    "ACTIVISIONBLIZZARD_2019_10K": {"pages": 198, "bytes": 1_274_941},
}

MANIFEST_NAME = "manifest.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pdf_pages(data: bytes) -> int:
    document = fitz.open(stream=data, filetype="pdf")
    try:
        return document.page_count
    finally:
        document.close()


def _validate_pdf(data: bytes, doc_name: str) -> None:
    if not data.startswith(b"%PDF-"):
        raise RuntimeError(f"{doc_name}: downloaded bytes are not a PDF")
    expected = DEMO_DOCUMENTS[doc_name]
    if len(data) != expected["bytes"]:
        raise RuntimeError(
            f"{doc_name}: size mismatch (expected {expected['bytes']}, got {len(data)})"
        )
    pages = _pdf_pages(data)
    if pages != expected["pages"]:
        raise RuntimeError(
            f"{doc_name}: page-count mismatch (expected {expected['pages']}, got {pages})"
        )


def _manifest_path() -> Path:
    return settings.CORPUS_PDF_DIR / MANIFEST_NAME


def load_manifest() -> Dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source": PATRONUS_PDF_BASE,
            "documents": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _download(doc_name: str, client: httpx.Client) -> bytes:
    url = f"{PATRONUS_PDF_BASE}/{doc_name}.pdf"
    response = client.get(url)
    response.raise_for_status()
    return response.content


def fetch(force: bool) -> None:
    settings.CORPUS_PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["source"] = PATRONUS_PDF_BASE
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("documents", {})

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for doc_name, expected in DEMO_DOCUMENTS.items():
            target = settings.CORPUS_PDF_DIR / f"{doc_name}.pdf"
            if target.exists() and not force:
                data = target.read_bytes()
                _validate_pdf(data, doc_name)
                digest = _sha256(data)
                previous = manifest["documents"].get(doc_name, {}).get("sha256")
                if previous and previous != digest:
                    raise RuntimeError(
                        f"{doc_name}: existing file does not match manifest; use --force"
                    )
                print(f"verified existing {target.name} ({len(data)} bytes)")
            else:
                data = _download(doc_name, client)
                _validate_pdf(data, doc_name)
                temp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                try:
                    temp_path.write_bytes(data)
                    os.replace(temp_path, target)
                finally:
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                print(f"downloaded {target.name} ({len(data)} bytes)")

            manifest["documents"][doc_name] = {
                "filename": target.name,
                "bytes": len(data),
                "pages": _pdf_pages(data),
                "sha256": _sha256(data),
                "source_url": f"{PATRONUS_PDF_BASE}/{doc_name}.pdf",
            }

    _manifest_path().write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {_manifest_path()}")


def verify() -> None:
    manifest = load_manifest()
    documents = manifest.get("documents", {})
    if set(documents) != set(DEMO_DOCUMENTS):
        raise RuntimeError(
            "manifest documents do not match the expected demo set: "
            f"{sorted(documents)} != {sorted(DEMO_DOCUMENTS)}"
        )
    for doc_name, expected in DEMO_DOCUMENTS.items():
        target = settings.CORPUS_PDF_DIR / f"{doc_name}.pdf"
        if not target.exists():
            raise FileNotFoundError(target)
        data = target.read_bytes()
        _validate_pdf(data, doc_name)
        digest = _sha256(data)
        recorded = documents[doc_name]
        if recorded.get("sha256") != digest:
            raise RuntimeError(f"{doc_name}: SHA256 mismatch")
        if recorded.get("bytes") != len(data):
            raise RuntimeError(f"{doc_name}: manifest byte count mismatch")
        if recorded.get("pages") != expected["pages"]:
            raise RuntimeError(f"{doc_name}: manifest page count mismatch")
        print(f"OK {doc_name} {len(data)} bytes {expected['pages']} pages {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the local corpus against the manifest without network access.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing PDFs even when they differ from the manifest.",
    )
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        fetch(force=args.force)


if __name__ == "__main__":
    main()
