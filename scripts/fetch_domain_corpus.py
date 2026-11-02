"""Download and verify the checked-in multi-domain benchmark corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import fitz
import httpx

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings

DOMAIN_CORPUS_DIR = settings.DATA_DIR / "domain_corpus_pdfs"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

DOMAIN_DOCUMENTS: Dict[str, Dict[str, Any]] = {
    "CDC_MMWR_72_42": {
        "url": "https://www.cdc.gov/mmwr/volumes/72/wr/pdfs/mm7242a1-H.pdf",
        "bytes": 197357,
        "pages": 6,
        "sha256": "970a773273a2ad9a1a1decfa1187aef07ec08c14fab0509629ded86f4231a40c",
        "domain": "public_health",
        "license": "public_domain_us_government",
        "attribution": "Centers for Disease Control and Prevention, MMWR 2023;72(42).",
    },
    "NIST_AI_RMF_1_0": {
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "bytes": 1946127,
        "pages": 48,
        "sha256": "7576edb531d9848825814ee88e28b1795d3a84b435b4b797d3670eafdc4a89f1",
        "domain": "ai_governance",
        "license": "public_domain_us_government",
        "attribution": "National Institute of Standards and Technology, NIST AI 100-1 (2023).",
    },
    "SCOTUS_303_Creative_v_Elenis": {
        "url": "https://www.supremecourt.gov/opinions/22pdf/600us1r58_7khn.pdf",
        "bytes": 450426,
        "pages": 73,
        "sha256": "87fd2aa79e9a73dc85458ffb4e41ac91a0c6d2e40bf2d2774a812a9225317587",
        "domain": "legal",
        "license": "public_domain_us_government",
        "attribution": "Supreme Court of the United States, 303 Creative LLC v. Elenis, 600 U.S. 570 (2023).",
    },
    "ARXIV_LLaMA_Open_Efficient_Foundation_LMs": {
        "url": "https://arxiv.org/pdf/2302.13971",
        "bytes": 726566,
        "pages": 27,
        "sha256": "2e663675ae36ad12adb2f5a05281bac2747ecf8d23d92bedd9f937a89fee7136",
        "domain": "academic_ml",
        "license": "CC-BY-4.0",
        "attribution": "Touvron et al., LLaMA: Open and Efficient Foundation Language Models, arXiv:2302.13971.",
    },
}


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
    expected = DOMAIN_DOCUMENTS[doc_name]
    if len(data) != expected["bytes"]:
        raise RuntimeError(
            f"{doc_name}: size mismatch (expected {expected['bytes']}, got {len(data)})"
        )
    pages = _pdf_pages(data)
    if pages != expected["pages"]:
        raise RuntimeError(
            f"{doc_name}: page-count mismatch (expected {expected['pages']}, got {pages})"
        )
    if _sha256(data) != expected["sha256"]:
        raise RuntimeError(f"{doc_name}: SHA256 mismatch")


def _manifest_path() -> Path:
    return DOMAIN_CORPUS_DIR / MANIFEST_NAME


def load_manifest() -> Dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "documents": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _download(url: str, client: httpx.Client) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def _manifest_document(data: bytes, expected: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "filename": f"{expected['doc_name']}.pdf",
        "bytes": len(data),
        "pages": _pdf_pages(data),
        "sha256": _sha256(data),
        "source_url": expected["url"],
        "domain": expected["domain"],
        "license": expected["license"],
        "attribution": expected["attribution"],
    }


def fetch(force: bool) -> None:
    DOMAIN_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("documents", {})

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for doc_name, expected in DOMAIN_DOCUMENTS.items():
            target = DOMAIN_CORPUS_DIR / f"{doc_name}.pdf"
            if target.exists() and not force:
                data = target.read_bytes()
                _validate_pdf(data, doc_name)
                print(f"verified existing {target.name} ({len(data)} bytes)")
            else:
                data = _download(expected["url"], client)
                _validate_pdf(data, doc_name)
                temp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                try:
                    temp_path.write_bytes(data)
                    os.replace(temp_path, target)
                finally:
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                print(f"downloaded {target.name} ({len(data)} bytes)")
                manifest["documents"][doc_name] = _manifest_document(
                    data, {"doc_name": doc_name, **expected}
                )
                continue
            manifest["documents"][doc_name] = _manifest_document(
                data, {"doc_name": doc_name, **expected}
            )

    _manifest_path().write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {_manifest_path()}")


def verify() -> None:
    manifest = load_manifest()
    documents = manifest.get("documents", {})
    if set(documents) != set(DOMAIN_DOCUMENTS):
        raise RuntimeError(
            "manifest documents do not match the expected domain set: "
            f"{sorted(documents)} != {sorted(DOMAIN_DOCUMENTS)}"
        )
    for doc_name, expected in DOMAIN_DOCUMENTS.items():
        target = DOMAIN_CORPUS_DIR / f"{doc_name}.pdf"
        if not target.exists():
            raise FileNotFoundError(target)
        data = target.read_bytes()
        _validate_pdf(data, doc_name)
        recorded = documents[doc_name]
        if recorded.get("sha256") != expected["sha256"]:
            raise RuntimeError(f"{doc_name}: manifest SHA256 mismatch")
        if recorded.get("bytes") != expected["bytes"]:
            raise RuntimeError(f"{doc_name}: manifest byte count mismatch")
        if recorded.get("pages") != expected["pages"]:
            raise RuntimeError(f"{doc_name}: manifest page count mismatch")
        print(
            f"OK {doc_name} {len(data)} bytes {expected['pages']} pages "
            f"{expected['sha256']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the local domain corpus against the manifest without network access.",
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
