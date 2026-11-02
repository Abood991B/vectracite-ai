"""Filesystem adapter for uploaded and checked-in PDFs."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List

DEFAULTS_CLEARED_MARKER = ".defaults-cleared"


class PdfFileStore:
    """Stores user uploads and resolves them against the checked-in corpus."""

    def __init__(self, uploads_dir: Path, corpus_dir: Path) -> None:
        self.uploads_dir = uploads_dir
        self.corpus_dir = corpus_dir

    def write(self, doc_name: str, pdf_bytes: bytes) -> Path:
        """Atomically persist a user PDF separately from the checked-in corpus."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        target = self.uploads_dir / f"{doc_name}.pdf"
        temp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_bytes(pdf_bytes)
            os.replace(temp_path, target)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return target

    def read(self, doc_name: str) -> bytes | None:
        path = self.uploads_dir / f"{doc_name}.pdf"
        return path.read_bytes() if path.exists() else None

    def delete(self, doc_name: str, *, is_demo: bool) -> None:
        """Remove a user PDF and any legacy non-demo copy from the corpus directory."""
        user_path = self.uploads_dir / f"{doc_name}.pdf"
        if user_path.exists():
            user_path.unlink()

        legacy_path = self.corpus_dir / f"{doc_name}.pdf"
        if not is_demo and legacy_path.exists():
            legacy_path.unlink()

    def find(self, doc_name: str) -> Path | None:
        for root in (self.uploads_dir, self.corpus_dir):
            pdf_path = root / f"{doc_name}.pdf"
            if pdf_path.exists():
                return pdf_path
        return None

    def list_upload_paths(self) -> List[Path]:
        return sorted(self.uploads_dir.glob("*.pdf"))

    def list_upload_names(self) -> List[str]:
        return [path.stem for path in self.list_upload_paths()]

    def defaults_cleared(self) -> bool:
        return (self.uploads_dir / DEFAULTS_CLEARED_MARKER).exists()

    def mark_defaults_cleared(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        (self.uploads_dir / DEFAULTS_CLEARED_MARKER).touch()

    def clear_defaults_marker(self) -> None:
        (self.uploads_dir / DEFAULTS_CLEARED_MARKER).unlink(missing_ok=True)
