"""Workbench delivery asset.

The HTML lives in ``static/workbench.html`` so it can be edited with HTML tooling, diffed without
Python string escaping, and exempted from the application's line-length and lint rules. This module
only loads it once and keeps the historical ``WORKBENCH_HTML`` import working.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
WORKBENCH_PATH = STATIC_DIR / "workbench.html"


@lru_cache(maxsize=1)
def load_workbench_html() -> str:
    """Read the workbench document from disk, cached for the process lifetime."""
    return WORKBENCH_PATH.read_text(encoding="utf-8")


WORKBENCH_HTML = load_workbench_html()

__all__ = ["STATIC_DIR", "WORKBENCH_PATH", "WORKBENCH_HTML", "load_workbench_html"]
