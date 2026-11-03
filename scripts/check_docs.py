"""Fail when a Markdown link in the published docs cannot work for someone who cloned the repo.

Two failure modes, both invisible on the author's machine:

* the target does not exist on disk;
* the target exists but is gitignored, so a reader who clones the repository gets a 404.

The check is deliberately narrow: relative link targets only, no anchor fragments and no
external URLs, so it stays deterministic and offline.

    python scripts/check_docs.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "tel:")
ROOT = Path(__file__).resolve().parent.parent


def markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def link_targets(text: str) -> list[str]:
    targets = []
    for raw in LINK_RE.findall(text):
        target = raw.split("#", 1)[0]
        if not target or target.startswith(SKIP_PREFIXES):
            continue
        targets.append(target)
    return targets


def is_gitignored(path: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(path)],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    problems: list[str] = []
    checked = 0
    for path in markdown_files():
        if not path.is_file():
            continue
        for target in link_targets(path.read_text(encoding="utf-8")):
            checked += 1
            resolved = (path.parent / target).resolve()
            source = path.relative_to(ROOT)
            if not resolved.exists():
                problems.append(f"{source} links to a missing file: {target}")
            elif resolved.is_file() and is_gitignored(resolved):
                problems.append(f"{source} links to an unpublished file: {target}")

    for entry in problems:
        print(entry)
    print(f"checked {checked} links in README.md and docs/, problems {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
