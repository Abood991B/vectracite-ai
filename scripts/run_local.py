"""Start VectraCite-AI for one user on one machine.

A bare ``uvicorn src.api.main:app`` needs a Qdrant server or an explicit in-memory flag, and both
are easy to miss on a first run. This entry point picks the zero-dependency local mode by default
and still honours any setting the caller exported, so ``QDRANT_IN_MEMORY=false`` with a real
Qdrant server keeps working.

    python scripts/run_local.py
    python scripts/run_local.py --port 8090
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    os.environ.setdefault("QDRANT_IN_MEMORY", "true")
    sys.path.insert(0, str(ROOT))

    import uvicorn

    vector_mode = "in-memory Qdrant" if os.environ["QDRANT_IN_MEMORY"] == "true" else "Qdrant server"
    print(f"VectraCite-AI local mode ({vector_mode})")
    print(f"Workbench:   http://{args.host}:{args.port}/")
    print(f"OpenAPI:     http://{args.host}:{args.port}/docs")
    print("First start loads the embedding and reranker models; later starts are faster.")
    print("Press Ctrl+C to stop.")

    uvicorn.run("src.api.main:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
