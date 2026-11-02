"""Evaluate non-financial PDF cases with the existing domain benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.domain_benchmark import (  # noqa: E402
    DomainBenchmarkSpec,
    load_domain_benchmark_spec,
    run_domain_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Destination for the JSON report")
    parser.add_argument("--lightweight", action="store_true", help="Use test-only models")
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    args = parser.parse_args()
    original = load_domain_benchmark_spec(ROOT / "data/domain_benchmark_cases.json")
    cases = tuple(case for case in original.cases if case.domain != "finance")
    spec = DomainBenchmarkSpec(
        cases=cases,
        corpus_dirs=(ROOT / "data/domain_corpus_pdfs",),
        description="Non-financial subset of the checked-in page-level PDF benchmark.",
    )
    report = run_domain_benchmark(
        spec=spec,
        lightweight=args.lightweight,
        bootstrap_samples=args.bootstrap_samples,
        output_path=args.output,
    )
    print(json.dumps({"summary": report["summary"], "signature": report["signature"]}, indent=2))


if __name__ == "__main__":
    main()
