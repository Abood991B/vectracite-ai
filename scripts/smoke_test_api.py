"""Container/API smoke test for the private VectraCite endpoints."""

import argparse
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--question", default="Did AMD report customer concentration in FY22?"
    )
    args = parser.parse_args()

    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    base = args.base_url.rstrip("/")
    failures = []

    with httpx.Client(timeout=120.0) as client:
        health = client.get(f"{base}/health")
        if health.status_code != 200 or health.json().get("status") != "healthy":
            failures.append(f"/health -> {health.status_code} {health.text[:200]}")

        ready = client.get(f"{base}/ready", headers=headers)
        if ready.status_code != 200:
            failures.append(f"/ready -> {ready.status_code} {ready.text[:300]}")

        logo = client.get(f"{base}/assets/logo.svg")
        if logo.status_code != 200 or "svg" not in logo.headers.get("content-type", ""):
            failures.append(f"/assets/logo.svg -> {logo.status_code}")

        documents = client.get(f"{base}/documents", headers=headers)
        if documents.status_code != 200:
            failures.append(f"/documents -> {documents.status_code} {documents.text[:200]}")
        elif documents.json().get("total_documents", 0) < 1:
            failures.append("/documents returned no indexed demo corpus")

        scenarios = client.get(f"{base}/scenarios", headers=headers)
        if scenarios.status_code != 200:
            failures.append(f"/scenarios -> {scenarios.status_code}")
        elif scenarios.json().get("total_scenarios", 0) < 1:
            failures.append("/scenarios returned no benchmark scenarios")

        query = client.post(
            f"{base}/query",
            headers=headers,
            json={"question": args.question, "top_k": 3, "use_cache": False},
        )
        if query.status_code != 200:
            failures.append(f"/query -> {query.status_code} {query.text[:300]}")
        elif not query.json().get("citations"):
            failures.append("/query returned no citations")

    if failures:
        print("Smoke test failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Smoke test passed: health, readiness, assets, corpus, scenarios, and query.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
