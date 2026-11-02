"""Enforce the Dependency Rule over `src/`.

Fails when an entity or use-case module imports anything outward, or when the module graph has a
cycle. Adapter -> framework edges are reported as warnings: both sit in the outer ring, so the rule
permits them, but they are listed so the testability cost stays visible.

Usage:
    python scripts/check_layers.py           # fail on inward violations and cycles
    python scripts/check_layers.py --graph   # also print the module graph
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

LAYERS = {
    "src.domain": "entity",
    "src.retrieval.financial_numbers": "entity",
    "src.retrieval.table_ast": "entity",
    "src.inference.schemas": "entity",
    "src.usecases": "usecase",
    "src.cache_and_guardrails.guardrails": "usecase",
    "src.cache_and_guardrails.numeric_verifier": "usecase",
    "src.adapters": "adapter",
    "src.retrieval.hybrid_retriever": "adapter",
    "src.retrieval.reranker": "adapter",
    "src.retrieval.embedding_cache": "adapter",
    "src.cache_and_guardrails.semantic_cache": "adapter",
    "src.inference.llm_client": "adapter",
    "src.ingestion": "adapter",
    "src.api": "framework",
    "src.config": "framework",
}
RANK = {"entity": 0, "usecase": 1, "adapter": 2, "framework": 3}


def module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def layer_of(module: str) -> str:
    best = ""
    chosen = "framework"
    for prefix, layer in LAYERS.items():
        if (module == prefix or module.startswith(prefix + ".")) and len(prefix) > len(best):
            best, chosen = prefix, layer
    return chosen


def imported_modules(tree: ast.AST, current: str) -> set[str]:
    found: set[str] = set()
    package = current.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(package[: len(package) - node.level])
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            if target.startswith("src"):
                found.add(target)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.startswith("src"))
    return found


def scan() -> tuple[dict[str, list[str]], dict[str, str]]:
    modules: dict[str, list[str]] = {}
    layers: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        name = module_name(path)
        if "." not in name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules[name] = sorted(imported_modules(tree, name))
        layers[name] = layer_of(name)
    return modules, layers


def cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    found: list[list[str]] = []
    colour: dict[str, str] = {}

    def visit(node: str, stack: list[str]) -> None:
        colour[node] = "grey"
        stack.append(node)
        for dep in graph.get(node, []):
            if dep not in graph:
                continue
            if colour.get(dep) == "grey":
                found.append(stack[stack.index(dep) :] + [dep])
            elif dep not in colour:
                visit(dep, stack)
        stack.pop()
        colour[node] = "black"

    for name in graph:
        if name not in colour:
            visit(name, [])
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", action="store_true", help="print the module graph")
    args = parser.parse_args()

    modules, layers = scan()
    inward: list[tuple[str, str, str, str]] = []
    warned: list[tuple[str, str, str, str]] = []
    for name, deps in modules.items():
        for dep in deps:
            if dep not in layers:
                continue
            if RANK[layers[dep]] > RANK[layers[name]]:
                entry = (name, layers[name], dep, layers[dep])
                if layers[name] in {"entity", "usecase"}:
                    inward.append(entry)
                else:
                    warned.append(entry)

    found_cycles = cycles(modules)

    if args.graph:
        for name in sorted(modules):
            if modules[name]:
                print(f"{name} [{layers[name]}] -> {', '.join(modules[name])}")

    print(f"modules={len(modules)} inward_violations={len(inward)} cycles={len(found_cycles)}")
    for name, from_layer, dep, dep_layer in inward:
        print(f"VIOLATION {name} ({from_layer}) -> {dep} ({dep_layer})")
    if warned:
        pairs = sorted({(layers[name], layers[dep]) for name, _, dep, _ in warned})
        summary = ", ".join(f"{outer}->{inner}" for outer, inner in pairs)
        print(f"outer-ring couplings allowed (reported for visibility): {summary}")
    for cycle in found_cycles:
        print("CYCLE " + " -> ".join(cycle))

    return 1 if inward or found_cycles else 0


if __name__ == "__main__":
    raise SystemExit(main())
