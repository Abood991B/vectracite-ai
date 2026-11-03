# Retrieval experiments

Phase 2 used the domain benchmark to test changes before adopting them. The checked-in matrix is
`data/retrieval_experiments_baseline_gates.json`; the embedding screen is
`data/retrieval_experiments_ragbench_embeddings.json`.

## Method

- Page-level suite: 17 cases, 9 real PDFs.
- Passage-level suite: 60 RAGBench cases across five datasets.
- Every variant uses the same cases and 2,000 bootstrap samples.
- Pairwise comparisons resample cases, not documents.
- The page-level suite is the primary citation target; RAGBench is a generalization guardrail.

## Results

These are the Phase 2 experiment results on the index at that time; the current domain baseline is in
`data/domain_benchmark_report.json`.

| Change | PDF R@3 | RAGBench R@3 | Decision |
|---|---:|---:|---|
| Adaptive reranker baseline | 0.5294 | 0.8000 | Kept as default |
| Forced top-2 cross-encoder | **0.7647** | 0.8000 | Rejected: FinanceBench P@1 0.25 -> 0.125 |
| Forced all candidates | 0.7059 | 0.7833 | Rejected: slower and weaker |
| Top-1 cross-encoder | 0.7647 | 0.8000 | Rejected: lower P@1 |
| Page fusion with adaptive reranker | 0.6471 | n/a | Not adopted |
| Page fusion with top-2 reranker | 0.6471 | n/a | Rejected |
| BM25 weight 0.30 | 0.5294 | 0.8500 | Rejected: dataset-dependent |
| BM25 weight 0.50 | 0.5294 | 0.8000 | Rejected |
| BM25 weight 0.70 | 0.4706 | 0.8167 | Rejected |
| Candidate depth 2, 8, 20 | 0.5294 | 0.8000 | No effect |
| BGE-base embedding | not run on PDF | 0.7833 | Rejected on screen |
| E5-base embedding | not run on PDF | 0.7667 | Rejected on screen |
| GTE-base embedding | not run on PDF | 0.8333 | Rejected: P@1/MRR tradeoff |

## Default configuration

```text
RERANKER_GATE=adaptive
RERANKER_CONFIDENCE_THRESHOLD=0.5
HYBRID_DENSE_WEIGHT=0.95
HYBRID_BM25_WEIGHT=0.05
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
```

The forced top-2 domain result looked strong, but it failed the original FinanceBench regression:

```text
FinanceBench P@1: 0.2500 -> 0.1250
FinanceBench MRR: 0.3125 -> 0.2500
FinanceBench R@3: 0.3125 -> 0.3125
```

The top-2 gate is therefore rejected. The adaptive gate preserves FinanceBench and all existing
behavior. The full top-2 matrix remains in the checked-in experiment report for auditability.

## Deferred

- Table-aware chunk-size sweep: the current parser already keeps Markdown table blocks intact, but
  a 900/1200/1600-token sweep would require re-embedding the full PDF corpus for each
  variant. It is deferred until a GPU embedding path is available.
- BEIR retrieval subsets and query rewriting/HyDE were not executed in Phase 2.
- A larger reranker (`bge-reranker-v2-m3`) was not tested; CPU cost is already the limiting factor.

## Known tradeoffs

- The force-top-2 gate improves the small domain suite but regresses FinanceBench P@1/MRR. It is
  not worth shipping for the current corpus.
- AI-governance P@1 would fall from `0.6667` to `0.0` under that gate, although R@3 stays at `1.0`.
- CPU latency is the cost of extra cross-encoder passes; GPU serving is a later optimization.
- The PDF suite is small; confidence intervals are wide. Do not interpret these numbers as a
  leaderboard result.
- BM25 fusion weights are corpus-sensitive. Do not tune them on a single suite.

## Reproduction

```bash
python -m evaluation.retrieval_experiments \
  --models bge-small --suites pdf ragbench \
  --output data/retrieval_experiments_baseline_gates.json

python -m evaluation.retrieval_experiments \
  --models bge-base e5-base gte-base --suites ragbench \
  --output data/retrieval_experiments_ragbench_embeddings.json
```

The experiment cache under `data/retrieval_experiment_cache/` is derived data and is ignored by git.
