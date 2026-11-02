# Domain-agnostic benchmark

The FinanceBench slice is a finance regression suite. The domain benchmark is a separate,
answer-blind evaluation of the same retrieval stack on real documents from several domains.

## Page-level PDF suite

- **Cases:** `data/domain_benchmark_cases.json`
- **Corpora:** `data/corpus_pdfs/` and `data/domain_corpus_pdfs/`
- **Documents:** 9 real PDFs
- **Cases:** 17
- **Domains:** finance, public health, AI governance, legal, academic ML
- **Evidence:** physical PDF page numbers, with `gold_answer` and `gold_keywords`

Run:

```bash
python scripts/fetch_domain_corpus.py --verify
python -m evaluation.domain_benchmark --output data/domain_benchmark_report.json
```

The report contains:

- P@1, R@3, MRR, and nDCG@3 for dense-only, hybrid RRF, and hybrid plus reranker.
- Bootstrap 95% confidence intervals.
- Per-domain metric breakdowns.
- Paired bootstrap deltas, win/loss counts, and approximate p-values.
- Per-query ranks and retrieved evidence metadata.
- A signature covering the corpus SHA256 values, parser settings, embedding model, reranker
  model, cases, and chunks.

## Passage-level RAGBench suite

RAGBench provides public question/passage data across domains. The adapter streams a stratified
subset and converts the relevant passages into expected chunk IDs. It is retrieval-only because
RAGBench does not provide PDF page geometry.

```bash
python -m evaluation.domain_benchmark --source ragbench \
  --max-cases-per-config 12 --output data/ragbench_report.json
```

The checked-in report uses `covidqa`, `cuad`, `emanual`, `finqa`, and `expertqa`, with 12 cases per
dataset and 2,000 bootstrap samples.

## Interpretation rules

- A page-level hit requires both the correct document and an expected physical page.
- A passage-level hit requires one of the expected RAGBench chunk IDs.
- Do not compare the page-level and passage-level aggregate scores directly.
- Treat a change as a candidate only when the paired confidence interval and per-domain breakdown
  support it. A single aggregate improvement is not enough.
- Legal retrieval is an observed weakness in the first baseline. It remains a target for Phase 2.

## Sources and licenses

See `data/domain_corpus_pdfs/LICENSE_NOTES.md`. The CDC, NIST, and Supreme Court PDFs are U.S.
government works. The LLaMA paper is CC BY 4.0 with attribution in the manifest.

## Extension points

- Add BEIR retrieval subsets when the Phase 2 embedding/fusion experiments begin.
- Add more domains and multi-page questions before making broad accuracy claims.
- Add answer-level faithfulness and refusal metrics after the retrieval baseline is stable.
- Keep the FinanceBench suite as a regression suite; do not replace it with the domain suite.
