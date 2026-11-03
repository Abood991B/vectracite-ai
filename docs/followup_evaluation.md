# Document workflow follow-up evaluation

These results are diagnostic observations from November 2, 2026, not claims
about production accuracy or throughput. The machine was running other tests.
It used a Ryzen 7 4800H, 32 GB RAM, Windows, and CPU-only PyTorch. The NVIDIA
GTX 1660 Ti has 6 GB VRAM but was not used by these runs.

## Retrieval

The checked-in domain benchmark contains 12 non-financial cases across four
complete PDFs. One run used the test-only hash embedder and lightweight reranker.
Hybrid retrieval with the lightweight reranker returned a labeled evidence page
in its first three results for 6 of 12 cases. A second run with BGE-small and the
configured BGE reranker returned a labeled page in the first three for 8 of 12.
The production-model run preceded the final change to chunk identifiers. The
lightweight run was repeated after that change. This measures page retrieval, not answer
correctness. Evidence lists may omit other valid pages, and the small sample does
not support a general accuracy estimate or a model ranking.

Reproduce the subset without overwriting the checked-in report:

```powershell
python scripts/evaluate_domain_subset.py --lightweight --bootstrap-samples 100 --output "$env:TEMP\vectracite-domain-report.json"
```

Omit `--lightweight` to run the configured production embedding and reranking
models. That run takes more resources.

## Answer check and ingestion

One complete six-page CDC PDF was uploaded to an isolated in-memory index with
the production BGE-small embedding model, BGE reranker, OCR disabled, and the
deterministic synthesizer. The parser produced 38 chunks, with text reported
on all six pages. One run took 4.28 seconds for parsing and chunking, 6.81 seconds
for embedding including model initialization, and 12.17 seconds for the upload.
Two of the three CDC answers contained the labeled expected figure. Their table
output was verbose and difficult to interpret. The third answer did not contain
the expected figure and mixed the report's columns. These are three cases, not
an accuracy benchmark. Other workloads ran concurrently.

## OCR comparison

Each candidate processed the same 150 DPI raster of physical page 2 of the CDC
PDF once on CPU. The reference was PyMuPDF's native text, not a human transcript.
The comparison counted matching case-insensitive alphanumeric word tokens,
ignoring order. It does not assess reading order, tables, or semantic accuracy.

| Engine | Initialization | Inference | Reference tokens matched | Output tokens matched |
| --- | ---: | ---: | ---: | ---: |
| RapidOCR ONNX Runtime 1.4.4 | 3.12 s | 19.39 s | 705 / 776 | 705 / 731 |
| PaddleOCR 3.7.0, PP-OCRv5 mobile | 47.02 s | 68.56 s | 763 / 776 | 763 / 770 |

The runs used one temporary Python environment, cached model files,
`OMP_NUM_THREADS=4`, and shared system load. Actual thread use was not measured.
The temporary environment
reported dependency-version warnings. These single runs do not establish a
speed or accuracy ranking across documents. A representative labeled scan set,
repeated alternating runs, and memory measurements are needed before changing
the OCR backend.

The project's `PaddleOCRBackend` and PDF parser also processed a one-page PDF
created from the same raster. The run took 102.84 seconds including model
initialization and produced four chunks marked as OCR-derived. It ran under
shared load in the temporary environment. With the default 120-second ingestion
timeout, this leaves little margin for scanned documents on this machine.
