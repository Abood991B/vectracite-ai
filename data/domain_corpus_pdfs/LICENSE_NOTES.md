# Multi-domain benchmark corpus

These PDFs are used only by the offline domain benchmark. They are not indexed by
the running application.

| Document | Domain | Source | License |
| --- | --- | --- | --- |
| `CDC_MMWR_72_42.pdf` | Public health | [CDC MMWR 2023;72(42)](https://www.cdc.gov/mmwr/volumes/72/wr/pdfs/mm7242a1-H.pdf) | U.S. government work; public domain |
| `NIST_AI_RMF_1_0.pdf` | AI governance | [NIST AI 100-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) | U.S. government work; public domain |
| `SCOTUS_303_Creative_v_Elenis.pdf` | Legal | [Supreme Court opinion](https://www.supremecourt.gov/opinions/22pdf/600us1r58_7khn.pdf) | U.S. government work; public domain |
| `ARXIV_LLaMA_Open_Efficient_Foundation_LMs.pdf` | Academic ML | [arXiv:2302.13971](https://arxiv.org/pdf/2302.13971) | CC BY 4.0; attribution: Touvron et al. |

Run `python scripts/fetch_domain_corpus.py --verify` to verify sizes, page counts,
and SHA256 digests against `manifest.json`.
