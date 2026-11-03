# Query interpreter: decision models vs. a tiny local rewriter

Status: evaluation, not shipped. Written 2026-11-03 after testing the live workbench.

## The question

Can a very small local model sit between the user and the retriever — seeing only the question, never
the documents — and make casual or vague questions retrieve the right evidence?

## What Jev actually is

Jev (TypeSafe AI, September 2026) is a **decision model**: it takes text and returns a choice, a
boolean, a score, or a box, with calibrated probabilities. It does not generate text, it is a hosted
closed API, and it is very fast (~0.15 s median) and cheap. Open Jev-class alternatives exist
(Kev 0.8B/4B/9B, Laya, decider-2B, and others), and several take the same request shape.

**It cannot rewrite a query.** Turning "yo what did amd make in fy22" into "AMD net revenue FY2022" is
a generation task. Jev can only answer a fixed question such as *"is this question a table lookup?"* or
*"should this be escalated?"*. It is the right shape for routing, not for interpretation.

## The experiment

A 0.5B instruct model (Qwen2.5-0.5B-Instruct, float32, CPU, transformers) rewrote eight casual
questions; the raw and rewritten questions were then sent to the live API with the document scope
fixed.

| Raw question | Rewrite | Latency | Cited before → after |
|---|---|---:|---|
| yo what did amd make in fy22 | AMD net revenue FY2022 | 2600 ms | 1 → 1 |
| how much moolah did amd pull in last year | moolah earnings last year | 1832 ms | 1 → 1 |
| amd reveune 2022 | amd net revenue 2022 | 2066 ms | **0 → 2** |
| whats the cash situation | cash flow | 1056 ms | 2 → 2 |
| did they beat last years numbers | net profit vs last year's earnings | 1702 ms | 1 → 2 |
| how many people work there | number of employees | 1307 ms | 1 → 1 |
| whats the net revenue for 3m in 2018 | net revenue 3m in 2018 | 2520 ms | 3 → 1 |
| 3m capex 2018 | capital expenditures 2018 | 1719 ms | 1 → 1 |

One failure fixed, none broken, two answers improved. The typo case is the clearest win: the rule-based
normalizer cannot recover `reveune`, the model can.

**The blocker is latency.** 1.0–2.6 s per rewrite on CPU with float32 transformers is too slow to run
on every question. Quantized GGUF + llama.cpp, a 135M–360M model, or a hosted decision endpoint would
each change that number, but none is in the current stack.

## Recommendation

Do not put a model on the hot path yet. Adopt a **cascade**:

1. **Deterministic normalizer (shipped).** Synonym mapping, fiscal-year normalisation, and query
   expansion. Zero cost, always on, domain-generic enough for financial documents. It already fixed
   the common cases.
2. **Optional local rewriter, only on failure.** When a scoped query returns no citations, or the
   interpreter's confidence is low, call a tiny quantized model once, cache the rewrite by question
   hash, and retry retrieval. This bounds the cost to the failing minority.
3. **Optional decision model (Jev-class) for routing.** Use it only for fixed classifications:
   table-vs-narrative, answerable-vs-out-of-scope, local-vs-remote, accept-vs-escalate. It is cheap and
   fast, but it never sees document text either.

The port belongs in `src/usecases/ports.py` as `QueryInterpreterPort` with two implementations
(rule-based, local-model) and a composition-root choice. The use case keeps working when no model is
installed.

### Privacy

Every option above sees the question only. Document text never reaches the interpreter, so the
confidentiality boundary is unchanged. A hosted endpoint would receive the question itself — a smaller
but real disclosure that must be an explicit deployment choice, not a default.

### Why this matters for arbitrary uploads

The shipped synonym map is finance-flavoured. A user uploading a legal contract, a clinical protocol,
or an academic paper will not have "revenue" in their question. That is the strongest argument for the
model layer: a rewriter generalises across domains where a hand-written map cannot. The cascade keeps
that generality available without paying for it on every fast, well-formed question.

## Adoption trigger

Adopt the model layer when, on a labelled set of at least 50 real user questions across at least three
document domains, the rule-based path leaves more than 15% of questions with no citation **and** the
quantized rewrite latency is under 300 ms p95 on the target hardware. Until then the cascade stays a
design, not a dependency.
