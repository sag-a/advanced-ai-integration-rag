# Evaluate the course

Use the same cases, corpus, final top-k (6), model, decoding temperature (0), prompts and judge for each registered architecture. The runner imports plugins through the registry; it does not pass expected answers, facts, evidence labels or question types to retrieval or generation. Each case passes its own message list directly to the graph. No conversation database or lock files are used.

The active dataset contains **10 reused Magenta Moments development cases** in
Greek and English, based solely on the original five-page PDF: factual questions,
follow-ups, a cross-page question, an ambiguous reference, account-data abstention
and a missing-policy question. These are existing classroom examples, not unseen
held-out data. No architecture improvement is assumed.

## Run

From the repository root, using the existing `magenta_course_pdf_v1` collection:

```bash
# Local embedding retrieval only; no Anthropic calls.
uv run --all-extras python -m evals.run --architecture classic_rag --retrieval-only

# Actual graph, answers and independent judge. Anthropic usage applies.
uv run --all-extras python -m evals.run --architecture classic_rag

# Compare installed plugins sequentially, using the existing original-PDF graph.
uv run --all-extras python -m evals.run --architecture all
```

`--split dev` is the default. `--case s01-enrollment --case s07-join-and-expiry` selects a smoke subset, explicitly recorded in the report; it is not a frozen baseline. `--langfuse` adds graph traces and computed scores to the configured project. Export errors leave local results intact and are reported explicitly. Questions, bounded histories, relevant passages and accepted answers are sent to Cloud when enabled. Judge usage stays separate in the local stage records.

The report is written as JSON and a Markdown comparison under `evals/reports/`. Existing reports are never overwritten. Use `--output outputs/my-run.json` for disposable trials. A failed graph preflight, exhausted research budget, provider error or judge error remains a failed row; the process returns nonzero. Successful retrieval is still scored if later generation fails. Follow-ups and model-based research are **skipped** in retrieval-only mode because they need Anthropic; they are not silently evaluated without history. The default 30-passage graph index matches the original corpus.

## Review and freeze

No held-out cases are currently configured. `--split heldout` stops before any
service or model access. The former 29-case synthetic review is withdrawn.
[REVIEW.md](datasets/REVIEW.md) reflects the current dataset.

If the instructor later supplies independent held-out labels, keep the existing
review gate: refresh after edits, review the source facts and physical-page
references, and explicitly approve the matching dataset identity. Refreshing
never grants approval automatically:

```bash
uv run --all-extras python -m evals.dataset refresh
```

Historical reports in `reports/` used the removed expanded corpus. Their IDs and
measured scores remain historical evidence; they are not baselines for these ten
cases. A new report must record the current corpus/dataset identities.

## What the scores mean

| Score | Computation | Limitation |
|---|---|---|
| Recall@k | Expected evidence labels covered / all expected labels | Canonical source coverage, not exhaustive semantic relevance |
| Precision@k | Selected passages matching a label / unique selected passages | Unlabeled but useful passages may count as irrelevant |
| MRR | Reciprocal rank of first labeled relevant passage | Does not measure full multi-hop coverage |
| Required-fact recall | Fully met required facts / required facts | Missing material qualifications count as missing |
| Factual precision | Correct atomic answer claims / actual claims | Judge reference is expected source passages plus retrieved passages |
| Factual correctness | Harmonic mean of precision and required-fact recall | A model judgment, not a verified truth score |
| Faithfulness | Claims fully supported by retrieved evidence / claims | Correct external knowledge is not grounding |
| Citation support | Claims fully supported by their cited passages / claims | Valid IDs alone are insufficient |
| Citation membership | Citations exactly matching selected original passages / citations | Deterministic membership check only |
| Evidence citation coverage | Labeled evidence covered by valid citations / labels | Distinct from semantic citation support |
| Status match | Expected answered / insufficient / clarification status matches | Human language and conversation quality still need review |

The judge returns atomic labels and exact answer quotes; Python calculates numbers. Invalid quotes, missing fact labels and invented citation support trigger at most one validation retry, then reject the judgment. Both attempts and their available usage remain in the report. A missing fact is not invented as an answer claim. The application's fixed refusal is scored deterministically without a judge call: all required answer facts are missing, and there are no factual claims. For an answerable case with no claims, factual precision and F1 use a zero convention, not perfect empty precision. Honest empty refusals/clarifications have **null** grounding and citation scores; an expected refusal with no required facts also has null factual scores. Nulls are excluded from means with a separate denominator shown for every metric. Execution failures are not hidden inside quality averages.

Reports retain search queries, actual selected counts, original passages, citations, research details, stage latency, logical completed LLM call counts and returned token usage for rewrite/research/generation/judge. Failed-call usage and SDK retries may be unavailable. The first query in a fresh process includes embedding-model loading; caches are not reset per case. These timings describe the run and are not a controlled warm-latency benchmark. Indexing usage is separate from question-time usage; costs are left unreported without a verified price table. Langfuse scores have stable IDs within a run and record the judge configuration. Queue flushing is recorded separately from server readback; inspect Cloud or query the API before claiming uploads are confirmed.

For manual scoring, sample successes, refusals, low scores and a few apparent perfect answers. Record correctness, requested language and conversation completion as 0/0.5/1 (or N/A) with a supporting quotation. Compare these with the atomic judge labels; review disagreements before changing a prompt. A small synthetic set supports classroom diagnosis, not claims about general production quality.

History isolation comes from a separate message list for each case. The runner uses the same application graph and citation validation as the UI/CLI. If generation fails, intermediate graph state retains completed retrieval for separate evaluation; no draft is accepted as an answer.
