> Historical reports: these measurements used the removed synthetic corpus.
> The active course now uses only the original Magenta PDF and ten development
> scenarios. These reports are not directly comparable to new runs.

# Development results — not the frozen baseline

The [architecture smoke comparison](2026-10-21_dev_architecture_smoke.md) contains
seven identical development cases per architecture: 21 results, Greek and English.
The [paired JSON](2026-10-21_dev_architecture_smoke.json) preserves passages,
answers, atomic labels, stage usage and source run IDs. Application outputs were
not replayed when the Classic judge protocol was calibrated.

The [retrieval-only comparison](2026-10-21_dev_retrieval_comparison.md) covers all
44 development cases for eligible plugins. It completed 38 direct-query cases per
Classic/Graph and intentionally skipped history rewriting and Agentic research.
Its coverage flag is false because those requested rows were skipped, not because
the eligible retrieval requests failed.

Measured canonical-source recall:

| Development group | Classic | Local Graph | Cases per architecture |
|---|---:|---:|---:|
| Factual, Greek | 0.889 | 0.778 | 9 |
| Multi-hop, Greek | 0.429 | 0.333 | 7 |
| Multi-hop, English | 0.500 | 0.333 | 2 |
| Global, Greek | 0.450 | 0.200 | 5 |
| Global, English | 0.500 | 0.250 | 1 |

The local graph regressed on this draft benchmark. Do not present it as an
improvement. Inspecting selected passages showed expansion into generic/older
program clauses; this is a plausible contributor, not a controlled causal result.
The later planned global-search capability is not implemented yet.

In the generation smoke cases, Agentic retrieved all three labeled passages for
each of the two multi-hop examples and answered; Classic and Graph abstained with
incomplete evidence. Agentic used more research calls and time. Two examples do
not establish a general advantage. All three answered `followup-05` instead of
asking the expected clarification. Valid source IDs are not a completeness score.

The first judge protocol incorrectly treated the fixed refusal as factual content
and copied reference facts into answer coverage. It was superseded. Current
refusals have deterministic empty-claim scoring; non-missing fact labels require
answer quotes; malformed labels get one bounded retry. Original component and
calibration reports are retained locally under
`outputs/checkpoints/smoke-runs/` and `outputs/checkpoints/`. Earlier Cloud score
batches are marked with `course.judge_protocol_superseded`; use the source run IDs
in the current JSON when inspecting scores. API readback matched all computed
metrics for the 21 current smoke trace groups.

Indexing recorded 1,649,136 input and 210,301 output tokens separately from query
usage. Four diagnostic attempt records lack usage; these are not necessarily four
provider calls. No monetary total is inferred. Some audited graph statements and
quotations omit conditions; original passages remain the answer evidence.

[manual_review.csv](manual_review.csv) has blank human fields for correctness,
language and conversation completion. Fill 0 / 0.5 / 1, reviewer and supporting
notes, then compare disagreements against the raw judge labels. No human scores
have been invented.

The held-out set remains pending instructor review. Its language allocation also
needs consideration before freezing: dev has 41 Greek / 3 English cases, while
held-out has 12 Greek / 17 English. Results must stay separate by language. Do not
interpret these draft smoke checks as an independent held-out baseline.
