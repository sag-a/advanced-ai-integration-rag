> **Historical record, kept for instructor reference.** Measured on 19–20 October 2026
> with the retired `base_rag_architecture` package, before the migration to `rag_core` and
> `./rag`, on the same PDF that is in `data/corpus` (SHA-256 `a2e492…`). Commands and paths below no longer run. Use `./rag` and
> `python -m evals.run` instead. The observations, failures and teaching examples still apply.

# Observe and evaluate the RAG with Langfuse Cloud

The RAG still runs locally: Chroma stores the PDF, Nomic embeds questions locally,
Anthropic generates answers, and SQLite remembers conversations. Langfuse stores
traces and evaluation results in the cloud. It does not replace conversation memory.

## Enable cloud tracing

From `python/`:

```bash
uv sync --extra chroma --extra pgvector --extra langfuse
uv run --no-sync python -m base_rag_architecture langfuse-check --env-file .env.example
```

The selected environment file must contain `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, and `ANTHROPIC_API_KEY`. Shell values
win. The application reads configuration without changing files or the shell.
Prefer an ignored `.env` for ongoing use; credentials must not enter commits or
shared artifacts. These examples select your existing file explicitly.

Start Streamlit:

```bash
uv run --extra chroma --extra pgvector --extra langfuse streamlit run src/base_rag_architecture/streamlit_app.py --server.address 127.0.0.1 --server.fileWatcherType none -- --env-file .env.example --langfuse
```

Or use the terminal:

```bash
uv run --no-sync python -m base_rag_architecture chat --env-file .env.example --langfuse --thread langfuse-lesson
```

Without `--langfuse`, ordinary chat does not initialize the Langfuse integration.
Enabling it sends the question, bounded recent history, prompts, retrieved PDF
passages, and answers to your selected cloud project. The masking layer removes
known credential values and credential-shaped fields; do not use private customer
information for classroom questions. Authentication headers and environment files
are never included in graph state. This is a teaching app, not a general PII scrubber.

## Read a trace: 10 minutes

Ask `Τι είναι οι Καρδιές;`, then `Πότε λήγουν;`.

Open **View trace in Langfuse** below an answer and sign in to your cloud account.
API credentials authorize application access; they do not sign a browser into the UI.

- A **trace** follows one executed turn. Its root `rag-turn` output contains the
  validated answer, retrieval query, passages, citations, and observations.
- The three graph nodes appear below `LangGraph`. Expand `prepare_query` on the
  follow-up to inspect the standalone query and Anthropic rewrite call.
- Expand `retrieve` to inspect page, clause, text, and distance. A nearby vector
  does not establish factual relevance.
- Expand `generate_answer` to inspect the actual model prompt, response, usage,
  and latency. Drafts are provisional; the application root records the accepted
  answer after validation. Low-level generation observations may contain rejected
  output, which is useful when diagnosing a failure.
- A **session** groups turns using the conversation ID. It is a tracing group;
  SQLite remains the source of conversation state.

Saved answers retain trace links across restart. Explicitly replaying a completed
message ID returns the same answer/link without another generation. A failed
attempt can have its own error trace; retrying an unfinished turn is deliberate.
Export failures show a warning and do not invalidate an otherwise valid answer.

## Publish the dataset and configure judges: 5 minutes

```bash
uv run --no-sync python -m base_rag_architecture langfuse-sync-dataset --env-file .env.example
uv run --no-sync python -m base_rag_architecture langfuse-setup-judges --env-file .env.example
```

The sync command publishes the existing 19 cases with stable IDs. Its dataset name
includes a PDF/fixture identity hash. Repeating it does not add duplicates; changed
fixtures get another dataset name. Experiments refuse cloud fixtures that differ
from the matching local version.

The explicit judge setup command creates a dedicated `magenta-course-anthropic`
connection using the selected Anthropic credential. This stores that credential
in your Langfuse project for cloud-side judge calls. It does not overwrite other
provider connections. The three prompts and rubrics are versioned in `judges.py`
and are editable in Langfuse. Each evaluator records its model/version.

Rules match only `experiment-item-task` observations for this dataset with
`judge_mode=anthropic`. Citation coverage additionally requires an expected answerable
case. Ordinary chat does not match. Changing the owned rules and then rerunning the
setup command reapplies the course mappings and scope.

| Judge variable | Rule mapping | Purpose |
|---|---|---|
| `input` | Observation input | Question and authored prior conversation |
| `output` | Observation output | Accepted answer, retrieved/cited text, status |
| `expected_output` | Observation output `$.expected_result` | Expected status, facts, pages/clauses |

Reference facts are attached **after inference**, never passed into the RAG model.
The output-field mapping makes the reference available to observation-level rules.
The evaluator's default `expected_output` mapping remains useful for prompt
experiments; the course rules explicitly override it for graph experiments.

The AI judges assess correctness, grounding, and citation coverage separately.
Their results are estimates, not independent truth. The default application and
judge use the same explicit Haiku model; inspect their disagreements manually.
Non-answerable citation cases stay unscored. An abstention with no unsupported
claims can pass grounding while still failing correctness if an answer was expected.

## Compare top-k 3 and 6: 10 minutes

```bash
uv run --no-sync python -m base_rag_architecture langfuse-experiment --env-file .env.example --top-k 3 --run-name topk-3 --judge --output outputs/base_rag_architecture/langfuse/topk-3.json
uv run --no-sync python -m base_rag_architecture langfuse-experiment --env-file .env.example --top-k 6 --run-name topk-6 --judge --output outputs/base_rag_architecture/langfuse/topk-6.json
```

These execute the actual graph sequentially, at temperature 0, on the same cases,
model, prompts, PDF, and embedding contract. Conversations are isolated in memory;
classroom SQLite history and Chroma contents remain untouched. Run names get unique
suffixes so repeat invocations do not silently merge experiments.

Use the printed experiment links and select both runs in Langfuse's comparison UI.
The deterministic scores have these meanings:

| Score | What it checks |
|---|---|
| `evidence_hit_at_k` | At least one expected page/clause was retrieved |
| `evidence_coverage` | Fraction of all expected page/clause pairs retrieved |
| `status_matches` | Answered/abstained/clarified as expected |
| `citation_membership_valid` | Citations refer to retrieved IDs and provenance |
| `expected_evidence_citation_coverage` | Fraction of expected evidence actually cited |

Run-level `el.*` and `en.*` scores keep languages separate. Evidence metrics exclude
non-answerable cases; failed answerable cases count as zero. Missing/unreviewed
semantic scores never become passing scores. A valid citation is not proof that
it supports every claim. The citation judge examines cited text, not just IDs.

`--judge` opts the run into the cloud judges and additional Anthropic charges.
Omit it to collect deterministic metrics without AI judge calls. Judge execution
is asynchronous and its usage belongs to separate Langfuse evaluation records;
do not add it to application generation usage when comparing retrieval settings.

The dataset’s prior answers are authored fixtures, not scored model outputs.
These experiments assess each requested turn; test generated multi-turn sessions
separately, as the Streamlit smoke checks do.

Local reports include configuration identities, per-case timings/usage, and
application failures. The runner reads back deterministic cloud scores with a
bounded wait. An incomplete upload is reported explicitly. It never automatically
repeats generation to repair telemetry. Review the existing run if ingestion is delayed.

To inspect asynchronous score arrival without repeating inference:

```bash
uv run --no-sync python -m base_rag_architecture langfuse-check --env-file .env.example --report outputs/base_rag_architecture/langfuse/topk-3.json --output outputs/base_rag_architecture/langfuse/topk-3-cloud-status.json
```

This reports how many deterministic case scores are visible and how many judge
scores have arrived. Unreviewed or still-pending judges remain absent.

### Re-evaluate saved answers without regenerating them

```bash
uv run --no-sync python -m base_rag_architecture langfuse-experiment --env-file .env.example --top-k 3 --run-name topk-3-rejudge --from-report outputs/base_rag_architecture/langfuse/topk-3.json --judge --output outputs/base_rag_architecture/langfuse/topk-3-rejudge.json
```

This creates a clearly labeled captured-output experiment. It reuses original
answers, passages, metrics, latency, and usage, and retains `source_trace_id` for
lineage. It makes no retrieval or answer-generation calls. New judge calls still
cost money. The source report must match the selected dataset/model/top-k/prompts.
Do not compare captured replay execution time with live RAG latency.

## Human review: 5 minutes

Open representative Greek, English, cross-page, unsupported, and ambiguous cases.
Add `human.correctness`, `human.language`, and `human.task_completion` scores:
0 failed, 0.5 partial, 1 passed. Leave unreviewed cases empty and explain each score.

For a concrete citation exercise, test an answer claiming Hearts expire after one
year while citing only clause 3.2. That citation exists, but expiry requires 3.3.
The membership check can pass while semantic citation coverage fails. Review a
copy as a judge test rather than corrupting the canonical benchmark.

Finish by comparing code metrics, AI reasoning, and your own reading of the PDF.
The authored dataset is small and not held out. A tie or perfect score here does
not establish production quality. Report failures and collect new reviewed cases
before claiming a retrieval or prompting improvement.

## Verified classroom baseline — 20 October 2026

The live RAG ran all 19 authored cases (15 Greek, 4 English) at each top-k with
the same model, prompts, PDF, embeddings, and temperature 0. Both runs completed
without application errors. Applicable deterministic metrics were 1.0 in both
language groups. These are small, authored fixtures, not a held-out benchmark.

| Top-k | Application input tokens | Application output tokens | Median case latency | Maximum case latency |
|---|---:|---:|---:|---:|
| 3 | 46,720 | 5,345 | 3.772 s | 12.008 s |
| 6 | 72,869 | 5,449 | 4.092 s | 12.032 s |

After correcting cloud evaluator mappings, the original outputs were submitted
as captured-output experiments without repeating retrieval or generation:

- [Top-k 3 scored experiment](https://cloud.langfuse.com/project/cmvfrfxxb004lad0miqv5n24i/experiments/results?baseline=e522ee9b5e365318)
- [Top-k 6 scored experiment](https://cloud.langfuse.com/project/cmvfrfxxb004lad0miqv5n24i/experiments/results?baseline=3dbe95dfb54b8ff1)

API readback confirmed all 19 deterministic case results and 52 AI scores per
experiment: 19 correctness, 19 grounding, and 14 citation coverage. Every received
AI score was 1.0. Citation coverage remains unscored for the five non-answerable
cases. There is no measured quality difference on this dataset; lower token use
at k=3 does not establish its adequacy for unseen questions. Judge usage is separate
from the application tokens above. Human scores still require manual review.

Local reports are under `outputs/base_rag_architecture/langfuse/`; the
`topk-*-cloud-status.json` files record the final asynchronous score readback.
Three real streamed UI turns (Greek question, Greek follow-up, English question)
also produced cloud traces grouped under one conversation. Cloud records were
verified through the API; browser dashboard inspection requires signing in.

Validation: 55 focused backend/UI tests passed, Ruff passed, and the wheel built.
The full repository suite and a separate type checker were not run.

### Check whether the judges can detect failures

On 20 October 2026, four controlled variants of the Greek expiry answer were
submitted separately to the existing cloud judges. They reused captured PDF
passages and did not run retrieval or answer generation. Baseline answers,
datasets, and evaluator definitions were unchanged. The judge inputs excluded
the original deterministic scores to avoid exposing a suggested verdict.

| Answer variant | Correctness | Grounding | Citation coverage |
|---|---:|---:|---:|
| Correct expiry, supporting clause 3.3 | 1 | 1 | 1 |
| False five-year expiry, citing clause 3.3 | 0 | 0 | 0 |
| Correct expiry, citations removed | 1 | 1 | 0 |
| Correct expiry, citing only clause 3.2 | 1 | 1 | 0.5 |

[Open the judge-control experiment](https://cloud.langfuse.com/project/cmvfrfxxb004lad0miqv5n24i/experiments/results?baseline=06772b33632d7a8e).
The saved evidence is `outputs/base_rag_architecture/langfuse/judge-calibration.json`.
All 12 scores were read back from the cloud. These controls demonstrate that the
judges are not constant-1 evaluators; they do not establish general reliability.

The last result is a calibration disagreement: clause 3.2 supports none of the
answer's expiry claims, so the authored expectation is 0 under the current rubric.
The judge's explanation correctly identifies the missing clause 3.3 but awards
partial credit. Review that explanation in class. Do not treat 0.5 as a calibrated
probability or silently change the label to count it as agreement. Collect more
independently reviewed failures before revising the rubric/model, then validate
the revision on separate controls. Perfect baseline scores alone cannot establish
that the judges are reliable.

References: [LangGraph integration](https://langfuse.com/integrations/frameworks/langgraph),
[experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk),
[AI judges](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge),
[troubleshooting mappings](https://langfuse.com/faq/all/observation-eval-not-executing).
