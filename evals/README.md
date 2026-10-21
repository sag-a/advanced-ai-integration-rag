# Evaluate the course with Ragas

`evals/run.py` runs the 21 cases in [datasets/cases.jsonl](datasets/cases.jsonl)
through the same graph as the CLI and UI, then scores them with
[Ragas](https://docs.ragas.io/) 0.4.3. They are Greek and English questions about
the Magenta Moments PDF, and classroom examples rather than unseen test data:

| Type | Cases | What it tests |
| --- | --- | --- |
| `factual` | s01–s03, s05, s06 | One clause, asked in the PDF's own words |
| `followup` | s04, s08 | Needs the conversation history; s08 is ambiguous on purpose |
| `multihop` | s07 | Two clauses on different pages |
| `abstain` | s09, s10 | The documents cannot answer; the turn must refuse |
| `paraphrase` | s11–s16 | One clause, asked in everyday words ("loyalty points", "voucher") |
| `multihop3` | s17–s21 | Three clauses, usually from different sections or pages |

Measured retrieval recall for Classic RAG shows where the difficulty is:

| Type | k=1 | k=2 | k=3 | k=6 |
| --- | ---: | ---: | ---: | ---: |
| factual | 1.000 | 1.000 | 1.000 | 1.0 |
| paraphrase | 1.000 | 1.000 | 1.000 | 1.0 |
| multihop | 0.000 | 0.500 | 1.000 | 1.0 |
| multihop3 | 0.267 | 0.533 | 0.733 | 1.0 |

Paraphrasing does not hurt: the multilingual Nomic encoder matches meaning, not
words, so "loyalty points" still finds the Καρδιές clause first. Questions that
span several clauses are what need retrieval depth. At the default `top_k=6` every
type reaches 1.0, because 6 of 30 passages is a fifth of the whole PDF; use
`--top-k 3` to see the multi-clause cases fail.

## Run

From the repository root, with Chroma running and the PDF prepared:

```bash
# Retrieval scores only: no model calls, no cost.
uv run --all-extras python -m evals.run --architecture classic_rag --retrieval-only

# Answers plus Claude-judged Ragas metrics. Anthropic usage applies.
uv run --all-extras python -m evals.run --architecture classic_rag

# Every registered architecture, one after another.
uv run --all-extras python -m evals.run --architecture all

# A cheap subset, or a different retrieval depth.
uv run --all-extras python -m evals.run --architecture graph_rag --case s01-enrollment --case s09-private-account
uv run --all-extras python -m evals.run --architecture classic_rag --retrieval-only --top-k 3
```

The command prints the mean of each score per architecture and writes one row per
case to `evals/reports/<date>_<time>.csv` (or `--output PATH`). It exits with 1 if
any case failed. Retrieval-only mode skips the two follow-up cases and Agentic RAG,
because both need the model.

## What is scored

| Score | Ragas metric | Model calls | Meaning |
| --- | --- | --- | --- |
| `id_based_context_recall` | `IDBasedContextRecall` | no | Share of the labelled evidence chunks that were retrieved |
| `id_based_context_precision` | `IDBasedContextPrecision` | no | Share of retrieved chunks that are labelled evidence |
| `faithfulness` | `Faithfulness` | yes | Share of the answer's claims supported by the retrieved passages |
| `context_recall` | `LLMContextRecall` | yes | Share of the expected facts that the retrieved passages support |
| `factual_correctness(mode=f1)` | `FactualCorrectness` | yes | Overlap between the answer's claims and the expected facts |
| `status_match` | (plain check) | no | 1 when the turn answered, refused or asked to clarify as expected |

- **Ragas scores only the 18 answerable cases.** The ambiguous and unanswerable
  cases (s08–s10) have no facts to compare against, so only `status_match` applies to them.
- **Evidence labels become chunk IDs.** Each case names its evidence by page and
  clause; the runner looks up the matching chunks in the prepared collection.
- **The judge is Claude.** It's the same model as the app (`claude-haiku-4-5` by
  default), wrapped with Ragas's `LangchainLLMWrapper`.
- **`factual_correctness` is often low for good answers.** The expected facts are
  short, and a complete answer adds true details they don't list, which lowers
  precision. Read the per-case CSV before drawing conclusions.
- **The reference-footer lines are removed** from each answer before scoring, so
  `[1] … · PDF p. 2` isn't counted as a claim.

Ragas 0.4 marks `evaluate()` and `LangchainLLMWrapper` as deprecated in favour of
`@experiment` and `llm_factory`. Both work in the pinned 0.4.3. The newer path
needs `instructor` 1.x, which needs `rich<15`, and this project pins `rich==15.0.0`.
The runner hides those warnings and turns off Ragas usage analytics.

## Change the cases

Edit `datasets/cases.jsonl`, one JSON object per line:

```json
{"id": "s01-enrollment", "language": "el", "type": "factual", "split": "dev",
 "question": "…", "history": [], "expected_status": "answered",
 "expected_evidence": [{"document": "Magenta_Moments_Terms_Conditions_gr.pdf", "page": 1, "clause": "2.1"}],
 "expected_facts": ["…", "…"]}
```

`history` holds earlier turns as `{"role": "human" | "ai", "text": "…"}`. An
evidence label that matches no chunk stops the run with the case ID.

The `.json`/`.md` files already in `reports/` come from the previous custom
evaluator and use different metrics; new runs write `.csv`.
