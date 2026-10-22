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
any case failed. Retrieval-only mode skips the two follow-up cases, Agentic RAG and
Jev RAG, because they need a paid model call. `--architecture all` stops if Jev RAG's
OpenRouter key or the reranker weights are missing; run the others one at a time.

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

## Harder corpus: tariffs

`datasets/tariffs.jsonl` has 14 cases for the COSMOTE price list (see
[data/README.md](../data/README.md)): near-duplicate programs, exact prices, a
country list, paraphrases (one in Greek), two- and three-program comparisons and
one question the list cannot answer. Point the runner at that corpus:

```bash
uv run --all-extras python -m evals.run --architecture hybrid_rag --retrieval-only --top-k 3 \
  --cases evals/datasets/tariffs.jsonl --corpus data/corpus-tariffs --collection cosmote_tariffs_v1
```

Measured retrieval recall (13 answerable cases, one run each):

| Type | Classic k=3 | Hybrid k=3 | Classic k=6 | Hybrid k=6 |
| --- | ---: | ---: | ---: | ---: |
| near_duplicate (3) | 0.33 | **1.00** | 1.00 | 1.00 |
| exact_number (3) | 1.00 | 1.00 | 1.00 | 1.00 |
| country_list (1) | 1.00 | **0.00** | 1.00 | **0.00** |
| paraphrase (3) | 0.33 | 0.67 | 0.67 | 0.67 |
| multihop (2) | 0.50 | 0.75 | 1.00 | 1.00 |
| multihop3 (1) | 0.33 | 0.00 | 1.00 | 0.67 |
| **mean** | 0.564 | **0.731** | **0.923** | 0.821 |

- **Hybrid wins at k=3,** mainly on near-duplicate programs: keywords such as
  "500" or "480" pick the right one of several plans with the same name.
- **Classic wins at k=6.** With more room, dense search already includes the right
  plan, and keyword candidates push useful passages out.
- **Keyword search fails on the Bulgaria question.** In the program asked about,
  PDF extraction split the word as "Bulgar ia", so keywords match "Bulgaria" in
  three other programs instead. Other lists spell it "Bulgary".
- **Neither finds the international-rollover passage (t10)** at either depth.

## Reranking

Classic RAG with `--option rerank=true` (20 candidates, bge-reranker-v2-m3), same
cases, retrieval-only:

```bash
uv run --all-extras python -m evals.run --architecture classic_rag --retrieval-only --top-k 3 --option rerank=true
```

| Corpus | k=3 off | k=3 on | k=6 off | k=6 on |
| --- | ---: | ---: | ---: | ---: |
| Tariffs (13 answerable cases) | 0.564 | **0.897** | **0.923** | 0.897 |
| Magenta (18 scored cases) | **0.922** | 0.863 | **1.000** | 0.961 |

- **On the tariffs corpus it is the best method at k=3,** ahead of Hybrid (0.731).
  All three near-duplicate cases go from 0.33 to 1.00, and precision rises from
  0.23 to 0.39: the right "Double Play 24 XL" plan moves to the top.
- **On the small Magenta PDF it slightly hurts.** Vector search was already
  close to perfect, so there is little to fix and some room to break: at k=3 the
  reranker drops §2.4 for the "switch provider" paraphrase (s14).
- **Multi-clause questions lose at k=6** (tariffs t13, Magenta s17/s21). The
  reranker scores each passage against the whole question, so a passage that
  answers one of three sub-questions scores lower than it deserves.

## Jev RAG: rerank and gate

`jev_rag` keeps the bge top 6, then asks [Jev](https://openrouter.ai/typesafe/jev-1.14)
(OpenRouter decisions API, `typesafe/jev-1.14`) whether those passages answer the
question, at least in part. Below 0.5 it abstains without calling the answer
model. Because Jev is a paid call, retrieval-only runs skip it; use a normal run:

```bash
uv run --all-extras python -m evals.run --architecture jev_rag \
  --case s01-enrollment --case s09-private-account --case s10-external-policy
```

Measured on 22 October 2026 (top 20 vector candidates; 30 answerable, 3 unanswerable cases):

| Jev as reranker, recall@3 | Dense | bge | Jev |
| --- | ---: | ---: | ---: |
| Magenta | 0.922 | 0.863 | **0.922** |
| Tariffs | 0.564 | **0.897** | 0.756 |

| Gate over the bge top 6 | Unanswerable s09 / s10 / t14 | Lowest answerable | False refusals at 0.5 |
| --- | --- | ---: | ---: |
| Strict: "contain the information needed" | 0.03 / 0.07 / 0.07 | 0.19 | 3 (s17, s21, t10) |
| **Partial: "fully or in part" (default)** | 0.06 / 0.32 / 0.18 | **0.76** | **0** |

- **The gate is the part that adds something.** It separates unanswerable from
  answerable questions, and an abstention costs no answer-model call.
- **The wording of the gate question is a design decision.** A strict question
  refused multi-clause questions whose top 6 held only some of their clauses.
- **bge is the better default reranker.** Jev scores each passage alone, so for
  "how do the two versions differ?" (t12) it rejects both evidence passages,
  which vector search had ranked first and second.
- **A full probe cost $0.03 for 693 Jev calls.** The gate alone is about $0.0001
  per question; `--option reranker=jev` adds 20 calls per question.
- **Questions and passages are sent to OpenRouter.** Fine for these public PDFs;
  say so before students type their own data.

### Routing between the two documents

With `--option route=true`, Jev first answers a `choice` question: is the topic
covered by the Magenta Moments terms, the tariffs price list, or neither? "Neither"
abstains before retrieval. Routing costs about $0.000016 per question.

Measured on all 35 cases plus 5 off-topic questions (weather, a poem, a capital
city, a router password, football): **39 of 40 routed acceptably**. For the three
unanswerable cases either their document or "neither" counts as acceptable. All 5
off-topic questions went to "neither" with probability 1.00.

- **Descriptions are part of the router.** The first Magenta description omitted
  the member profile (§3.4), and s16 was refused.
- **Keep the router to topics.** Describing "neither" as including "personal
  account data" sent profile questions there. Now the router only judges the
  topic and the gate judges answerability: "What is my Hearts balance?" routes to
  Magenta (0.89) and the gate abstains (0.06).
- **One known miss remains:** s16 ("if I delete my age and interests from my
  profile…") never names the programme and still routes to "neither" (0.85). It is
  a fair example of a router lacking context; it was not tuned away.

GLiNER2.5-Decide was tested for the same role and rejected: recall@3 of 0.36
(Magenta) and 0.15 (tariffs), and no separation between answerable and
unanswerable questions. It is a text classifier, not a relevance model.

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
