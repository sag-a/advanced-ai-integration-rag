# Capstone: build a RAG application

After lessons 1–3 · about 60 minutes · no Docker; an Anthropic key only for real answers

A command-line app that answers questions about five support documents in
[`docs/`](docs/), cites its sources, and says so when the documents don't contain the
answer. [`starter.py`](starter.py) already runs, but it makes seven mistakes. Each one is
marked `STEP n` in the code, says what you will see and how to fix it, and needs a change
of about one line. For every step:

1. Run the command in the table and look at what goes wrong.
2. Find `STEP n` in `starter.py` and make the fix.
3. Run the command again and compare, then run the step's test.

## The steps

| Step | What goes wrong | See it with | Test |
| --- | --- | --- | --- |
| 1. Explore | Nothing: read `docs/`, write 3 answerable and 1 unanswerable question | Point to the file and section for each answer | — |
| 2. Load | The front matter is thrown away: metadata has only `source` | `… load` | `-k step2` |
| 3. Chunk | The refund rule and its exception land in different chunks | `… chunks` | `-k step3` |
| 4. Store | Every `index` run adds the chunks again (18 → 36 records) | `… index`, twice | `-k step4` |
| 5. Retrieve | `--current` still finds the superseded 2023 refund policy | `… search "full refund" --current` | `-k step5` |
| 6. Answer | Unanswerable questions get an improvised excuse and no citations | `… ask "What is the customer service phone number?"` (or `--dry-run`) | `-k step6` |
| 7. Evaluate | Add your own question from Step 1, then experiment | `… eval --chunk-size 150 --n 3` | `-k step7` |
| 8. LangGraph | The graph calls the model even when no chunk is close to the question | `… graph "What are your opening hours?" --dry-run` | `-k step8` |

`…` stands for `uv run --all-extras python -m exercises.capstone_rag.starter`. Tests:
`uv run --all-extras pytest exercises/capstone_rag -k step2` (then `step3`, ...). Before
you change anything, 13 tests pass and 8 fail; each fix turns its own step green.

Without an Anthropic key, use `ask --dry-run`: it prints the exact prompt the model would get.

## What a finished app does

```text
$ … ask "Can I get a full refund within 14 days if I already used the service?"
=== ANSWER ===
No, you cannot get a full refund within 14 days if you have already used the service.
While customers can receive a full refund if they cancel within 14 days of purchase,
there is an exception: "Customers who have already used the service are not eligible
for a full refund." [refunds.md]

$ … ask "What is the customer service phone number?"
=== ANSWER ===
The documents do not contain this information.
```

## Step 7: experiment

`--chunk-size N` builds a throwaway in-memory collection with that chunk size, so you can
try settings without touching the saved one in `db/`. It works with `eval`, `search` and
`ask`. Change one setting at a time:

| Setting | Evidence found (`eval`: the six built-in questions, after Steps 2–6) |
| --- | --- |
| `--n 1` (chunk size 500) | 6/6 |
| `--chunk-size 150 --n 1` | 2/6: rules lose their exceptions |
| `--chunk-size 150 --n 3` | 5/6 |
| `--chunk-size 300 --n 1` | 6/6 |

To change the saved collection, edit `CHUNK_SIZE` and run `index --rebuild`: without
`--rebuild` the old chunks stay next to the new ones. The saved collection in `db/` is
local and shared by `starter` and `solution`, so rebuild it when you switch between them.

## Step 8: the same app as a LangGraph graph

`graph` runs the pipeline as a graph with a decision after retrieval:

```text
START → retrieve ─┬─► answer → END     the closest chunk is within MAX_DISTANCE: call the model
                  └─► refuse → END     nothing is close: reply with the refusal, no model call
```

`route()` makes the decision; the fix connects it with `graph.add_conditional_edges`. With
Nomic, the questions the documents answer have a closest chunk at about 0.30–0.47, and
questions they don't answer at 0.70 or more, so `MAX_DISTANCE = 0.6` sits between them.
Then experiment: set `MAX_DISTANCE = 0.3` and ask "Can I pay with cash?" (0.415). What
goes wrong, and which is worse for a support bot: refusing too often or too rarely?
