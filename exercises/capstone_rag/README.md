# Capstone: build a RAG application

After lessons 1–3 · about 90 minutes · no Docker; an Anthropic key only for real answers

Build a command-line app that answers questions about five support documents in
[`docs/`](docs/), cites its sources, and says so when the documents don't contain the
answer. You write the pipeline yourself in [`starter.py`](starter.py), one step at a time.

## The steps

| Step | You write | Check it with |
| --- | --- | --- |
| 1. Explore | Nothing: read `docs/`, write 3 answerable and 1 unanswerable question | Point to the file and section for each answer |
| 2. Load | `load_documents()`: filename and front matter as metadata | `… starter load` and `-k step2` |
| 3. Chunk | `chunk_documents()`: split at `#`/`##`, then by size | `… starter chunks` and `-k step3` |
| 4. Store | `open_collection()`, `index_chunks()`: Nomic embeddings, stable IDs, no duplicates | `… starter index` (twice) and `-k step4` |
| 5. Retrieve | `retrieve()`: compare `--n 1/3/5`, try `--current` | `… starter search "…"` and `-k step5` |
| 6. Answer | `SYSTEM_PROMPT`, `build_prompt()`, `generate_answer()` | `… starter ask "…" --dry-run` and `-k step6` |
| 7. Evaluate | Change one setting at a time and record the effect | `… starter eval --n 1` and `-k step7` |

`…` stands for `uv run --all-extras python -m exercises.capstone_rag`. Tests:
`uv run --all-extras pytest exercises/capstone_rag -k step2` (then `step3`, ...). Steps
build on each other, so later tests fail until the earlier steps work.

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

## Step 7: what to measure

| Setting | Evidence found (`eval`) |
| --- | --- |
| `CHUNK_SIZE = 500`, `--n 1` | 5/5 |
| `CHUNK_SIZE = 150`, `--n 1` | 1/5: rules lose their exceptions |
| `CHUNK_SIZE = 150`, `--n 3` | 4/5 |

Remember `index --rebuild` after changing `CHUNK_SIZE`: otherwise the old chunks stay in
the collection next to the new ones.
