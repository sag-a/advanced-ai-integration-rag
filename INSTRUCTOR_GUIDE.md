# Instructor guide: Advanced RAG

How to deliver the five lessons: timetable, what each lesson must land, live
demos, whiteboard questions with answers, and the traps students fall into.
Setup for every machine is in [SETUP.md](SETUP.md).

## Audience and format

For engineers who have called an LLM API and want to build retrieval they can
inspect and measure. Every lesson follows the same shape: 15 minutes of concept
from the lesson page, 20 minutes of live demo, 15 minutes of questions and a
whiteboard prompt.

| Format | Duration | Structure |
| --- | --- | --- |
| One day | about 6 h | Five 50-minute lessons, breaks, and a closing comparison |
| Two half-days | 2 × 3 h | Day 1: lessons 1–3. Day 2: lessons 4–5 and evaluation |
| Weekly | 5 × 1.5 h | One lesson per week, with the demo questions as homework |

## One-day timetable

| Time | Lesson | Teach from |
| --- | --- | --- |
| 09:00–09:20 | Setup check: `uv run --all-extras pytest -q`, Docker running | [SETUP.md](SETUP.md) |
| 09:20–10:10 | 1. Embeddings, chunking, semantic search | [c01 README](modules/c01_embeddings/README.md) |
| 10:10–10:25 | Break | |
| 10:25–11:15 | 2. Vector databases: pgvector and Chroma | [c02 study guide](modules/c02_vector_databases/study_guide.html) |
| 11:15–12:05 | 3. Classic (base) RAG | [c03 lesson](modules/c03_classic_rag/lesson.html) |
| 12:05–13:00 | Lunch | |
| 13:00–13:50 | 4. Graph RAG | [c04 lesson](modules/c04_graph_rag/lesson.html) |
| 13:50–14:05 | Break | |
| 14:05–14:55 | 5. Agentic RAG | [c05 lesson](modules/c05_agentic_rag/lesson.html) |
| 14:55–15:30 | Comparison and evaluation: same question, three architectures | `#compare` section of any lesson page |

Present lessons 3–5 with **P** (one section per screen, ← → to move, Esc to exit).

## Before class

- [ ] `docker compose -f compose.chroma.yaml up -d --wait`, then `./rag check`.
- [ ] `./rag check --architecture graph_rag | head -20` passes. Do **not** rebuild the
      graph in class; it is about 30 paid calls.
- [ ] Run each demo question once and note the timings, especially Agentic RAG.
- [ ] Lesson 2 services are on 8002/5434; lessons 3–5 use 8004. They can run together.
- [ ] Run one warm-up question per terminal: the first turn loads the Nomic encoder (about 9 s).
- [ ] Add `--process` to CLI demos so students see each node and model call. With `./rag ui`,
      keep its terminal visible next to the browser: the steps are printed there.
- [ ] Keep the default `--model`. Any other model makes the graph index incompatible.

## Lesson 1: Embeddings, chunking, semantic search

**Must land:** text becomes a vector, similar meaning gives nearby vectors, and the
chunk boundaries you pick decide what can ever be retrieved.

**Demo:** run `m01` → `m05` in order (commands in [SETUP.md](SETUP.md#4-lesson-1-embeddings-no-api-key-no-docker)).
In `m02`, show the same word moving with its context.

**Whiteboard:** *“Why does the encoder need `search_query:` and `search_document:`
prefixes?”* Nomic v2 is trained for asymmetric retrieval. A short question and a
long passage are embedded in two task modes. Mixing them, or dropping them,
silently lowers similarity.

**Traps:**
- Assuming v1.5 vectors are compatible because both have 768 dimensions. They are
  not, which is why the embedding contract is hashed into every collection.
- Thinking long chunks are “more context”. Beyond 512 tokens the input is rejected,
  not truncated.

## Lesson 2: Vector databases

**Must land:** a vector store is an index plus metadata filters. pgvector puts it
in SQL; Chroma wraps it in an API with a configured embedding function.

**Demo:** M01 `setup → insert → index → search → explain`, then M02 `filter` and
`search --method compare` for dense vs lexical vs hybrid.

**Whiteboard:** *“HNSW returned a different top 3 than the exact scan. Is the index
broken?”* No. HNSW is approximate and trades exactness for speed. `explain` shows
when the planner used the index at all.

**Traps:** a filter that matches nothing returns nothing, not an error. Show the
`--category absent-category` case.

## Lesson 3: Classic (base) RAG

**Must land:** the three-node turn every architecture shares, and that the
application, not the model, validates citations and writes the page footers.

**Sections:** [#flow](modules/c03_classic_rag/lesson.html#flow) →
[#steps](modules/c03_classic_rag/lesson.html#steps) →
[#answer](modules/c03_classic_rag/lesson.html#answer) →
[#identity](modules/c03_classic_rag/lesson.html#identity).

**Demo:** `./rag search`, then `./rag ask`, then in `./rag chat` the follow-up
“Και μπορώ να τις μεταφέρω;” to show the rewrite. Finish with “What is my Hearts
balance?” and point out that six passages came back but the status is
`insufficient_evidence`.

**Whiteboard:** *“The answer cites [2] and [2] is a real retrieved passage. Is the
answer grounded?”* Not necessarily. Validation proves membership, not support.
Use the judge-calibration case in
[LANGFUSE.md](modules/c03_classic_rag/instructor_notes/LANGFUSE.md): a correct
expiry answer citing only §3.2 passes membership but fails citation coverage.

**Traps:**
- Brackets such as “[2024]” in an answer are parsed as citations and reject the turn.
- History is the last six messages. A subject from earlier is lost, and the turn
  asks for clarification instead.

**Measured history:** [BASELINE.md](modules/c03_classic_rag/instructor_notes/BASELINE.md)
lists three real failures and their fixes. They make good “what would you do?” prompts.

## Lesson 4: Graph RAG

**Must land:** the graph is built once by an LLM and walked at question time with
no LLM. Every edge carries an exact quote, so the walk always lands on original text.

**Sections:** [#graph](modules/c04_graph_rag/lesson.html#graph) →
[#build](modules/c04_graph_rag/lesson.html#build) →
[#query](modules/c04_graph_rag/lesson.html#query) →
[#contract](modules/c04_graph_rag/lesson.html#contract).

**Demo:** “Μπορώ να εξαργυρώσω τις Καρδιές για μετρητά;” to show the negative edges,
then “Who can join Magenta Moments and what do members get?” for two hops. Compare
the passage list with lesson 3 on the same question.

**Whiteboard:** *“Why does changing `--model` for answering break the graph?”*
The extraction model is part of the index contract. A different model would have
extracted a different graph, so the code refuses to mix them.

**Traps:**
- Expecting better scores. On the ten-case dataset Graph RAG tied the baseline;
  see [MEASUREMENTS.md](modules/c04_graph_rag/instructor_notes/MEASUREMENTS.md). The
  value is inspectable expansion, not a measured gain on a five-page PDF.
- The error message mentions `--index`; the real flag is `--option graph_index=PATH`.

## Lesson 5: Agentic RAG

**Must land:** an agent can plan retrieval, but its limits are enforced by code.
It selects evidence and stops; the shared generator answers.

**Sections:** [#shape](modules/c05_agentic_rag/lesson.html#shape) →
[#loop](modules/c05_agentic_rag/lesson.html#loop) →
[#budgets](modules/c05_agentic_rag/lesson.html#budgets) →
[#guard](modules/c05_agentic_rag/lesson.html#guard).

**Demo:** the narrow expiry question, then the two-topic “How do I join … and can I
transfer my Hearts?” and watch for `delegate`. Finish with `--option max_searches=1`
to show a rejected turn on purpose.

**Whiteboard:** *“The agent ran out of budget after finding two good passages.
Why not answer with those?”* Because it has not shown the evidence is complete.
Failing closed is cheaper than a confident partial answer.

**Traps:**
- Unplanned budget failures in the middle of a demo. Keep defaults for the main run.
- Agentic is slower and costlier (about 14 s per turn in the measured run). For
  simple questions Base RAG is the better tool; say so.
- Abstentions lower faithfulness and citation-support scores. Explain why using
  [MEASUREMENTS.md](modules/c05_agentic_rag/instructor_notes/MEASUREMENTS.md).

## Closing: compare and evaluate

Ask one question with all three architectures and put the passage lists side by
side. Then run:

```bash
uv run --all-extras python -m evals.run --architecture classic_rag --retrieval-only
```

Close on the honest finding: on this small corpus all three tie on accuracy. They
differ in what you can inspect, what they cost, and how they fail. Choosing one is
an engineering trade-off, not a leaderboard.
