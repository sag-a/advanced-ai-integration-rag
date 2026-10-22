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
| One day | about 7 h | Five lessons, one optional segment, breaks, and a closing comparison |
| Two half-days | 2 × 3 h | Day 1: lessons 1–3. Day 2: lessons 4–5 and evaluation |
| Weekly | 5 × 1.5 h | One lesson per week, with the demo questions as homework |

## One-day timetable

| Time | Lesson | Teach from |
| --- | --- | --- |
| 09:00–09:20 | Setup check: `uv run --all-extras pytest -q`, Docker running | [SETUP.md](SETUP.md) |
| 09:20–10:10 | 1. Embeddings, chunking, semantic search | [c01 README](modules/c01_embeddings/README.md) |
| 10:10–10:25 | Break | |
| 10:25–11:25 | 2. Vector databases: pgvector and Chroma (60 min, as planned in the study guide) | [c02 study guide](modules/c02_vector_databases/study_guide.html) |
| 11:25–12:15 | 3. Classic (base) RAG | [c03 lesson](modules/c03_classic_rag/lesson.html) |
| 12:15–13:10 | Lunch | |
| 13:10–14:00 | 4. Graph RAG | [c04 lesson](modules/c04_graph_rag/lesson.html) |
| 14:00–14:15 | Break | |
| 14:15–15:05 | 5. Agentic RAG | [c05 lesson](modules/c05_agentic_rag/lesson.html) |
| 15:05–15:25 | Optional, pick one: Jev gate and router, or dense vs hybrid on the tariffs corpus | Sections below |
| 15:25–16:00 | Comparison and evaluation | `#compare` section of any lesson page |

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

## Optional: a gate before the LLM with Jev (15 min)

**Must land:** a small decision model can decide *whether* to answer before the
expensive model runs. Abstaining early saves a call and avoids answering from the
wrong passages, but every false refusal is a lost correct answer.

**Demo:** with `--process`, ask Jev RAG "What is my current Hearts balance?" The
gate prints P = 0.07 and `generate_answer` makes no model call. Then ask "Πότε
λήγουν οι Καρδιές;": P = 0.99, and Claude answers.

**Whiteboard:** *“Our first gate asked whether the passages contain the information
needed. It refused three answerable questions. Why, and what fixed it?”* All three
needed several clauses, and the top 6 held only some of them. Asking whether the
passages answer the question "at least in part" kept every answerable question
while still blocking the three unanswerable ones (see `evals/README.md`).

**Trap:** students assume a stricter gate is safer. It's only safer against one
kind of error.

**Extension: Jev as supervisor.** Add `--option route=true`. Jev now picks the
document before any search: ask a Magenta question, a tariffs question and "What's
the weather in Athens tomorrow?" and point at the `Jev router:` line for each. Ask
why "What is my Hearts balance?" routes to Magenta instead of "neither": the router
judges the topic, the gate judges whether the answer is there. Two small decisions
with one job each are easier to test than one decision doing both.

## Optional: dense vs hybrid on a harder corpus (20 min)

**Must land:** keyword search helps when many passages look alike and the
question names an exact number, and it fails on spelling and extraction noise.
Neither method wins everywhere.

**Setup (before class):** `./rag prepare --corpus data/corpus-tariffs --collection cosmote_tariffs_v1`.
The corpus is OTE's 2019 residential price list; say that the prices are historical.

**Demo:** run the two `./rag search` commands in [SETUP.md](SETUP.md) for "Double
Play 24 XL 500 free minutes". Seven programs are called "Double Play 24 XL", and only
§4 (page 5) has 500 free minutes. Classic's top 3 are all other 24 XL plans (150 and
360 minutes); Hybrid ranks §4 second because the keyword "500" matches. Both still
rank §48 first, so point out that hybrid improved the list, not the top hit. Then run
the tariffs evaluation at `--top-k 3` and `--top-k 6` for both architectures (commands
in [evals/README.md](evals/README.md#harder-corpus-tariffs)).

**Whiteboard:** *“Hybrid scores 0.73 at k=3 but 0.82 at k=6, while dense goes from
0.56 to 0.92. Why does hybrid gain less from depth?”* Fusion fills the extra slots
with keyword matches from other plans that share the same boilerplate, so more
depth adds distractors as well as the right passage.

**Extension:** add `--option rerank=true` to Classic RAG. On this corpus the
reranker reaches 0.897 recall at k=3, beating both plain dense (0.564) and hybrid
(0.731), and the terminal prints the passages before and after reranking. Download
the weights before class (`SETUP.md`, step 3).

**Trap:** the Bulgaria case. The PDF text reads "Bulgar ia" in the program asked
about, so keyword search confidently picks other programs. Inspect the extracted
text before trusting either method.

## Labs

Five labs in [`exercises/`](exercises/README.md) fit the two-half-day or weekly formats,
or work as homework after a one-day course. Each takes 30–45 minutes, is graded by
offline tests (`uv run --all-extras pytest exercises/<lab>`), and ends with a measure
step that compares the student's code with the course's numbers.

| Lab | Pairs with | The lesson it lands |
| --- | --- | --- |
| 1. BM25 + hybrid | Lesson 2 | Keywords fix look-alike passages, but Greek inflection breaks BM25 |
| 2. Query decomposition | Lesson 3 | A textbook technique that doesn't pay off here; measure before adopting |
| 3. Citation validator | Lesson 3 | Membership is not support; numbers are cheap to check |
| 4. Gate calibration | Jev segment | Tune on dev, report on held-out |
| 5. Semantic cache | Lesson 3 | Near-misses look more similar than paraphrases |

Grading: the tests are pass/fail. For a graded course, weight the discussion questions at
the end of each README; they ask students to interpret their measurements.

## Closing: compare and evaluate

Ask one question with Classic, Graph and Agentic RAG (add Hybrid or Jev if you ran an
optional segment) and put the passage lists side by side. Then run:

```bash
uv run --all-extras python -m evals.run --architecture classic_rag --retrieval-only
```

Close on the honest finding: on this small corpus all three tie on accuracy. They
differ in what you can inspect, what they cost, and how they fail. Choosing one is
an engineering trade-off, not a leaderboard.
