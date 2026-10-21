# Lesson 3: Classic (base) RAG

Suggested teaching time: 50 minutes. Teach from [lesson.html](lesson.html); press
**P** to present one section per screen.

## Concept

One LangGraph turn of three nodes: `prepare_query` rewrites follow-ups,
`retrieve` runs one cosine search for the top 6 passages, and `generate_answer`
answers from those passages only. The application then checks every citation
number and builds the page footers itself. The other architectures in the course
replace only the `retrieve` node.

Code: `rag_core/graph.py`, `rag_core/store.py`, `rag_core/answers.py`,
`architectures/classic_rag/retriever.py`.

## Three commands

```bash
./rag search "Πότε λήγουν οι Καρδιές;"                  # passages only, no model call
./rag ask --process "Πότε λήγουν οι Καρδιές;"          # full turn, each step shown
./rag chat                                              # then ask a follow-up: "Και μπορώ να τις μεταφέρω;"
```

## What to look for

- The search query in a follow-up turn: it becomes a standalone question.
- Six passages are always retrieved, even when the question cannot be answered.
  “What is my Hearts balance?” still ends as `insufficient_evidence`.
- The footer is built from metadata: `[1] … · PDF p. 2 · §3.3`. The model never
  types page numbers.
- A valid citation proves membership, not support. Ragas `faithfulness` in `evals/` measures support.

## Reranking (optional)

`--option rerank=true` adds a second step: vector search returns 20 candidates,
then a local cross-encoder, [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
(free, Apache-2.0, multilingual), scores each one against the question and keeps
the best six. It reads question and passage together, so it judges relevance more
precisely than comparing two embeddings, but it is too slow to run on every passage.

```bash
uv run --all-extras python -m rag_core.reranker --download   # once, about 2.3 GB
./rag search --option rerank=true "Πότε λήγουν οι Καρδιές;"
```

The terminal prints the 20 passages before reranking (vector distance) and the six
after (reranker score, with each passage's old rank in `was #`). Scoring 20
passages takes about 1.5 s on a laptop CPU. Change the pool with
`--option rerank_candidates=12` (top-k to 20). Code: `rag_core/reranker.py` (shared with Jev RAG).

Measured on the course cases (retrieval recall, `top_k=3`): it lifts the 156-passage
tariffs corpus from 0.564 to 0.897, but slightly lowers the small Magenta PDF from
0.922 to 0.863. See [evals/README.md](../../evals/README.md#reranking).

## Instructor notes

- [BASELINE.md](instructor_notes/BASELINE.md): the first measured baseline, the three
  failures found and how they were fixed.
- [LANGFUSE.md](instructor_notes/LANGFUSE.md): historical only, since Langfuse was removed
  from the code. Its judge-calibration exercise (a correct answer citing only §3.2)
  still works as a whiteboard discussion.
