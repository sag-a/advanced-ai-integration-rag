# Labs

Five hands-on labs. Each one targets a weakness the course measured, and each ends by
measuring your code against the course's numbers.

| Lab | After | You build | You measure | Cost |
| --- | --- | --- | --- | --- |
| [1. BM25 from scratch, then hybrid](lab01_bm25_hybrid/) | lesson 2 | BM25 scoring and reciprocal rank fusion | Recall@3 on both corpora vs dense and TF-IDF | free* |
| [2. Query decomposition](lab02_query_decomposition/) | lesson 3 | Rule-based and LLM splitting, a fair merge | Whether decomposition helps at all | free*; `--llm` < $0.01 |
| [3. Citation validator](lab03_citation_validator/) | lesson 3 | Citation checks without the "[2024]" bug, a number-grounding check | — | free |
| [4. Gate calibration](lab04_gate_calibration/) | Jev segment | Threshold choice on dev data | Dev vs held-out errors: the overfitting trap | free |
| [5. Semantic cache](lab05_semantic_cache/) | lesson 3 | A cache with a threshold and a guard | Correct reuse vs wrong answers served | free |

\* Labs 1 and 2 measure against the prepared Chroma collections (see `SETUP.md`): start
Chroma and run `./rag prepare` for both corpora first. Lab 5 loads the local Nomic encoder.

## How a lab works

Each folder has a `README.md` (problem, background, tasks, discussion), a `starter.py`
with numbered `TODO`s, a `solution.py`, a `test_lab.py`, and usually a `measure.py`.

```bash
# Grade your starter (fails until you finish the TODOs)
uv run --all-extras pytest exercises/lab01_bm25_hybrid

# Grade the reference solution
LAB_TARGET=solution uv run --all-extras pytest exercises/lab01_bm25_hybrid

# All labs at once
uv run --all-extras pytest exercises

# Measure your implementation, or the reference one
uv run --all-extras python -m exercises.lab01_bm25_hybrid.measure
LAB_TARGET=solution uv run --all-extras python -m exercises.lab01_bm25_hybrid.measure
```

On Windows PowerShell, set the variable first: `$env:LAB_TARGET = "solution"`.

The tests never call a model or a database: they use small fixtures and fakes, so they
run in seconds and cost nothing. CI grades the reference solutions on every push.

## For instructors

- Results quoted in each README were measured on 22 October 2026 with the course's
  default settings.
- Lab 4's `gate_scores.json` was recorded with `record.py` (about 45 Jev calls, under
  $0.01). Re-record after changing the gate, the reranker or the corpora.
- To hide the solutions from students, delete the `solution.py` files on the branch you
  hand out; the tests and measure scripts only need the starter.
