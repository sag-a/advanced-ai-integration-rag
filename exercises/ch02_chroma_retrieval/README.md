# Chapter 2 exercise: fix three Chroma queries

After lesson 2 (vector databases, slides 40–46) · about 20 minutes · no API key, no Docker

[`starter.py`](starter.py) loads eight short policy chunks into an in-memory Chroma
collection. Each chunk has metadata: `category` (billing, technical, account),
`version` and `status`. Version 1 of the refund rule ("30 days") was replaced by
version 2 ("14 days") and is marked `superseded`.

Three queries are broken. Fix each by changing **one setting** at the top of
`starter.py`. Leave `POLICIES` and the query functions unchanged.

| Exercise | Intentionally broken setup | You change | Success criterion |
| --- | --- | --- | --- |
| 1. Missing information | One chunk is retrieved for a question whose answer needs two | `N_RESULTS`: compare 1, 3 and 5 | The retrieved chunks include both the rule and its exception |
| 2. Wrong category | A billing question is filtered to technical support documents | The category in `BILLING_FILTER` | Results come from the correct category |
| 3. Outdated information | The search returns both current and outdated policies | Add a version or status condition to `POLICY_FILTER` | Only the current policy version appears |

## Run

```bash
uv run --all-extras python -m exercises.ch02_chroma_retrieval.measure     # what each query returns, with Nomic
uv run --all-extras pytest exercises/ch02_chroma_retrieval                # 6 tests
```

You're done when `measure` prints `OK` three times and all 6 tests pass.

## Chroma syntax you need

```python
collection.query(query_texts=[question], n_results=3)                     # how many chunks
collection.query(query_texts=[question], n_results=3,
                 where={"category": "billing"})                           # one condition
where={"$and": [{"category": "billing"}, {"status": "current"}]}          # two conditions
where={"version": {"$gte": 2}}                                            # a comparison
```

Other operators: `$ne`, `$gt`, `$lt`, `$lte`, `$in`, `$nin`, `$or`. See
[Chroma's metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering).

## Discuss

- Exercise 1: with `N_RESULTS = 1` you get the **exception** without the rule. With 5 you
  get both, plus unrelated chunks. What does each extra chunk cost? Which chunk comes back
  with 3 that you don't want? (The outdated 30-day rule: that is exercise 3.)
- Exercise 2: the wrong filter still returned two results. Why didn't Chroma report an
  error? Who should decide the category: the user, the model or the application?
- Exercise 3: `{"status": "current"}` and `{"version": {"$gte": 2}}` both work. Which
  survives the next policy update? What happens if you drop the category and filter on
  status alone?
- Slide 54 shows the same idea in SQL: `tenant_id IN (...) AND valid_to IS NULL`. In
  production, the server adds these filters, never the user's question.

## For instructors

Answers: `N_RESULTS = 3` (2 also works with Nomic; 1 fails), `BILLING_FILTER =
{"category": "billing"}`, `POLICY_FILTER = {"$and": [{"category": "billing"}, {"status":
"current"}]}` or the `$gte` version. The tests use a small word-overlap embedding so they
run offline in under a second; `measure.py` uses the real Nomic encoder. Grade the
reference with `LAB_TARGET=solution uv run --all-extras pytest exercises/ch02_chroma_retrieval`.
