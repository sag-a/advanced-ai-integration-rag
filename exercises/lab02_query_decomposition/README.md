# Lab 2: Query decomposition, and whether it is worth it

After lesson 3 · about 45 minutes · no API key for the tests; `--llm` measuring costs under $0.01

## The problem

Questions that span several clauses are the hardest in this course: *"How do I get into
the rewards scheme, how do I collect points, what can I spend them on, and how long do
they last?"* One vector for the whole question lands near *some* of the clauses. The
textbook fix is **query decomposition**: search for each part separately and merge.

This lab has a twist: on this course's data, decomposition is **not** a free win. You
will build it, measure it, and decide.

## Background

- **Rule-based splitting** cuts on punctuation and "and"/"και". It's free and fast, but
  the parts lose their subject: *"how long do they last"* doesn't say what "they" is.
- **LLM decomposition** asks a model for *standalone* sub-questions that repeat the
  subject. It costs one call per question.
- **Merging** must be fair. With a budget of k passages, a **round-robin** merge gives
  every search a turn: the full question's best hit, then each part's best hit, then
  everyone's second hit. Keeping the full question first protects single-topic answers.

## Your tasks

Edit [`starter.py`](starter.py):

1. `split_question`: rules, with a 3-word minimum per part and at most 4 parts.
2. `merge`: round-robin without duplicates.
3. `DECOMPOSE_PROMPT`: instructions that produce standalone sub-questions.
4. `llm_decompose`: call the model, clean the answer, fall back safely.

## Check your work

```bash
uv run --all-extras pytest exercises/lab02_query_decomposition
LAB_TARGET=solution uv run --all-extras pytest exercises/lab02_query_decomposition
```

The tests use a fake model, so they cost nothing.

## Measure it

```bash
uv run --all-extras python -m exercises.lab02_query_decomposition.measure          # rules
uv run --all-extras python -m exercises.lab02_query_decomposition.measure --llm    # Claude Haiku
```

Reference solution, recall on the answerable cases (`split` = questions it split):

| Corpus | Method | split | dense@3 | decomposed@3 | dense@6 | decomposed@6 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Magenta | rules | 14/17 | 0.922 | 0.863 | 1.000 | 0.941 |
| Magenta | LLM | 12/17 | 0.922 | 0.922 | 1.000 | 0.980 |
| Tariffs | rules | 9/13 | 0.564 | 0.564 | 0.923 | **0.667** |
| Tariffs | LLM | 11/13 | 0.564 | 0.603 | 0.923 | 0.897 |

Per question, LLM decomposition helps some cases (s17 from 0.67 to 1.00, t13 from 0.33
to 0.67) and hurts others (s18 from 0.67 to 0.33). Two LLM runs gave 0.590 and 0.603 on
tariffs: the model is not deterministic, so report a range, not one number.

## Discuss

1. Why does decomposition *lower* recall at k=6 even though each part finds good passages?
   (Hint: count how many slots each part gets.)
2. Rules split 14 of 17 Magenta questions, but only 6 really have several parts. What
   does over-splitting cost?
3. The rules drop "πώς αποχωρώ" ("how do I leave") because it has two words. Fix it
   without breaking the other tests. What did you trade?
4. Agentic RAG decomposes *only when needed*. Sketch a rule or a classifier that decides
   when to decompose, and how you would measure it.
5. Would you ship decomposition for this course? Defend your answer with the table.
