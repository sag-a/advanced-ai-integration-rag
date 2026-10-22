# Lab 4: Calibrating a gate, and the overfitting trap

After the Jev segment · about 30 minutes · offline, no API key, no Docker

## The problem

Jev RAG lets a question reach the LLM only when the Jev gate's probability is at least
a threshold (default 0.5). Pick the threshold too low and the system answers questions
its documents can't support. Pick it too high and it refuses questions it could answer.
Your job: choose the threshold from data, and find out how far you can trust the choice.

## The data

[`gate_scores.json`](gate_scores.json) holds the real gate probability for 45 questions,
recorded with [`record.py`](record.py) (Jev RAG retrieval, then the gate):

- **30 answerable** questions from the course's two evaluation sets;
- **15 unanswerable** ones, on-topic but not covered by the documents (support email,
  minimum age, roaming charges, upload speed, a free router…). Each was checked
  against the extracted PDF text.

The questions alternate between a **dev** split (23) and a **held-out** split (22).

## Background

- A gate makes two kinds of mistake: a **false refusal** (answerable, blocked) and a
  **false answer** (unanswerable, let through, so the LLM answers from the wrong passages).
- They don't cost the same. Here a false answer costs 5 and a false refusal 1.
- **Tune on dev, report on held-out.** A threshold chosen on the same data you report
  is guaranteed to look good there. The held-out number is the honest one.

## Your tasks

Edit [`starter.py`](starter.py): `errors`, `cost`, `candidate_thresholds`,
`choose_threshold`, `rates`.

## Check and measure

```bash
uv run --all-extras pytest exercises/lab04_gate_calibration
uv run --all-extras python -m exercises.lab04_gate_calibration.measure
```

Reference solution:

| Threshold | Dev errors | Held-out errors |
| --- | --- | --- |
| Course default 0.500 | 0 | 1 false answer |
| Tuned on dev 0.600 | 0 | **1 false answer** |
| Tuned on everything 0.680 | 0 | 0 |

The false answer is *"What are the roaming charges in Italy?"* (P = 0.65): the price list
mentions Italy many times, but only in international-calling country lists. It sits in
held-out, so dev never sees it.

## Discuss

1. "Tuned on everything" has zero errors everywhere. Why is it the one you must *not*
   report?
2. With 15 unanswerable questions, one mistake is a 14% false-answer rate. How many
   would you need before trusting a number like 2%?
3. Set `answer_cost=1` in `measure.py`. What changes, and for which product would that
   be the right choice?
4. Roaming in Italy fooled the gate. Write three more questions you expect to fool it,
   add them to `record.py`, and check.
5. The course routes topics with one Jev question and judges answerability with another.
   Why not one question for both?
