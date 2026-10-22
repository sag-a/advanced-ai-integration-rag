# Lab 3: A citation validator that also checks the numbers

After lesson 3 · about 30 minutes · offline, no API key, no Docker

## The problem

Every answer in this course cites numbered passages, and the application checks those
numbers before showing the answer (`rag_core/answers.py`, `format_answer`). Two things
are wrong with that check:

1. **A bug.** It reads every `[number]` as a citation, so an answer mentioning
   "[2024]" cites "source 2024", fails, and the whole turn is rejected.
2. **A limit.** It checks *membership*: the cited passage exists. It does not check
   *support*. In a measured run, the model cited the right clause (§3.3) but wrote that
   Hearts expire after "exactly 12 months"; the clause says the 1st day of the 13th month.

You will write a stricter validator that fixes the bug, and a cheap grounding check that
would have caught the "12 months" answer.

## Background

- **Membership vs support.** A valid citation proves the source exists and was given to
  the model, not that it says what the answer claims. Full support needs a judge model
  (Ragas `faithfulness` in `evals/`). Cheap checks catch some errors for free.
- **Numbers are high-risk claims.** Prices, durations and counts are what users act on,
  and they are easy to compare: "46.05 €" in the answer must appear in a cited passage as
  "46,05". A clock time like "12:00" is one token, so it can't vouch for "12 months".
- **The check matches numbers, not meanings.** "12" would pass next to an unrelated "12";
  it's a filter, not a proof.

## Your tasks

Edit [`starter.py`](starter.py):

1. `CITATION`: a pattern for `[3]` and `[1, 3]` with one or two digits, so `[2024]` isn't one.
2. `inline_citations`: the cited numbers, in order, without repeats.
3. `validate`: accept or reject with a reason (empty, duplicate, unknown, at least one,
   clarification).
4. `strip_citations`: remove markers and the space before them; keep `[2024]`.
5. `unsupported_numbers`: numbers in the answer that no cited passage contains.

## Check your work

```bash
uv run --all-extras pytest exercises/lab03_citation_validator
LAB_TARGET=solution uv run --all-extras pytest exercises/lab03_citation_validator
```

The tests include the "[2024]" case and the "exactly 12 months" case from the course's
own measurements.

## Discuss

1. Your validator accepts "[2024]" as text. What if a document really has 100 passages?
   Which rule would you change, and what would break?
2. Should an answer with an unsupported number be rejected, regenerated, or shown with a
   warning? Who decides?
3. `unsupported_numbers` passes "the plan includes 24 Mbps" if "24" appears anywhere in
   the cited passage. Write a case it gets wrong. How would you tighten it?
4. Port your `validate` into `rag_core/answers.py`. Which existing tests change, and why?
