# Held-out review — draft v1

Reviewer: course instructor. Review status is recorded in `review.json`.

Check Greek/English wording, expected status, every material qualification and the canonical source map against physical PDF pages. Cases refer only to the supplied original Magenta Moments PDF. Flag ambiguous wording rather than guessing an intended answer.

Dataset identity: `d19dd28b52f7c6e0cf13875cc8e359b51394dfb11713c14fc766507bf192812b`.

After corrections to `cases.jsonl`, run `uv run --all-extras python -m evals.dataset refresh`. This regenerates this sheet and invalidates any approval for changed labels. After your human review, set `review.json` status to `approved`, with the matching dataset identity, your name in `reviewed_by` and ISO date in `reviewed_on`. Do not approve unreviewed cases.

No held-out cases are configured. The current cases are classroom development examples; no held-out baseline or approval is claimed.

