> Current scope (2026-10-21): the instructor removed the six synthetic PDFs and
> their generation tooling. Use only `data/corpus/`, `magenta_course_pdf_v1` and
> the existing original-PDF graph. Evaluation now reuses ten Magenta-only
> development scenarios. The expanded-corpus review and Step 4 gate below are
> superseded; the remaining content is historical, not a request to recreate it.

# Migration progress

Working root: `python/rag-course/`. Preserve unrelated GenAI lessons and source
material until the corresponding replacement checkpoint passes.

- [x] 1. Installable scaffold, safe template, macOS smoke/lint/build checks.
  - [ ] Linux and Windows checks (CI matrix; not locally verified).
- [x] 2. Shared multi-document core, one checkpoint, Classic RAG, focused tests,
  Greek first-turn and restart/follow-up check.
- [x] 3. Registry, Graph and Agentic ports, generic CLI/UI, real smoke checks.
- [ ] 4. Synthetic corpus, draft typed/split dataset, metrics with N/A handling.
  - [ ] Instructor review of held-out Greek questions and facts before freezing.
  - [ ] New measured baseline; ties are acceptable evidence, not a reason to tune tests.
- [ ] 5. System1 semantic cache, Graph global search, seven course modules.
- [ ] Final cleanup only after replacement checks pass; no credential or user-data loss.

## Accepted decisions

- New standalone project inside `python/rag-course/`; no replacement of unrelated
  `python/pyproject.toml` dependencies.
- Synthetic documents are authorized. They must not be presented as real Magenta
  terms, live offers or account information.
- The instructor will review the held-out Greek dataset.
- Langfuse remains opt-in. Chroma uses a new isolated volume/port by default.
- Continue after macOS checkpoints, with Linux/Windows runtime checks explicitly
  pending (instructor approved 2026-10-21).

## Verification: 2026-10-21

- macOS Apple Silicon / Python 3.12: `uv sync --all-extras` succeeded.
- `uv run --all-extras pytest -q`: 8 passed.
- `uv run --all-extras ruff check .`: passed.
- Wheel build and isolated imports: passed; no env files in the wheel.
- Dependency resolution initially found NumPy's Python 3.12 requirement; the
  project now explicitly supports Python 3.12–3.13.
- Linux/Windows runtime checks have not run. Their matrix is configured in
  `.github/workflows/test.yml` for the standalone project.
- Step 2: 57 tests passed, Ruff passed, wheel imports passed. The isolated Chroma
  service on port 8004 contains 30 prepared passages. A real Greek first question
  and follow-up ran in separate processes; SQLite restored the subject and the
  follow-up cited physical page 2, clause 3.3. Captured local results are in
  ignored `outputs/checkpoints/core-smoke.jsonl`.
- `rag_core` has no old-package imports, legacy tuple adapter or `fcntl` usage.
- Step 3: 96 tests passed, Ruff and wheel build passed. Plugin discovery and all
  three bundled skills work from the wheel. Real CLI expiry questions passed for
  all three architectures. Classic and Graph selected six original passages;
  Agentic selected one and used five research model calls in its CLI smoke test.
- The unified browser picker, Greek follow-up, English answer, graph expansion,
  refresh and server restart were verified. A post-restart Agentic follow-up
  restored its subject and saved its research details plus Langfuse trace link.
- Langfuse initialization initially failed because cached Settings contained a
  dict. The redundant wrapper cache was removed, a regression test was added,
  and Cloud readback confirmed graph nodes and research tools.
- A broad three-topic Agentic question exhausted its search budget twice (one
  explicit retry). It saved no assistant answer and did not duplicate the human
  submission. This is a recorded limitation, not a passing quality result.
- Graph quotation audit found an exact but incomplete quotation for one extracted
  positive statement. Answer generation uses original passages, not extracted
  statements. Semantic extraction support still needs review.

## Step 4 current checkpoint

- [x] Six labeled fictional PDFs, deterministic generation, 174-passage collection.
- [x] 73 typed/split draft cases and a readable 29-case review sheet.
- [x] Remove named test examples and clause hints from shared/research prompts.
- [x] Registry-driven evaluation, separate retrieval/generation metrics, nulls,
  bounded judge validation retries, per-stage usage and Cloud score metadata.
- [x] Classic/Agentic Greek and English development smoke reports; record failures.
- [x] Expanded graph built, representative facts audited, retrieval compared; regressions recorded.
- [ ] Instructor review, freeze and first complete held-out comparison.
- [ ] Step 5 starts only after that checkpoint. No architecture win is assumed.

The CI workflow under `.github/` is a standalone-project template. While this
project remains nested in the original repository, GitHub does not automatically
discover that nested workflow; Linux/Windows execution is still pending.
