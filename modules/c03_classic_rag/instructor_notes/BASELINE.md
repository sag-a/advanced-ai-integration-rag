> **Historical record, kept for instructor reference.** Measured on 19–20 October 2026
> with the retired `base_rag_architecture` package, before the migration to `rag_core` and
> `./rag`, and on an earlier PDF (SHA-256 `a2e492…`; the current `data/corpus` file is
> `b1ee7f…`). Commands and paths below no longer run. Use `./rag` and
> `python -m evals.run` instead. The observations, failures and teaching examples still apply.

# Initial Magenta Moments baseline — 19 October 2026

These are observations on 19 authored development cases, not a claim of general
retrieval quality or production readiness. An independent Greek-speaking
instructor should review the cases and answers before class.

## Corpus and execution

- PDF SHA-256: `a2e492bd4a7e31b6aaba418ed25c2ef600af2862bbb5a5b3abf686f80ee52bc2`.
- Five physical pages, visually checked against extraction. No OCR was needed.
- 30 passages, actual Nomic token counts at most 287 of the 512-token ceiling.
- `magenta_moments_pdf_v1`, cosine metric, the Chapter 2 configured Nomic adapter.
- Nomic v2 MoE weights/code and preprocessing remain pinned to Chapter 1's contract.
- `top-k=6`; Anthropic `claude-haiku-4-5-20251001` for rewriting and generation.
- Preparation was repeated without duplicate records. Existing telecom/course
  collection names and counts were preserved.

## Retrieval observations

[Recorded retrieval report](evaluation/retrieval.json) includes the exact model
contract, chunking settings, prompts, expected evidence, distances, and timings.

| Language | Cases | Answerable cases | Expected-evidence hit rate @6 | Mean evidence coverage |
| --- | ---: | ---: | ---: | ---: |
| Greek | 15 | 11 | 1.00 | 1.00 |
| English against Greek PDF | 4 | 3 | 1.00 | 1.00 |

The two-page case retrieved both clause 2.5/page 1 and clause 3.3/page 2.
Retrieval-only follow-ups use authored standalone questions; they do not measure
automatic rewriting. Unsupported and ambiguous cases are not counted as
retrieval successes merely because nearest neighbors were returned.

## Answer and conversation observations

[Recorded answer report](evaluation/answers.json) contains generated answers,
validated source maps, token usage, timings, and source-based review notes.

- Expected status matched in **15/15 Greek** and **4/4 English** cases.
- All answered cases cite the expected supporting evidence; no unknown citation
  references were accepted. The ambiguous case asks which subject is intended.
- Source-based review checked required facts, material grounding, language, and
  task completion separately from retrieval. Review was by the implementing
  assistant, not an independent reviewer. Minor Greek grammar/phrasing issues
  remain, especially in the rejoining answer's supplementary eligibility summary.
- A separate live two-command rehearsal restarted the process between a Hearts
  question and “Και μπορώ να τις μεταφέρω;”. SQLite restored the subject, and the
  answer cited clause 3.2. Unit tests separately check isolated threads, explicit
  retries, ambiguous clarification/replies, topic changes, locks, and stale evidence.
- No API price table was configured. Available token usage and timings are
  recorded; monetary cost is deliberately `null`.

The final answer report composes the 19-case run with four focused live refusal
rechecks after controlled refusal rendering was added. Dates and the replaced
case IDs are recorded. Other answer paths were unchanged by that rendering fix.
Historical diagnostic calls are additional to the final report's token totals.

## Failures found and addressed

1. The initial rewrite expanded an ambiguous follow-up into a question about
   both subjects. The rewrite instruction now explicitly requires clarification
   when a pronoun could refer to multiple subjects; revised evaluation asked it.
2. The privacy-policy refusal was rejected by an overly strict output rule.
   Validated references may now support an explanation of insufficient evidence.
   Known inline references are normalized into metadata-backed source footers;
   unknown references remain errors.
3. Some model refusals included unsupported instructions about where to find
   account balances. The displayed insufficient-evidence text is now a fixed
   Greek/English limitation statement. Four affected cases were rechecked live.

The initial run's Greek status checks were 13/15, including one rejected answer;
the English status checks were 4/4. These failures are retained in the final
report rather than relabeled as successes.

## Verification and limits

Focused tests, Ruff, package resource/import checks outside the source tree,
live Chroma preparation/retrieval, a real Anthropic evaluation, and restart
memory checks were executed. No project-wide application suite or separate
static type checker was run. No course demonstration source files were modified.

The small development set does not establish a broader quality threshold.
Prompt defenses and deterministic tests do not prove universal resistance to
malicious document instructions. Local CPU inference has no hard cancellation
boundary, and per-thread file locking currently targets macOS/Linux. This is
a single-user terminal lesson; multi-user authorization, OCR, APIs, streaming,
hybrid retrieval, reranking, and live account tools are outside its scope.
