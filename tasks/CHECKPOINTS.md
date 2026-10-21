> Current scope (2026-10-21): the instructor removed the six synthetic PDFs and
> their generation tooling. Use only `data/corpus/`, `magenta_course_pdf_v1` and
> the existing original-PDF graph. Evaluation now reuses ten Magenta-only
> development scenarios. The expanded-corpus review and Step 4 gate below are
> superseded; the remaining content is historical, not a request to recreate it.

# Measured migration checkpoints - 2026-10-21

These are functional smoke checks on the original five-page, 30-passage PDF,
not an evaluation baseline and not evidence of an architecture improvement.

## Shared core and application

- macOS Apple Silicon, Python 3.12: dependency installation succeeded.
- 96 focused tests passed; Ruff and wheel/import checks passed.
- One SQLite checkpoint stores architecture-prefixed conversation IDs. Greek
  follow-ups survived both separate CLI processes and a Streamlit server restart.
- `./rag ask` answered the same Greek expiry question through all three plugins
  and cited original clause 3.3 on physical page 2.
- The UI supports Greek/English, explicit retry, sources, saved paths/research and
  trace links. Reload did not resubmit completed questions.
- Linux and Windows runtime checks remain pending, as agreed with the instructor.

## Prepared graph

- 77 canonical entities, 115 directed relationship facts.
- Index fingerprint: `5d5fa8f234ff64b390b61d066cfd4312de5ad84861f9fe094916d13c5f00ded6`.
- Completed extraction run recorded 119,506 input and 27,681 output tokens.
  Initial failed-run usage is not included or fully available. A separate
  diagnostic call used 1,697 input and 498 output tokens. No price is inferred.
- The first attempt failed on undeclared relationship endpoints. Schema guidance
  and actionable validation feedback fixed the observed failure. Schema identity
  is now part of the index contract. Future exhausted attempts retain diagnostics.
- The graph added three final context passages beyond the vector seeds in the
  CLI smoke test. That alone does not prove those passages were useful.
- One reviewed positive statement names items outside its selected quotation
  lines, although the full original passage contains them. Exact substring
  matching is a provenance check, not semantic entailment. Review extraction
  support before treating graph statements as authoritative.

## Agentic behavior

- Simple expiry query: one selected passage, five logical research model calls
  in the CLI smoke test; the browser run used six. SDK HTTP retries are separate.
- A broad joining/Hearts/coupon question exhausted the search budget, including
  one explicit retry. No final assistant answer was accepted. The failed turn
  remains saved as one unfinished human submission.
- A simpler Greek query and a subsequent transfer follow-up succeeded. The
  accepted research timeline and citations survived restart.

## Cloud verification

- Classic trace: `d1c05aa5899ab1f65cc57ba03b219a5f`.
- Post-restart Agentic trace: `2b827965b652ac43a5f7e5146145b244`.
- Readback confirmed node/tool observations. Observation counts are not provider
  call counts; logical calls and available token usage are recorded separately.
- New local credentials are in ignored `.env`; the tracked template is empty.
  Original credential files and existing port 8002 collections were unchanged.

Detailed captures are local, ignored files under `outputs/checkpoints/`. The new
baseline will use the reviewed expanded dataset, not these smoke results.

## Step 4 draft corpus and evaluation (in progress)

- Added six explicitly fictional AsteriaLab PDFs alongside an unchanged original.
  The isolated expanded collection `magenta_corpus_v1` has 174 passages across
  seven PDFs; the maximum measured prefixed token length is 287. Original demo
  collections and the default app corpus remain unchanged.
- The fixture PDFs reproduce byte-for-byte. First/last pages were rendered and
  visually checked; Greek extraction and all expected evidence labels resolve.
- Authored 73 typed cases: 44 development and 29 held-out. The instructor will
  review held-out wording, expected facts and physical-page/clause labels. Review
  is still pending; held-out runs are blocked before service/credential access.
- The initial Classic no-model development retrieval run completed 38 cases and
  skipped six history cases. Canonical-source recall was 0.889 on nine factual
  cases, 0.429 on seven Greek multi-hop cases and 0.500 on two English multi-hop
  cases. These are draft retrieval scores, not answer-quality results.
- Six-case Greek generation smoke checks and one English multi-hop case ran for
  Classic and Agentic. On the Greek multi-hop example, Classic retrieved one of
  three labeled passages and refused; Agentic retrieved all three and answered.
  Agentic took more logical calls and time. This is illustrative, not a general
  performance claim. Both failed the expected clarification on `followup-05`.
- The first judge calibration confused a fixed refusal with factual claims and
  copied reference facts as if stated in the answer. The fixed application refusal
  now gets deterministic empty-claim scoring, and every non-missing fact verdict
  needs an exact answer quote. Invalid labels get one bounded retry; both attempts
  and usage are retained. Invalid judgments are not scored. Prior smoke reports
  are retained and their Cloud protocol marked superseded.
- Langfuse API readback confirmed initial smoke traces and computed scores. Later
  score batches remain queued until separately checked; queue flush is not readback.
- The expanded UI answered Greek and a follow-up, then English, and restored all
  three turns after a server restart. The English Gold answer omitted the 90-day
  qualification despite a valid citation: UI success and citation membership do
  not establish completeness. This manual UI check is not a held-out baseline.
- English refusals now use the dominant script rather than treating any Greek
  proper name as a Greek-language question. Very short mixed-language queries
  remain heuristic.
- Latest local checks: 118 tests passed and Ruff passed. Wheel/import checks passed
  during development; a final wheel check follows the remaining index/report work.
- Expanded graph extraction is still running in a separate output path. No
  Step 4 baseline or Step 5 completion is claimed.

## Step 4 final development checkpoint

- Expanded graph completed: 329 entities, 1,119 relationships, 174 passages.
  Fingerprint `e485bac5f49f2bccc3e07f92762c1ac4fdd764908dca3a07828aba194f5bbddc`.
  Recorded indexing usage: 1,649,136 input / 210,301 output tokens. Four diagnostic
  attempt records lack usage; this does not mean four distinct provider calls.
  A late incomplete extraction recovered through bounded retries. Repeating
  preparation with the placeholder env template reused the completed index.
- Representative extraction audit retained the V1 transfer prohibition. Some
  V2 relationship summaries/quotations omit conditions, so the index is a draft
  navigation structure, not independently verified facts. Audit capture is in
  `outputs/checkpoints/expanded-extraction-audit.json`.
- Local Graph source recall regressed against Classic in the development
  retrieval comparison. Seven matching generation smoke cases per architecture
  are combined in `evals/reports/2026-10-21_dev_architecture_smoke.json`.
- Cloud API readback matched all computed metrics for the 21 current smoke trace
  groups. The JSON records source run IDs to distinguish calibrated scores.
- Live timing exposed server-side buffering of structured tool arguments. The
  stream now sets Anthropic's per-tool `eager_input_streaming=True`; final JSON
  and citations are still validated before persistence. A gated-node regression
  test proves early LangGraph delivery, and an adapter test verifies the field.
  Before fixing, draft events arrived in a burst. After fixing, 115 draft updates
  spanned ~2.97 seconds before final completion in the real timing probe.
- Browser verification captured an actual partial draft with its label, followed
  by a validated answer with original citations. Graph details show three added
  final passages and survive saved-state restoration. Both local teaching servers
  were restarted with the fix; prior conversations were retained.
- 120 tests passed, Ruff passed, final wheel built. Linux/Windows remain pending.
- Held-out review, its first frozen baseline, and Step 5 are still outstanding.

No separate static type checker is configured, so none ran. Browser console
inspection also recorded the expected WebSocket/health errors during deliberate
server shutdown, plus a nonfatal upstream Graphviz worker-fallback warning; the
chart rendered successfully. The held-out review file remains unapproved.
