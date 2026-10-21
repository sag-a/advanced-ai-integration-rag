> **Historical record, kept for instructor reference.** Measured on 19–20 October 2026
> with the retired `graph_rag_architecture` package, before the migration to `rag_core` and
> `./rag`, on the same PDF that is in `data/corpus` (SHA-256 `a2e492…`). Commands and paths below no longer run. Use `./rag` and
> `python -m evals.run` instead. The observations, failures and teaching examples still apply.

# Measured classroom run — 20 October 2026

Corpus: `Magenta_Moments_Terms_Conditions_gr.pdf`; PDF SHA-256 `a2e492bd4a7e31b6aaba418ed25c2ef600af2862bbb5a5b3abf686f80ee52bc2`.
Index fingerprint: `f9a9825579afc49788816f89a6a03d6bbbf0561536b3c758e6c42c2e926f8794`.
Extraction: `claude-haiku-4-5-20251001`, temperature 0; 30 passages, 42 entities, 95 facts (23 negative).

Command: `./evaluate architectures --langfuse --output outputs/evaluation/graph-architectures-20261020.json`.
Both variants used the unchanged 10 scenarios (6 Greek, 4 English), final k=6, the same answer/rewrite prompts, model and temperature 0. Judges used the existing shared configuration.

| Metric | Baseline | Graph RAG |
|---|---:|---:|
| Evidence recall (7 answerable cases) | 1.000 | 1.000 |
| Evidence precision (7 answerable cases) | 0.190 | 0.190 |
| Reciprocal rank | 0.929 | 0.929 |
| Judge factual correctness | 1.000 | 1.000 |
| Judge required-fact recall | 1.000 | 1.000 |
| Citation membership validity | 1.000 | 1.000 |
| Expected-status match | 1.000 | 1.000 |

Both runs completed without application/judge errors. Greek and English separately had evidence recall, expected-status match and judge factual correctness 1.0. These authored cases already had a strong baseline; this run demonstrates inspectable expansion, **not a measured quality improvement**.

- base_rag: mean application 5.06s; mean retrieval 1.46s. Generation usage: 35,013 input / 2,296 output tokens (rewriting is recorded separately per case).
- graph_rag: mean application 5.94s; mean retrieval 0.92s. Generation usage: 34,484 input / 2,319 output tokens (rewriting is recorded separately per case).

Timings are one sequential run with local model warm-up and concurrent browser verification, not a controlled performance benchmark. Do not infer a retrieval speed advantage from these averages. No monetary cost was calculated.

Graph expansion selected 3–5 passages beyond the initial vector seeds in each of the nine retrieval turns; the ambiguous case correctly skipped retrieval. The Greek expiry search added seven candidates, traversed the 30-fact cap, and selected three passages beyond its seeds. Hub entities can contribute irrelevant context: inspect page/clause provenance rather than assuming every connection helps.

## Indexing and inspection

Accepted extraction cache reports 109,505 input and 25,739 output tokens, separately from application and judge usage. The final chunk needed format correction because the model encoded arrays as strings. Earlier abandoned/diagnostic calls and failed calls without usage are not fully represented in that total; it is not a complete bill. Cached preparation was repeated successfully without additional inference. The failed verbatim-quotation prototype cache was preserved separately; no completed graph was overwritten.

Manually inspected representative relationships against the original passages:

- §2.4 / page 1: automatic membership termination retains its customer-status condition.
- §3.2 / page 2: inability to transfer/redeem Hearts is preserved as a negative fact with a matching quotation. Another mixed statement combines virtual-currency definition and restrictions under positive polarity; polarity alone is not a sufficient semantic summary.
- §3.3 / page 2: one-year validity, first day of month 13, and 12:00 are present in both fact and quotation.
- §7.3 / page 4: expired coupons are invalid and cannot be used.

A separate English browser answer correctly cited §3.3 but then glossed its calendar rule as “exactly 12 months,” which is too strong. This example is outside the unchanged 10-case dataset. Keep it as a human-review teaching example: valid source membership and perfect small-set judge scores do not prove every generated claim is correct. The shared generation prompt was preserved for comparison.

## Browser, persistence and cloud

Verified Greek question, contextual follow-up and English question; refreshed and restarted Streamlit on loopback port 8502. All accepted messages, citations, trace links and historical graph explanations restored without automatic resubmission. Architecture isolation, explicit retries, stale-evidence prevention and draft handling also have deterministic tests.

Cloud observation readback for trace `98096059e83534317f96492e8545c52a` confirmed `graph-vector-seeds`, `graph-traversal`, `graph-passage-selection`, the three graph nodes and nested `ChatAnthropic`. Architecture metadata was `graph_rag`. Local reports retain per-case graph counts, usage, judge usage and trace links. Cloud score readback is recorded separately at `outputs/evaluation/graph-architectures-cloud-verification-20261020.json`.

Generated indexes, SQLite files and detailed evaluation reports stay under ignored `outputs/`; fresh checkouts must explicitly prepare their own compatible graph.

Cloud score readback verified all expected calculated scores on **20/20 application
traces** across the two architectures. No judge failures or score mismatches were
observed in this run. A separate live browser turn visibly showed
“Draft — checking sources” while Anthropic generation was still running, before
validated citations replaced the draft.

Final deterministic verification: **96 focused tests passed** across graph preparation,
retrieval, shared backend, streaming, Streamlit and evaluation. Ruff passed; launchers
passed shell syntax/routing checks; lock consistency and built-wheel imports passed.
The full unrelated repository test suite and a separate type checker were not run.
