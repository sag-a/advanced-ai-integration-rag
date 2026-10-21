> **Historical record, kept for instructor reference.** Measured on 19–20 October 2026
> with the retired `agentic_rag_architectures` package, before the migration to `rag_core` and
> `./rag`, and on an earlier PDF (SHA-256 `a2e492…`; the current `data/corpus` file is
> `b1ee7f…`). Commands and paths below no longer run. Use `./rag` and
> `python -m evals.run` instead. The observations, failures and teaching examples still apply.

# Agentic RAG verification — 20 October 2026

The implementation uses Deep Agents 0.7.19 with the same Magenta Moments PDF, Anthropic answer model, local embeddings, Chroma collection, answer prompt and 10-case evaluation dataset. It adds bounded skill-guided research before the shared generation step. No Chroma records or existing conversations were replaced.

## Checks

- 116 focused tests passed across agentic, baseline, Graph RAG, streaming, Streamlit and shared evaluation.
- Ruff, launcher syntax/routing, lock consistency, wheel imports and all three packaged SKILL.md resources passed.
- The full unrelated repository suite and a separate static type checker were not run.
- Live Greek question, Greek follow-up and English coupon question worked in Streamlit. A server restart restored messages, citations, trace links and research activity.
- A live multi-topic CLI question read membership/Hearts/coupons skills, delegated to the constrained PDF researcher, used five searches and three clause reads, and selected five original passages. It used eight research model calls and one delegation.
- Cloud observation readback of trace `42cbe9bd5f3576d83015be03fd0786b3` confirmed architecture `agentic_rag`, nested task execution, four skill reads, and nine model generations (research plus final answer).

## Initial failure and correction

The initial run in `outputs/evaluation/agentic-initial-20261020.json` retrieved all labeled evidence for seven answerable cases but recorded two application failures. For unsupported account/privacy-policy questions, the model ended with a refusal without calling select_evidence. A missing selection correctly failed closed rather than turning arbitrary model text into an accepted answer.

The correction permits one bounded finalization reminder within the original shared budgets. The model must explicitly select existing IDs or select [] to abstain. Missing selection after that, service failures, and budget violations still fail visibly. A deterministic test covers early refusal followed by an explicit empty selection.

## Corrected run

Command: `./evaluate run --architecture agentic_rag --langfuse --output outputs/evaluation/agentic-final-20261020.json`.
Dataset SHA-256: `7189a7811c4f950f8dbd0e988678e14bc9c52079be99adb8b99bf6889cad59e1`.

| Metric | Result | Scored cases |
|---|---:|---:|
| recall_at_k | 1.000 | 7/10 |
| precision_at_k | 0.738 | 7/10 |
| reciprocal_rank | 0.929 | 7/10 |
| status_accuracy | 1.000 | 10/10 |
| factual_correctness | 1.000 | 10/10 |
| factual_recall | 1.000 | 10/10 |
| citation_validity | 1.000 | 7/10 |
| faithfulness | 0.889 | 9/10 |
| citation_support | 0.778 | 9/10 |

The corrected report is complete with no application or judge errors. Mean application latency was 14.31s in this single run; research added 47 logical model calls across the ten scenarios. The ambiguous case skipped retrieval. Actual selected evidence was one to three passages for answerable cases, below the maximum budget of six. Token usage is recorded separately for research, final generation, rewriting and judging; no monetary price was calculated.

Greek and English separately had expected-status accuracy, labeled evidence recall and judge factual correctness 1.0. This small authored dataset already had a strong baseline; the run does not demonstrate a general quality improvement. Agentic research adds model usage and latency; the baseline may be preferable for simple questions.

## Scoring limitation requiring review

The account-balance abstention has no selected passages or citations. Its generic insufficient-evidence statement was labeled factually correct but ungrounded and uncited by the judge. The Greek privacy-policy abstention was labeled correct and grounded but uncited. These cases reduce aggregate faithfulness to 0.889 and citation support to 0.778; all seven answered cases scored 1.0 on both. They expose ambiguity/non-applicability in grading abstentions against empty evidence, rather than an observed wrong account balance or fabricated privacy-policy detail. The raw scores were preserved. Review applicability rules before using the aggregate as a quality ranking.

Cloud score readback is saved at `outputs/evaluation/agentic-cloud-verification-20261020.json`. Detailed generated reports and SQLite files remain under ignored outputs/; the source package includes the reproducible lesson and tests.
