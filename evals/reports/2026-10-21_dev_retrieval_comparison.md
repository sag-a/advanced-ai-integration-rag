# RAG evaluation report

Run: `61aed0100b0d492d81ca5f0e89c50b19` · 2026-10-21T02:16:00.665401+03:00 · split **dev** · mode **retrieval-only**.

Dataset `45d95493bc7e07223c16f54d11934ca75dae93ac492f981dafbd80fb99fa3549` · corpus `60701aec817bc2d950cd03d67f709f953f7cda1556bc73a2a8075a56015f9f81` · final top-k 6.

Completed without execution/judge errors: **False**. A successful run does not establish correctness or an architecture improvement.

Scores are means with scored-case counts. Null is not applicable or unavailable; failures and skips remain visible. Retrieval measures canonical labeled source coverage, not all semantically relevant paraphrases. Factual correctness is F1 of judged claim precision and required-fact recall.

| Architecture | Type | Lang | Done / attempted | Recall@k | Precision@k | Factual F1 | Fact recall | Faithfulness | Citation support | Status match | Mean seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agentic_rag | abstain | el | 0 / 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| agentic_rag | factual | el | 0 / 9 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| agentic_rag | followup | el | 0 / 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| agentic_rag | global | el | 0 / 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| agentic_rag | global | en | 0 / 1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| agentic_rag | multihop | el | 0 / 7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| agentic_rag | multihop | en | 0 / 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| agentic_rag | repeated | el | 0 / 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| agentic_rag | version | el | 0 / 3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| classic_rag | abstain | el | 5 / 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.06 |
| classic_rag | factual | el | 9 / 9 | 0.889 (n=9) | 0.148 (n=9) | N/A | N/A | N/A | N/A | N/A | 0.98 |
| classic_rag | followup | el | 0 / 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| classic_rag | global | el | 5 / 5 | 0.450 (n=5) | 0.233 (n=5) | N/A | N/A | N/A | N/A | N/A | 0.06 |
| classic_rag | global | en | 1 / 1 | 0.500 (n=1) | 0.333 (n=1) | N/A | N/A | N/A | N/A | N/A | 0.06 |
| classic_rag | multihop | el | 7 / 7 | 0.429 (n=7) | 0.214 (n=7) | N/A | N/A | N/A | N/A | N/A | 0.07 |
| classic_rag | multihop | en | 2 / 2 | 0.500 (n=2) | 0.250 (n=2) | N/A | N/A | N/A | N/A | N/A | 0.06 |
| classic_rag | repeated | el | 6 / 6 | 1.000 (n=6) | 0.167 (n=6) | N/A | N/A | N/A | N/A | N/A | 0.06 |
| classic_rag | version | el | 3 / 3 | 0.667 (n=3) | 0.278 (n=3) | N/A | N/A | N/A | N/A | N/A | 0.06 |
| graph_rag | abstain | el | 5 / 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.12 |
| graph_rag | factual | el | 9 / 9 | 0.778 (n=9) | 0.130 (n=9) | N/A | N/A | N/A | N/A | N/A | 0.12 |
| graph_rag | followup | el | 0 / 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| graph_rag | global | el | 5 / 5 | 0.200 (n=5) | 0.100 (n=5) | N/A | N/A | N/A | N/A | N/A | 0.12 |
| graph_rag | global | en | 1 / 1 | 0.250 (n=1) | 0.167 (n=1) | N/A | N/A | N/A | N/A | N/A | 0.12 |
| graph_rag | multihop | el | 7 / 7 | 0.333 (n=7) | 0.167 (n=7) | N/A | N/A | N/A | N/A | N/A | 0.13 |
| graph_rag | multihop | en | 2 / 2 | 0.333 (n=2) | 0.167 (n=2) | N/A | N/A | N/A | N/A | N/A | 0.13 |
| graph_rag | repeated | el | 6 / 6 | 0.500 (n=6) | 0.083 (n=6) | N/A | N/A | N/A | N/A | N/A | 0.12 |
| graph_rag | version | el | 3 / 3 | 0.667 (n=3) | 0.278 (n=3) | N/A | N/A | N/A | N/A | N/A | 0.12 |

## Failures, skips and upload warnings

- classic_rag / followup-01: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- classic_rag / followup-02: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- classic_rag / followup-03: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- classic_rag / followup-04: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- classic_rag / followup-05: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- classic_rag / followup-06: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- graph_rag / followup-01: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- graph_rag / followup-02: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- graph_rag / followup-03: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- graph_rag / followup-04: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- graph_rag / followup-05: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- graph_rag / followup-06: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / factual-01: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / factual-02: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / factual-03: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / factual-04: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / factual-05: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / factual-06: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / factual-07: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / factual-08: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / factual-09: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / followup-01: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / followup-02: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / followup-03: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / followup-04: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / followup-05: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / followup-06: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / multihop-01: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / multihop-02: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / multihop-03: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / multihop-04: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / multihop-05: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / multihop-06: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / multihop-07: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / multihop-08: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / multihop-09: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / global-01: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / global-02: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / global-03: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / global-04: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / global-05: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / global-06: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / version-01: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / version-02: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / version-03: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / abstain-01: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / abstain-02: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / abstain-03: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / abstain-04: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / abstain-05: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / repeated-01: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / repeated-02: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / repeated-03: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / repeated-04: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / repeated-05: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.
- agentic_rag / repeated-06: Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.

Raw answers, source passages, labels, stage timing and available token usage are in the paired JSON. Monetary cost is unreported because no verified price table was supplied. Human review is still needed for judge disagreements and language/conversation quality.
