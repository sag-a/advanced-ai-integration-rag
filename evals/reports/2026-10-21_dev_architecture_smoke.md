# RAG evaluation report

Run: `f3dcf0ef112a4e56b2d1b569ee82e64c` · 2026-10-21T02:22:32.486503+03:00 · split **dev** · mode **full**.

Dataset `45d95493bc7e07223c16f54d11934ca75dae93ac492f981dafbd80fb99fa3549` · corpus `60701aec817bc2d950cd03d67f709f953f7cda1556bc73a2a8075a56015f9f81` · final top-k 6.

Completed without execution/judge errors: **True**. A successful run does not establish correctness or an architecture improvement.

Scores are means with scored-case counts. Null is not applicable or unavailable; failures and skips remain visible. Retrieval measures canonical labeled source coverage, not all semantically relevant paraphrases. Factual correctness is F1 of judged claim precision and required-fact recall.

| Architecture | Type | Lang | Done / attempted | Recall@k | Precision@k | Factual F1 | Fact recall | Faithfulness | Citation support | Status match | Mean seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agentic_rag | abstain | el | 1 / 1 | N/A | N/A | N/A | N/A | N/A | N/A | 1.000 (n=1) | 5.29 |
| agentic_rag | factual | el | 1 / 1 | 1.000 (n=1) | 0.500 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 24.96 |
| agentic_rag | followup | el | 1 / 1 | N/A | N/A | N/A | N/A | 1.000 (n=1) | 1.000 (n=1) | 0.000 (n=1) | 14.11 |
| agentic_rag | global | el | 1 / 1 | 0.667 (n=1) | 1.000 (n=1) | 0.857 (n=1) | 0.750 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 17.80 |
| agentic_rag | multihop | el | 1 / 1 | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 21.56 |
| agentic_rag | multihop | en | 1 / 1 | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 29.34 |
| agentic_rag | version | el | 1 / 1 | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 17.94 |
| classic_rag | abstain | el | 1 / 1 | N/A | N/A | N/A | N/A | N/A | N/A | 1.000 (n=1) | 4.29 |
| classic_rag | factual | el | 1 / 1 | 1.000 (n=1) | 0.167 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 13.95 |
| classic_rag | followup | el | 1 / 1 | N/A | N/A | N/A | N/A | 1.000 (n=1) | 0.833 (n=1) | 0.000 (n=1) | 7.13 |
| classic_rag | global | el | 1 / 1 | 0.667 (n=1) | 0.333 (n=1) | 0.857 (n=1) | 0.750 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 5.57 |
| classic_rag | multihop | el | 1 / 1 | 0.333 (n=1) | 0.167 (n=1) | 0.000 (n=1) | 0.000 (n=1) | N/A | N/A | 0.000 (n=1) | 6.39 |
| classic_rag | multihop | en | 1 / 1 | 0.667 (n=1) | 0.333 (n=1) | 0.000 (n=1) | 0.000 (n=1) | N/A | N/A | 0.000 (n=1) | 13.04 |
| classic_rag | version | el | 1 / 1 | 1.000 (n=1) | 0.333 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 0.833 (n=1) | 1.000 (n=1) | 5.10 |
| graph_rag | abstain | el | 1 / 1 | N/A | N/A | N/A | N/A | N/A | N/A | 1.000 (n=1) | 4.73 |
| graph_rag | factual | el | 1 / 1 | 1.000 (n=1) | 0.167 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 13.87 |
| graph_rag | followup | el | 1 / 1 | N/A | N/A | N/A | N/A | 1.000 (n=1) | 0.667 (n=1) | 0.000 (n=1) | 5.31 |
| graph_rag | global | el | 1 / 1 | 0.333 (n=1) | 0.167 (n=1) | 0.857 (n=1) | 0.750 (n=1) | 1.000 (n=1) | 0.857 (n=1) | 1.000 (n=1) | 7.15 |
| graph_rag | multihop | el | 1 / 1 | 0.333 (n=1) | 0.167 (n=1) | 0.000 (n=1) | 0.000 (n=1) | N/A | N/A | 0.000 (n=1) | 7.20 |
| graph_rag | multihop | en | 1 / 1 | 0.333 (n=1) | 0.167 (n=1) | 0.000 (n=1) | 0.000 (n=1) | N/A | N/A | 0.000 (n=1) | 4.62 |
| graph_rag | version | el | 1 / 1 | 1.000 (n=1) | 0.333 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 1.000 (n=1) | 5.37 |

## Failures, skips and upload warnings

None recorded.

Raw answers, source passages, labels, stage timing and available token usage are in the paired JSON. Monetary cost is unreported because no verified price table was supplied. Human review is still needed for judge disagreements and language/conversation quality.
