# Course source document

The course uses only `corpus/Magenta_Moments_Terms_Conditions_gr.pdf`, the supplied
unchanged five-page Greek PDF. SHA-256:
`a2e492bd4a7e31b6aaba418ed25c2ef600af2862bbb5a5b3abf686f80ee52bc2`.

The six synthetic teaching PDFs, their generator, source JSON and font assets
have been removed. There is no expanded or draft corpus.

The existing default collection `magenta_course_pdf_v1` contains 30 passages from
this original document. The existing default graph index is
`outputs/graph_rag/graph.json`. Ordinary use needs no re-preparation:

```bash
./rag check
./rag ui
```

Evaluation reuses the ten existing Magenta-only classroom scenarios in the
current JSONL schema. They are development examples, not an unseen benchmark.
Historical reports from the removed synthetic corpus are retained for reference
and must not be compared as if they used this dataset.
