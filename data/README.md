# Course source documents

The lessons use `corpus/Magenta_Moments_Terms_Conditions_gr.pdf`, the supplied
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

Evaluation uses the 21 Magenta cases in `evals/datasets/cases.jsonl`. They are
classroom examples, not an unseen benchmark.

## Harder corpus: COSMOTE residential tariffs (2019)

`corpus-tariffs/cosmote_residential_tariffs_2019.pdf` is OTE's English price list
*Discount calling packages for residential customers*, dated 1/10/2019: 52 pages,
about 120 numbered programs, 156 passages. SHA-256:
`7a6ca7e4d7814fbf77425045f76d13a8c40647596ff52528da8a48bc4ccc9bfe`.

- **The prices are from 2019.** Present it as a historical price list, never as
  current offers.
- **Pages 1–3 were the table of contents.** They now carry one line, "Table of
  contents removed for the course", so program names in the index don't match
  every question. The pages stay in place, so PDF page numbers match the print.
- **Why it's harder:** several programs share a name but differ in terms (three
  "Double Play 24 XL (12 month)" plans, two "Fiberspeed 100 XL" plans with 420
  minutes), and the same boilerplate appears in most programs.

It has its own collection, so the Magenta setup is unaffected:

```bash
./rag prepare --corpus data/corpus-tariffs --collection cosmote_tariffs_v1
./rag ask --corpus data/corpus-tariffs --collection cosmote_tariffs_v1 \
  --architecture hybrid_rag "Which Double Play 24 XL plan has 500 free minutes?"
```

Classic RAG and Hybrid RAG work on it. Graph RAG would need its own index (about
156 extraction calls), and Agentic RAG's skills are written for Magenta Moments.
