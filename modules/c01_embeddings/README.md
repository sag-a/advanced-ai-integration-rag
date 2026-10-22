# Lesson 1 — representations, chunking, and context

Suggested teaching time: 50 minutes. Focus on the output and concepts; participants
do not need to study every implementation detail.

| Order | Module | What to explain |
| --- | --- | --- |
| 1 | `m01_nomic_embeddings` | Sentences become vectors; compare their similarity. |
| 2 | `m02_contextual_words` | The same token changes its representation with context. |
| 3 | `m03_chunking_comparison` | Fixed, recursive, and section boundaries produce different chunks. |
| 4 | `m04_nomic_chunking` | Encode the resulting chunks. |
| 5 | `m05_nomic_semantic_search` | Embed a question and retrieve the closest passages. |

- [Full chapter guide and run commands](GUIDE.md)
- [Contextual-word explanation and current v2 plots](contextual_embeddings/README.md)
- [Source modules](./)

The shared Nomic loader lives in this lesson's `local_embeddings.py` and is also
used by lesson 2. Lessons 1–2 use the fictional telecom text; lessons 3–5 use the
Greek Magenta Moments PDF in `data/corpus/` with the same pinned encoder.

Next: [Lesson 2](../c02_vector_databases/study_guide.html).
