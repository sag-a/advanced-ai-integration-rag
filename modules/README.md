# Lessons

Teach in this order. Each lesson folder has a one-page `README.md`. Lessons 1
and 3–5 also have a `lesson.html` page to teach from (lesson 2 has its study guide): press **P** for present mode, then
← and → to move and Esc to exit. The browser's print dialog saves a PDF with one
section per page.

| Lesson | Teach from | Time |
| --- | --- | --- |
| 1. Embeddings, chunking, semantic search | [c01_embeddings/lesson.html](c01_embeddings/lesson.html), [GUIDE.md](c01_embeddings/GUIDE.md) | 50 min |
| 2. Vector databases: pgvector and Chroma | [c02_vector_databases/study_guide.html](c02_vector_databases/study_guide.html) | 50 min |
| 3. Classic (base) RAG | [c03_classic_rag/lesson.html](c03_classic_rag/lesson.html) | 50 min |
| 4. Graph RAG | [c04_graph_rag/lesson.html](c04_graph_rag/lesson.html) | 50 min |
| 5. Agentic RAG | [c05_agentic_rag/lesson.html](c05_agentic_rag/lesson.html) | 50 min |

Labs for these lessons are in [`exercises/`](../exercises/README.md).

Not built yet: `c06_system1_rag` (semantic cache fast path) and `c07_capstone`
(compare all architectures with `evals`).

Lessons 1–2 contain runnable Python (`python -m modules.c01_embeddings…`).
Lessons 3–5 teach the code in `rag_core/` and `architectures/`, run through `./rag`.
The `instructor_notes/` folders hold measured runs from before the migration; see
their header before quoting numbers.
