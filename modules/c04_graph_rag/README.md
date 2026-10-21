# Lesson 4: Graph RAG

Suggested teaching time: 50 minutes. Teach from [lesson.html](lesson.html); press
**P** to present one section per screen.

## Concept

Before class, Claude extracts entities and quoted relationships from every passage
into `outputs/graph_rag/graph.json`. At question time there is no extra model call:
three vector seeds pick anchor entities, a breadth-first walk of up to two hops
collects passages linked to them, and reciprocal rank fusion merges the vector
and graph orders into the final top 6.

Code: `architectures/graph_rag/index.py` (build and contract),
`retrieval.py` (seeds, traversal, fusion), `schema.py`, `templates/extraction.j2`.

## Three commands

```bash
./rag check --architecture graph_rag | head -20                      # index is compatible
./rag ask --process --architecture graph_rag "Μπορώ να εξαργυρώσω τις Καρδιές για μετρητά;"
./rag search --architecture graph_rag "Who can join Magenta Moments and what do members get?"
```

## What to look for

- Anchors, hops and the passages each hop added, in the CLI panels or the UI.
- Negative facts keep their polarity and their exact Greek quote.
- When the walk adds nothing, the retriever falls back to plain vector search.
- Hub entities such as `magenta_moments` can pull in irrelevant passages. Check
  page and clause provenance instead of assuming every connection helps.
- Any non-default `--model` fails with “Graph index is incompatible”, because the
  extraction model is part of the index contract.

## Instructor notes

- [MEASUREMENTS.md](instructor_notes/MEASUREMENTS.md): baseline vs Graph RAG on ten
  scenarios (a tie, not an improvement), extraction cost, and the “exactly 12
  months” human-review example.
