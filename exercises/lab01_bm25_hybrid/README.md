# Lab 1: BM25 from scratch, then hybrid search

After lesson 2 · about 40 minutes · no API key

## The problem

The COSMOTE price list has seven programs called "Double Play 24 XL". Vector search
finds *a* 24 XL plan, but rarely the one with "500 free minutes": to an embedding,
420 and 500 look almost the same. Keyword search treats "500" as an exact term. The
course's `hybrid_rag` uses TF-IDF for that; your job is to write the stronger
standard, **BM25**, and fuse it with dense search.

## Background

BM25 scores a document for a query by summing, over the query terms it contains:

```text
idf(t) × tf × (k1 + 1) / (tf + k1 × (1 − b + b × len(d) / avg_len))
idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
```

- **tf saturates.** With `k1 = 1.5`, the tenth occurrence of a term adds far less than
  the first. TF-IDF keeps growing linearly; that is the main difference.
- **Long documents are normalized.** `b = 0.75` divides by relative length, so a term
  in a short passage counts more. `b = 0` switches this off.
- **Rare terms weigh more** through `idf`. This version never goes negative.

**Reciprocal rank fusion (RRF)** combines rankings that have incomparable scores (a
cosine distance and a BM25 score). Each item gets `1 / (k + rank)` from every ranking
it appears in, with `k = 60`. Only positions matter. One detail worth noticing: an
item ranked 1st and 3rd (1/61 + 1/63 = 0.032267) narrowly beats one ranked 2nd twice
(2/62 = 0.032258).

## Your tasks

Edit [`starter.py`](starter.py). The five TODOs:

1. `tokenize`: lowercase, remove accents, split into tokens.
2. `BM25.__init__`: term counts, lengths, average length, IDF.
3. `BM25.score`: the formula above.
4. `BM25.top`: best indexes first, without zero scores.
5. `rrf`: fuse rankings.

## Check your work

```bash
uv run --all-extras pytest exercises/lab01_bm25_hybrid            # your starter
LAB_TARGET=solution uv run --all-extras pytest exercises/lab01_bm25_hybrid
```

## Measure it

With Chroma running and both collections prepared (see `SETUP.md`):

```bash
uv run --all-extras python -m exercises.lab01_bm25_hybrid.measure
```

Reference solution, recall@3 on the answerable cases:

| Corpus | Dense | BM25 alone | BM25 + dense (RRF) | Course's TF-IDF hybrid |
| --- | ---: | ---: | ---: | ---: |
| Tariffs (13 cases) | 0.564 | 0.641 | **0.833** | 0.731 |
| Magenta (17 cases) | **0.922** | 0.373 | 0.833 | — |

## Discuss

1. Why does BM25 beat TF-IDF in the hybrid on the tariffs corpus?
2. On the Greek PDF, BM25 alone scores 0.373 and drags the hybrid below dense. Look at
   `tokenize("Καρδιές")` and `tokenize("Καρδιών")`. What would fix this, and what
   would it cost? (Hint: stemming.)
3. Try `k1 = 0.5` and `b = 0` in `measure.py`. What changes, and why?
4. When would you *not* add keyword search to a RAG system?
