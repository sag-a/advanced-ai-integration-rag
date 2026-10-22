# Lesson 1: Embeddings — five demonstrations in teaching order

Use these local demonstrations in sequence. Participants predict a result,
inspect the output, and explain what they observed; reading the Python
implementation is not required. No provider API key is needed.

The `m01`–`m05` prefixes keep the teaching order visible in the file browser
and remain valid Python module names. The former unnumbered commands have
been replaced by the commands below.

| Order | Module | Question to answer |
|---|---|---|
| 1 | [m01_nomic_embeddings.py](m01_nomic_embeddings.py) | Can differently worded sentences have similar vectors? |
| 2 | [m02_contextual_words.py](m02_contextual_words.py) | Does a word's representation change with its context? |
| 3 | [m03_chunking_comparison.py](m03_chunking_comparison.py) | Where should we cut a document without losing useful context? |
| 4 | [m04_nomic_chunking.py](m04_nomic_chunking.py) | How do the resulting chunks become vectors? |
| 5 | [m05_nomic_semantic_search.py](m05_nomic_semantic_search.py) | Which passages does a question retrieve? |

## Setup

Run from the repository root:

```bash
uv sync --all-extras
```

Keep `--all-extras`: a plain `uv sync` removes the optional packages that the
chunking demo and later lessons need.

The embedding examples use local Nomic v2 MoE on CPU. Prepare the environment
before class: M1, M4, and M5 may download model files on their first run.
All experiments share pinned weights and pinned official MoE code through
`modules/c01_embeddings/local_embeddings.py`. M2 defaults to cached files only;
use its explicit `--allow-download` flag for preparation.
The chunk-boundary comparison does not load an embedding model.

V2 uses 768 dimensions, masked mean pooling and L2 normalization, with a
512-token limit including task prefixes and special tokens. Oversized input
raises an error instead of being silently truncated. No v1.5 layer normalization
is applied. The local runtime uses Nomic's plain-PyTorch MoE path; MegaBlocks is an
optional GPU speed-up, and the course silences its install suggestion.
Weights and code take roughly 2 GB on disk. No provider API key is needed.

Existing v1.5 vectors are incompatible even though their dimensions match.
Course caches and database namespaces use separate v2 names. Historical reports
and plots remain labeled with the model that produced them.

## 1. Compare sentence embeddings

```bash
uv run --all-extras python -m modules.c01_embeddings.m01_nomic_embeddings
```

Inspect the five supplied sentences, the vector shape, and their cosine
similarity matrix. Each sentence becomes one 768-dimensional vector even when
sentences have different token counts. Pooling is handled inside the embedding
shared encoder; participants do not need to implement it.

Ask participants to predict which English paraphrase will be closest to the
original. The Greek example is an observation to inspect, not a guarantee of
multilingual quality on this dataset. V2 is multilingual, but language coverage
does not guarantee that a particular question retrieves the right passage.
Similarity scores are not probabilities.

## 2. Inspect a word in context

```bash
uv run --all-extras python -m modules.c01_embeddings.m02_contextual_words
```

For first-time preparation only:

```bash
uv run --all-extras python -m modules.c01_embeddings.m02_contextual_words --allow-download
```

Show the orange/red sentences and before/after results. This demonstration
selects only the target word's token pieces. It does not return a pooled
sentence embedding. Different contexts can move the representation even when
the word retains the same sense.

The v2 output directory is `outputs/contextual_words_v2`. Use the
[classroom guide and recorded plots](contextual_embeddings/README.md)
for prepared results and interpretation. Full-dimensional cosine comparisons
are the measurements; the 2D projection can distort distances.

Keep character offsets, token selection, and plotting code in instructor notes.

## 3. Compare chunk boundaries

```bash
uv run --all-extras python -m modules.c01_embeddings.m03_chunking_comparison
```

Inspect the same support document using three strategies:

| Order | Strategy | Rule | Trade-off |
|---|---|---|---|
| 1 | Fixed characters | Cut every 300 characters | Predictable sizes, but can split words and sentences. |
| 2 | Recursive | Prefer paragraphs, lines, and spaces within 300 characters | Preserves some boundaries, but can still separate headings and instructions. |
| 3 | Sections | Keep each heading/body block together | Preserves this guide's sections, but may exceed 300 characters. |

Fixed and recursive use the same character limit and zero overlap to focus on
boundary selection. Sections deliberately have no size cap. The section helper
uses this specific file's blank-line format and retains the document title with
the first section; it is not a general Markdown, HTML, or PDF parser.

The output shows a short explanation, chunk count, and each chunk's length and
text. No embedding model or API key is required. The recursive example uses the
same installed library as M4 and M5; see the
[LangChain explanation](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).

Ask: which chunks separate a heading from its instructions? Which sections are
larger than the fixed budget? As a follow-up, change `chunk_size` in `main()`
from 300 to 600 and rerun. Whole sections stay the same. Overlap is a separate
setting; M4 demonstrates recursive splitting with target overlap 50.

All sizes here are characters, not tokens. A real pipeline must split oversized
sections further to meet model limits. This demonstration inspects boundaries;
it does not measure retrieval or answer quality.

**Advanced discussion only:** semantic splitting uses embedding similarities to
estimate topic changes. It adds computation and threshold choices; it is not
implemented here and should not be assumed to improve retrieval.

## 4. Embed the chunks

```bash
uv run --all-extras python -m modules.c01_embeddings.m04_nomic_chunking
```

This short bridge prints each chunk, its character and Nomic token counts,
and its vector shape. It uses a 300-character budget with target overlap 50.
One chunk produces one vector. Nomic input counts include the task prefix
and special tokens; padding is not document content.

## 5. Retrieve passages

```bash
uv run --all-extras python -m modules.c01_embeddings.m05_nomic_semantic_search

uv run --all-extras python -m modules.c01_embeddings.m05_nomic_semantic_search \
  --query "There are two completed payments on my card" --top-k 2
```

This example embeds the document chunks and questions locally, compares their
vectors, and prints the best matching passages. Documents use
`search_document: ` and questions use `search_query: ` with the same encoder.
It uses 500-character chunks with target overlap 50, so its chunks differ from
module 4. It performs in-memory retrieval without a database or generated answer.

Focus on cosine ranking and the returned text. The additional dot-product and
Euclidean comparisons are supporting output. For normalized vectors those
rankings agree apart from numerical rounding and ties. Top-k can return
passages even for an unsupported question; a high rank is not proof of relevance.

## Supporting files

- `data.py`: shared sentences and support-document loading.
- `display.py`: vector and similarity output formatting.
- `local_embeddings.py`: the pinned Nomic encoder shared with lesson 2.
- `telecom_support.txt`: the fictional source document.
- `__init__.py`: Python package marker.

Continue with [lesson 2: vector databases](../c02_vector_databases/README.md).
Lessons 3–5 turn retrieved passages into cited answers.
