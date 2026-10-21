# Same word, different context — Nomic v2 MoE

This short instructor demonstration inspects target-token vectors, not pooled
retrieval embeddings. It uses the same pinned multilingual model as the other
course experiments. No API key is needed.

## Run

From the repository root, after `uv sync --all-extras`:

```bash
# Prepare while online, once:
uv run --all-extras python -m modules.c01_embeddings.m02_contextual_words --allow-download

# Reuse cached weights and code offline:
uv run --all-extras python -m modules.c01_embeddings.m02_contextual_words

# Optional: observe another transformer layer:
uv run --all-extras python -m modules.c01_embeddings.m02_contextual_words \
  --layer 6 --output-dir outputs/contextual_words_v2_layer6
```

The default output is `outputs/contextual_words_v2`. Model inputs over 512 tokens
are rejected. Both the weights and the official MoE implementation are pinned
in `modules/c01_embeddings/local_embeddings.py`. The CPU adapter restores the
padding-mask helper required by the official implementation under Transformers 5.

## What to explain

1. A token ID selects a fixed vocabulary vector.
2. Transformer layers combine it with information from its sentence.
3. We locate the target word's pieces and average only those pieces.
4. We compare vectors in 768 dimensions; PCA is a lossy display only.

V2 exposes its encoder blocks rather than `output_hidden_states`. A temporary
forward hook observes the requested block and is removed after inference.
It does not change the model's computation.

**Capitalization matters:** unlike the earlier v1.5 tokenizer, v2 gives `Red`
and `red` different token IDs in the supplied examples. R1/R2 therefore already
differ at vocabulary lookup; do not attribute that entire difference to context.
Use R1/R3 (red apple/red grape) as the same-token, same-sense context control.
O1/D1 is the identical-sentence repeat control.

Different vectors occupy different positions in the same layer's space. A
changed vector alone does not establish a changed word sense. These vectors are
not the mean-pooled and L2-normalized vectors used for passage retrieval.

## Recorded v2 observations — 2026-10-19

Final transformer layer 12; weights revision
`1066b6599d099fbb93dfcb64f9c37a7c9e503e85`, code revision
`7710840340a098cfb869c4f65e87cf2b1b70caca`.

| Pair | Lookup cosine | Contextual cosine | Same token IDs? |
|---|---:|---:|---|
| O1/O2 | 1.0000 | 0.6669 | yes |
| O1/O3 | 1.0000 | 0.7230 | yes |
| O2/O4 | 1.0000 | 0.8230 | yes |
| O5/O6 | 1.0000 | 0.6906 | yes |
| R1/R2 | 0.8037 | 0.6917 | no |
| R1/R3 | 1.0000 | 0.8546 | yes |
| R4/R5 | 1.0000 | 0.9653 | yes |
| O1/D1 | 1.0000 | 1.0000 | yes |

The contextual PCA retains about 64.2% of variance. Before/after projections
are fitted independently; their axes cannot be compared directly. These twelve
authored examples demonstrate context dependence, not word-sense accuracy or
Greek retrieval quality.

- [Before/after plot](contextual_word_positions.png)
- [Printable plot](contextual_word_positions.pdf)
- [Cosine matrices](contextual_word_similarity.png)
- [Measurements and exact token pieces](measurements.json)
- [Saved vectors](target_vectors.npz)
- [Implementation](../m02_contextual_words.py)
- [Official v2 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)

