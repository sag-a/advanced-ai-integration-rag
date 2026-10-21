# Advanced AI Integration & RAG — Unit 2, Part 2: ChromaDB

Configure a collection's embedding function, add text documents, perform similarity
and metadata-filtered searches, then update and delete a temporary record.
The module supports each stage separately and an `all` rehearsal command.

This uses the same six support passages as Part 1, with normalized 768-dimensional
Nomic v2 MoE embeddings generated on CPU. No OpenAI key or LLM is needed.

## Start Chroma

From the repository root:

```bash
docker compose -f modules/c02_vector_databases/compose.chroma.yaml up -d --wait
```

The service uses the pinned image `chromadb/chroma:1.5.9`, binds only to
**127.0.0.1:8002**, and persists data in its own named volume,
`advanced-rag-chroma_chroma_data`, mounted at `/data`. It runs separately from
the pgvector demo and the other course's Chroma data. This is a local teaching
service without authentication; keep its loopback binding.

Install everything from the repository root:

```bash
uv sync --all-extras
```

The project pins the full `chromadb==1.5.9` package, shared with the RAG
architectures. The demo uses its HTTP client to connect to the Docker server. Its configured embedding function runs Nomic
locally in the Python client; the Docker server stores and searches the vectors.
Do not also install `chromadb-client`: both packages provide the same `chromadb`
Python namespace.

## Rehearse the complete lesson

From the repository root:

```bash
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo all
```

This creates or reopens the collection, adds missing support passages, searches,
compares an unfiltered query with a filtered query, and demonstrates update/delete
on the reserved `mutation-demo` record. A completed run leaves the six support
passages stored. It never resets the server or deletes the collection.

The header prints the client version and the server's version-endpoint response.
In the verified run, the pinned `1.5.9` image reported `1.0.0` through that
endpoint; the reported value is printed as returned rather than inferred from
the image tag.

## Present the stages

Run from the repository root:

```bash
# 1. Create an explicitly configured cosine collection.
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo create

# 2. Add document text and metadata; Chroma calls the configured encoder.
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo add

# 3. Pass the question as query_texts; Chroma embeds it automatically.
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo search --query "My Wi-Fi stopped working" --top-k 3

# 4. Compare the same question with and without a metadata filter.
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo filter --query "I was charged twice" --category billing

# 5. Update scratch text and metadata; Chroma recomputes the embedding.
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo update

# 6. Delete only that scratch record.
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo delete
```

`create` and `delete` do not load the model. Nomic loads lazily when text needs embedding;
the first model run may need to download its public weights. `--top-k` must be
positive and a provided `--query` must contain text.

## Collection and embedding behavior

The collection is `rag_chroma_demo_nomic_v2_auto`. The previous explicit-vector
collection, `rag_chroma_demo_nomic_v2`, is preserved. Run `all` once to populate
the new collection; old vectors are not silently copied or replaced.

The key teaching code is:

```python
from modules.c02_vector_databases.nomic_embedding_function import NomicEmbeddingFunction

collection = client.create_collection(
    name="my_nomic_collection",
    embedding_function=NomicEmbeddingFunction(),
    configuration={"hnsw": {"space": "cosine"}},
)
collection.add(ids=["example-1"], documents=["Check duplicate payments with billing."])
collection.query(query_texts=["I was charged twice"], n_results=1)
collection.update(ids=["example-1"], documents=["Contact billing with both payment dates."])
```

Chroma invokes the configured function for `add`, `upsert`, `update` with documents,
and `query` with `query_texts`. The main demo never passes an `embeddings` argument.
This is automatic **client-side** embedding, not model execution inside Docker.

[nomic_embedding_function.py](nomic_embedding_function.py) is a small adapter
around the existing pinned course encoder. Its document method adds
`search_document: `; its `embed_query` method adds `search_query: `. The shared
encoder handles masked mean pooling, L2 normalization, and the 512-token limit.
Text stored in Chroma remains unprefixed. Stable IDs are `telecom-001` through
`telecom-006`.

Chroma persists the function name and model contract in the collection configuration.
The adapter is registered on import and supports configuration reconstruction.
The demo also supplies the function when reopening so one lazy encoder instance
is reused for that collection's operations. Reopening in another Python program
requires importing the adapter; configuration does not install its code or weights.
Changed model contracts are rejected and require a new collection with re-embedding.
Collection metadata and the cosine metric are checked as well.

Repeated `add` stages skip existing fixture IDs. Unlike Part 1's upserts, `add`
does not refresh previously stored text. If the teaching document is edited,
existing IDs retain their prior content until explicitly updated.

## Similarity and filtered search

The default `--method dense` prints each result's ID, document, metadata, cosine distance, and
`1 - distance` as similarity. Smaller cosine distances mean closer vectors.
Scores are not probabilities, and a top-k search does not guarantee relevance.
Chroma returns one inner result list per query; the code shows how to unpack
that nested response.

Document metadata includes source, chunk number, language, model, record type,
and a human-assigned category: `cancel`, `billing`, `technical`, or `general`.
Categories are teaching tags derived from the document headings, not model
predictions. The filtered query supplies:

```python
where={"category": {"$eq": "billing"}}
```

The `filter` stage first prints the unfiltered results, then runs the same
question against only the selected category. Filtering can return fewer results
than `--top-k`. Try `--category absent-category` to see an empty filtered result.
With six fixture records, billing matches two passages and technical matches two.

Metadata filtering uses `where`. Chroma also offers a separate `where_document`
argument for document-text filtering; this lesson focuses on metadata filters.

## Hybrid search: semantic meaning plus keywords

Choose a retrieval method on `search`, `filter`, or `all`:

| Method | What runs |
| --- | --- |
| `dense` (default) | Chroma vector query using automatically generated Nomic embeddings |
| `lexical` | TF-IDF cosine keyword ranking of the eligible stored document text; no model inference |
| `hybrid` | Both branches, combined by reciprocal rank fusion (RRF) |
| `compare` | Print lexical, dense, and hybrid rankings for the same question and candidate pools |

```bash
# Run from the repository root. Keep both extras installed when switching between M01/M02.
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo search --method hybrid --query "temporary card authorization"

# Compare the three rankings. Each branch contributes at most 10 candidates.
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo search --method compare --query "temporary card authorization" --top-k 3 --candidate-k 10

# Compare unfiltered hybrid search with hybrid search restricted to billing.
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo filter --method hybrid --query "I was charged twice" --category billing
```

Chroma's native hybrid **Search API is Cloud-only**, according to the
[official availability documentation](https://docs.trychroma.com/cloud/search-api/overview).
This local Docker lesson uses `query()` for dense retrieval and `get()` to read
the keyword corpus. A small [helper](chroma_hybrid.py) uses the chapter's
[TF-IDF and RRF implementation](keyword_search.py) in Python. This is application-side hybrid search,
not a server configuration option. No extra service, dependency, or re-embedding
is required. The keyword branch is **TF-IDF, not BM25**.

Both branches receive the same metadata filter before their top candidates are
combined. TF-IDF uses the current eligible text, case-insensitive Unicode tokens,
accent folding, and hyphenated terms. Zero-score keyword results are excluded.
If there are no keyword matches, only dense candidates contribute to hybrid.
If the filter matches no records, all three rankings are empty.

For every document, RRF adds `1 / (60 + rank)` for each branch in which it appears;
ranks start at 1. A missing branch contributes zero. The helper prints both
component ranks so the combination can be explained. The constant 60 is different
from `--top-k` (results shown) and `--candidate-k` (candidates per branch).
Raw TF-IDF and dense scores are not added, and an RRF score is not a cosine
similarity or confidence percentage. Final score ties are resolved by document ID.

The keyword index is rebuilt from stored text on every question, so updates and
deletions are reflected immediately on the next retrieval. This is intentionally
a small classroom example: it rejects more than 1,000 eligible records instead
of silently searching an arbitrary subset. For larger corpora, use a persistent
keyword index or a backend with native hybrid support. No quality or performance
improvement is claimed; compare judged results under the same workload first.

## Update and delete scope

`update` uses `upsert` to prepare one reproducible temporary record named
`mutation-demo`. It starts with an internet-connectivity message, then calls
`collection.update` to change the text to a billing question, change its category
and revision. Because documents are supplied without embeddings, Chroma calls
the configured function to recompute the vector automatically. A metadata-only
update does not need a new vector when metadata is not part of the encoded text.

The stage reads back the stored document and metadata before and after the
update and prints the L2 change between stored embeddings. It temporarily raises
the count from six to seven when all fixtures are present.

`delete` calls `collection.delete(ids=["mutation-demo"])`, checks that the record
is absent, and prints the new count. Repeating deletion when it is already absent
is a no-op. A guard refuses mutation if that ID exists without the expected
scratch-record metadata. No arbitrary ID, filtered bulk deletion, or collection
deletion is exposed by this demo.

## Connection overrides and shutdown

The default client endpoint is `127.0.0.1:8002`. Override it using `--host` and
`--port`, or `CHROMA_HOST` / `CHROMA_PORT` in the shell or ignored `.env`.
CLI flags take precedence over the shell, which takes precedence over the file.
Changing the Compose host port via `CHROMA_PORT` requires the client to use the
same port. The script never reads tracked `.env.example` files.

From the repository root, stop only this lesson's container/network while
retaining its data volume:

```bash
docker compose -f modules/c02_vector_databases/compose.chroma.yaml down
```

Restart with `up -d --wait` to reopen the stored collection. No volume deletion is
part of the lesson.

## Sources

- [Docker deployment](https://docs.trychroma.com/guides/deploy/docker)
- [Python thin client](https://docs.trychroma.com/guides/deploy/python-thin-client)
- [Collection configuration](https://docs.trychroma.com/docs/collections/configure)
- [Configured and custom embedding functions](https://docs.trychroma.com/docs/embeddings/embedding-functions)
- [Add documents and explicit embeddings](https://docs.trychroma.com/docs/collections/add-data)
- [Query and get](https://docs.trychroma.com/docs/querying-collections/query-and-get)
- [Metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
- [Update and upsert](https://docs.trychroma.com/docs/collections/update-data)
- [Delete records](https://docs.trychroma.com/docs/collections/delete-data)
