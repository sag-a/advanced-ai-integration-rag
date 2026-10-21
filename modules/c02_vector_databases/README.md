# Advanced AI Integration & RAG — Unit 2: Vector Databases

Start with the [HTML instructor study guide](study_guide.html)
for explanations, diagrams, a filter simulator, rehearsal commands, and self-check questions.

Parts:

- **M01: PostgreSQL + pgvector** — `m01_pgvector_demo.py`, this guide.
- **[M02: ChromaDB](CHROMA.md)** — `m02_chroma_demo.py`: configured automatic embeddings, similarity
  and filtered searches, optional local hybrid search, and record update/delete operations.

## Live demo, Part 1: PostgreSQL + pgvector

Create a table with a vector column, insert real embeddings with JSONB metadata,
create an HNSW index, and search using SQL. The final stage compares exact,
normal-planner, and deliberately encouraged HNSW query plans.

This continues Unit 1: the same support document, six passages, Nomic v2 MoE on
CPU, and normalized 768-dimensional embeddings. Documents use `search_document:`;
questions use `search_query:`. No OpenAI key or LLM is needed. The first model
run may download public weights; later runs can use the existing cache.

## Start the isolated database

From the repository root:

```bash
docker compose -f modules/c02_vector_databases/compose.pgvector.yaml up -d --wait
```

The configuration uses `pgvector/pgvector:0.8.5-pg17-bookworm`, reusing that
cached image when available. It creates its own Compose project and named volume,
`advanced-rag-pgvector_pgvector_data`. The database is exposed only on loopback
at **127.0.0.1:5434**. It does not use the other course's PostgreSQL 16 data
directory or its port 5433.

Public credentials for this local teaching database are user `rag_demo`, password
`rag_demo_local`, database `rag_demo`. These are demo defaults, not production
credentials. The script creates objects only in the dedicated
`rag_pgvector_demo_nomic_v2` schema, plus the database's `vector` extension.

Install the optional drivers from the repository root:

```bash
uv sync --all-extras
```

The `pgvector` extra adds Psycopg 3 binary support and the Python pgvector adapter.
Keep `--all-extras` on the following `uv run` commands so uv retains those
optional dependencies. The examples use direct SQL, without an ORM.

## Rehearse the whole lesson

From the repository root:

```bash
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo all
```

The command runs setup, upserts the passages, prints an exact-search baseline,
creates the index, performs normal searches, and compares query plans. A repeated
run upserts the same stable IDs instead of duplicating or dropping rows. It also
keeps the existing index; the exact baseline disables index scans locally even
if an index exists from a previous rehearsal.

The default questions cover cancellation, duplicate charges, and Wi-Fi problems.
For a shorter rehearsal with one question:

```bash
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo all --query "What should I do if I was billed twice?"
```

## Present one stage at a time

Run these commands in order from the repository root:

```bash
# 1. Enable pgvector, then create a schema and table with embedding vector(768).
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo setup

# 2. Encode six passages and upsert their content, metadata, and vectors.
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo insert

# 3. Build the cosine HNSW index and update planner statistics.
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo index

# 4. Embed a question and search through SQL.
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo search --query "My Wi-Fi stopped working" --top-k 3

# 5. Inspect exact, normal, and encouraged HNSW plans for that question.
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo explain --query "My Wi-Fi stopped working" --top-k 3
```

`setup` and `index` do not load the embedding model. `insert`, `search`, and
`explain` do. The script reports missing setup/data/index prerequisites with
instructions. `--top-k` must be positive; `--query` must contain text.

## What the SQL teaches

The four files in `sql/` are printed as they execute:

| File | Lesson |
| --- | --- |
| `01_setup.sql` | Extension, dedicated schema, text, JSONB, and `vector(768)` |
| `02_insert.sql` | Bound parameters and repeatable `ON CONFLICT` upserts |
| `03_index.sql` | `USING hnsw (embedding vector_cosine_ops)` and `ANALYZE` |
| `04_search.sql` | Cosine distance, similarity, and index-compatible ordering |

The insert/search files contain **Psycopg named placeholders**, not psql variables;
run them through the Python module. Values are bound through `Vector` and `Jsonb`
adapters rather than interpolated into SQL. Printed output shows dimensions and
metadata instead of dumping 768-number vectors or database connection secrets.

Each row stores the source filename, chunk number, language, and embedding model
in JSONB. IDs 1–6 correspond to this fixture's chunk numbers. Rehearsals update
matching IDs; they do not delete additional rows if you later edit the dataset.
Stored model metadata is checked before searching to avoid mixing embedding
spaces. Matching dimensions alone do not make different encoders compatible.

`<=>` returns **cosine distance**: smaller values are closer. Similarity is
`1 - distance`, where larger values are closer. Similarities are not probabilities,
and top-k returns candidates even for an unrelated question.

For index eligibility, order directly by the distance operator:

```sql
ORDER BY embedding <=> %(query)s
LIMIT %(top_k)s;
```

Ordering by the computed `cosine_similarity DESC` expression does not give the
index-compatible form. The operator class must also match the metric.

## Reading the query plans

`explain` executes `EXPLAIN (ANALYZE, BUFFERS)` and prints the actual plans,
abbreviating only long vector literals for readability. `ANALYZE` here executes
the search to collect runtime measurements; this differs from the separate
`ANALYZE table` command that updates planner statistics.

- **Exact:** Transaction-local settings disable index and bitmap scans. Expect
  a sequential scan and a sort by vector distance.
- **Normal:** PostgreSQL chooses the plan. Six rows can legitimately favor a
  sequential scan even after HNSW has been created.
- **HNSW demonstration:** `SET LOCAL enable_seqscan = off` discourages a sequential
  scan for this one transaction. Look for
  `Index Scan using documents_embedding_hnsw` in the printed plan.

The script reports returned IDs, exact/HNSW top-k overlap, and whether ordering
matches in that run. HNSW is approximate and can miss neighbors. A perfect overlap
on this small fixture does not establish recall on a larger corpus. Reported
query times are observations, not evidence of an HNSW speedup for six rows.

The connection uses autocommit with explicit transaction blocks; `SET LOCAL`
settings reset when each block exits. A final snapshot verifies they returned
to their original values. No server-wide settings or index drops are needed.

## Inspect with SQL directly

From the repository root, open psql in the isolated service:

```bash
docker compose -f modules/c02_vector_databases/compose.pgvector.yaml exec postgres psql -U rag_demo -d rag_demo
```

Inspect persisted metadata and dimensions:

```sql
SELECT id, metadata, vector_dims(embedding) AS dimensions
FROM rag_pgvector_demo_nomic_v2.documents
ORDER BY id;
```

For a SQL-only similarity exercise, use stored passage 3 as the query vector and
exclude that passage from the results. The Python `search` stage instead embeds
an actual question:

```sql
SELECT id, content, metadata,
       1 - (embedding <=> (
           SELECT embedding FROM rag_pgvector_demo_nomic_v2.documents WHERE id = 3
       )) AS similarity
FROM rag_pgvector_demo_nomic_v2.documents
WHERE id <> 3
ORDER BY embedding <=> (
    SELECT embedding FROM rag_pgvector_demo_nomic_v2.documents WHERE id = 3
)
LIMIT 3;
```

## Connection overrides and shutdown

`PGVECTOR_DATABASE_URL` in the shell takes precedence over the same variable in
ignored `.env`. If neither is set, the script uses the local demo URL.
It never reads tracked `.env.example` files or prints the connection string.

Compose accepts `PGVECTOR_PORT` and `PGVECTOR_PASSWORD` as overrides in its shell
environment. If you change either, set a corresponding `PGVECTOR_DATABASE_URL`
for Python. Changing `POSTGRES_PASSWORD` through Compose does not change the
password of an already initialized database volume.

Stop and remove this demo container/network while **retaining the named volume**
by running from the repository root:

```bash
docker compose -f modules/c02_vector_databases/compose.pgvector.yaml down
```

The next `up -d --wait` reuses the data. No cleanup command in this lesson deletes
the volume, and the other course's containers and files remain separate.

## Sources

- [pgvector 0.8.5: operators, HNSW, and query plans](https://github.com/pgvector/pgvector/tree/v0.8.5)
- [pgvector Python adapters](https://github.com/pgvector/pgvector-python)
- [Psycopg transaction management](https://www.psycopg.org/psycopg3/docs/basic/transactions.html)
- [Official PostgreSQL container](https://hub.docker.com/_/postgres)
