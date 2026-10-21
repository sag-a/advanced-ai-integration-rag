# Advanced RAG course

Teaching code for advanced retrieval-augmented generation: embeddings and vector
databases first, then one shared RAG core with Classic, Graph and Agentic
retrievers, one CLI, one Streamlit page and comparable evaluation reports.

## Layout

```text
.
├── modules/               # lessons, in teaching order
│   ├── c01_embeddings/        # Nomic embeddings, contextual words, chunking, search
│   └── c02_vector_databases/  # pgvector + Chroma demos, hybrid search, study guide
├── rag_core/              # everything shared: corpus, store, model, answers, LangGraph turn
├── architectures/         # one package per retriever: classic_rag, graph_rag, agentic_rag
├── app/                   # CLI (./rag) and Streamlit UI, generic over the registry
├── evals/                 # dataset, metrics, judges, committed reports
├── data/corpus/           # the Magenta Moments PDF
├── docs/                  # architecture reference pages, prompt-template lesson
├── tasks/                 # migration plan and progress notes
├── tests/                 # core/ and architectures/ suites
└── compose.chroma.yaml    # Chroma for the RAG architectures (loopback :8004)
```

Dependencies only point inward: `app`/`evals` → `architectures` → `rag_core`.
`modules/` is teaching material and may import anything.

| Lesson | Start here |
| --- | --- |
| 1. Embeddings, chunking, semantic search | [modules/c01_embeddings](modules/c01_embeddings/README.md) |
| 2. Vector databases | [modules/c02_vector_databases](modules/c02_vector_databases/study_guide.html) |
| 3. Classic (base) RAG | [docs/architectures/base-rag.html](docs/architectures/base-rag.html) |
| 4. Graph RAG | [docs/architectures/graph-rag.html](docs/architectures/graph-rag.html) |
| 5. Agentic RAG | [docs/architectures/agentic-rag.html](docs/architectures/agentic-rag.html) |

## Setup

Install Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker. From this directory:

```bash
uv sync --all-extras
docker compose -f compose.chroma.yaml up -d --wait
cp .env.example .env
```

In PowerShell, use `Copy-Item .env.example .env`. Put your Anthropic key in `.env`;
the tracked example contains placeholders only. Langfuse credentials are optional.
Shell variables take precedence. Do not commit `.env` or `outputs/`.

Chroma is isolated on loopback port **8004** with its own Docker volume. Existing
course collections on port 8002 are not modified. The client/server version is
pinned to 1.5.9.

The local encoder uses pinned Nomic v2 MoE weights and official model code. It
requires roughly 2 GB in the Hugging Face cache. Downloads happen only through an
explicit model-preparation command; ordinary questions never download weights.

```bash
uv run python -m rag_core.embeddings --download
```

This CPU implementation does not require MegaBlocks or an NVIDIA GPU. Nomic's
optional MegaBlocks speed warning can be ignored on a Mac.

## Run the course

```bash
./rag architectures
./rag prepare
./rag check
./rag search "Πότε λήγουν οι Καρδιές;"
./rag ask "Πότε λήγουν οι Καρδιές;"
./rag chat
./rag ui
```

Lessons 1–2 run as modules from the same environment and need no API key:

```bash
uv run --all-extras python -m modules.c01_embeddings.m01_nomic_embeddings
docker compose -f modules/c02_vector_databases/compose.pgvector.yaml up -d --wait
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo all
```

Lesson 2 uses its own Chroma on port 8002 and pgvector on 5434, separate from the
architectures' Chroma on 8004, so the demos never touch each other's data.

The UI is at <http://127.0.0.1:8504>. On Windows PowerShell, replace `./rag` with
`uv run --all-extras python -m app`; all commands have the same arguments.

Graph preparation is explicit and uses Anthropic. It reads the prepared corpus,
caches accepted extractions and publishes a complete graph atomically:

```bash
./rag prepare --architecture graph_rag
./rag prepare --architecture graph_rag --resume  # only after an interrupted build
./rag ask --architecture graph_rag "Πότε λήγουν οι Καρδιές;"
./rag ask --architecture agentic_rag "Πότε λήγουν οι Καρδιές;"
```

Classic and local graph search use local embeddings without a research model
call. Agentic search adds paid, bounded Anthropic research. All accepted answers
use Anthropic and cite original PDF passages. Exact quotation matching and valid
citation IDs do not prove semantic correctness; inspect source support.

Add `--langfuse` to trace questions, bounded history, retrieved evidence and final
answers to the configured Cloud project. Credentials are scrubbed; tracing errors
warn without discarding an otherwise accepted answer. Use `LANGFUSE_HOST` in `.env`.

Follow-ups use a plain list of messages passed directly into the graph. The CLI
keeps this list for the current `./rag chat` process; Streamlit keeps it in
`st.session_state` across widget reruns. Rewriting and generation see the last six
prior messages (roughly three exchanges). No conversation database is used.
Refresh, restart, New chat or switching architecture starts an empty history.
The URL conversation ID groups traces; it cannot restore a saved conversation.
Failed turns keep their draft out of history and require an explicit Retry click
or CLI `/retry`. Previously saved SQLite files are left in place but are not read.

Architecture options are owned by the plugin; for example:

```bash
./rag check --architecture graph_rag --option graph_index=outputs/another/graph.json
./rag chat --architecture agentic_rag --option max_searches=3
```

Changed corpus or embedding/chunking settings need a new collection. Changed
graph extraction model, prompt or schema needs a new graph index path. Answer
prompt changes are recorded in trace/report identity and do not re-embed PDFs.
Existing prepared collections and completed graph indexes are never
automatically replaced. The original project's port 8002 data is not used.

## Prompt templates lesson

Preview trusted configuration with no model or database calls:

```bash
./rag prompt answer --option answer_style=detailed
./rag prompt research --architecture agentic_rag --option max_searches=3
```

Each preview includes the rendered text and a side-by-side default/customized
comparison. See [the Jinja classroom lesson](docs/prompt-templates.md) for variables,
conditionals, loops, filters, identity compatibility and safety boundaries.

## Migration checkpoints

The course now uses only the original Magenta Moments PDF. The synthetic PDFs
and their generation tooling were removed at the instructor's request. See
[data/README.md](data/README.md) and [evals/README.md](evals/README.md).
Evaluation uses ten reused Magenta-only development scenarios; there is no
held-out set or frozen baseline for this simplified corpus. Earlier migration
checkpoints and reports describe historical work, not the current course scope.

```bash
uv run --all-extras python -m evals.run --architecture classic_rag --retrieval-only
```

See [tasks/todo.md](tasks/todo.md). The original requested design is preserved in
[tasks/migration-plan.md](tasks/migration-plan.md). This fresh project does not
read old checkpoint files or provide compatibility launchers.

New documents will be **fictional teaching fixtures**, clearly distinguished from
the supplied Magenta Moments PDF. Held-out Greek questions remain drafts until
the instructor reviews them. Report ties and regressions honestly; do not tune a
held-out set until an expected architecture wins.
