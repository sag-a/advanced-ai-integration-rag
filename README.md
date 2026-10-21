# Advanced RAG course

Teaching code for advanced retrieval-augmented generation: embeddings and vector
databases first, then one shared RAG core with Classic, Graph and Agentic
retrievers, one CLI, one Streamlit page and comparable evaluation reports.

## Layout

```text
.
├── modules/               # lessons, in teaching order (see modules/README.md)
│   ├── c01_embeddings/        # Nomic embeddings, contextual words, chunking, search
│   ├── c02_vector_databases/  # pgvector + Chroma demos, hybrid search, study guide
│   ├── c03_classic_rag/       # lesson.html + instructor_notes/
│   ├── c04_graph_rag/         # lesson.html + instructor_notes/
│   └── c05_agentic_rag/       # lesson.html + instructor_notes/
├── rag_core/              # everything shared: corpus, store, model, answers, LangGraph turn
├── architectures/         # one package per retriever: classic (+ optional reranker), graph, agentic, hybrid, jev
├── app/                   # CLI (./rag) and Streamlit UI, generic over the registry
├── evals/                 # Ragas evaluation: cases.jsonl, run.py, reports/
├── data/corpus/           # the Magenta Moments PDF
├── data/corpus-tariffs/   # harder corpus: COSMOTE 2019 price list (see data/README.md)
├── docs/                  # prompt-template lesson
├── tests/                 # core/ and architectures/ suites
├── compose.chroma.yaml    # Chroma for the RAG architectures (loopback :8004)
├── SETUP.md               # install uv on macOS/Linux/Windows, run every example
└── INSTRUCTOR_GUIDE.md    # timetable, demos, whiteboard prompts, common traps
```

Dependencies only point inward: `app`/`evals` → `architectures` → `rag_core`.
`modules/` is teaching material and may import anything.

| Lesson | Teach from | Notes |
| --- | --- | --- |
| 1. Embeddings, chunking, semantic search | [c01_embeddings](modules/c01_embeddings/README.md) | [guide](modules/c01_embeddings/GUIDE.md) |
| 2. Vector databases | [study_guide.html](modules/c02_vector_databases/study_guide.html) | [pgvector](modules/c02_vector_databases/README.md), [Chroma](modules/c02_vector_databases/CHROMA.md) |
| 3. Classic (base) RAG | [lesson.html](modules/c03_classic_rag/lesson.html) | [README](modules/c03_classic_rag/README.md), [instructor notes](modules/c03_classic_rag/instructor_notes/) |
| 4. Graph RAG | [lesson.html](modules/c04_graph_rag/lesson.html) | [README](modules/c04_graph_rag/README.md), [instructor notes](modules/c04_graph_rag/instructor_notes/) |
| 5. Agentic RAG | [lesson.html](modules/c05_agentic_rag/lesson.html) | [README](modules/c05_agentic_rag/README.md), [instructor notes](modules/c05_agentic_rag/instructor_notes/) |

Open the HTML pages in a browser. Press **P** to present one section per screen.
Instructors: start with [INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md).

## Install uv

| System | Command |
| --- | --- |
| macOS | `curl -LsSf https://astral.sh/uv/install.sh \| sh` or `brew install uv` |
| Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows (PowerShell) | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` or `winget install --id=astral-sh.uv -e` |

Open a new terminal and check with `uv --version`. uv installs Python 3.12 itself.
[SETUP.md](SETUP.md) has the full walkthrough and the commands for every example.

## Setup

Install [Docker](https://docs.docker.com/get-docker/) as well. From this directory:

```bash
uv sync --all-extras
docker compose -f compose.chroma.yaml up -d --wait
cp .env.example .env
```

In PowerShell, use `Copy-Item .env.example .env`. Put your Anthropic key in `.env`;
the tracked example contains placeholders only.
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

Add `--process` to `ask` or `chat` to watch a turn as it runs: each graph node,
every model call with its token counts, tool calls, and Graph RAG's seed, traversal
and selection stages. It is a LangChain callback (`rag_core/callbacks.py`) passed in
the turn's config, so LangGraph hands it to every nested run, including the Deep
Agents research loop. `./rag ui` always prints the same steps in the terminal it
runs in, with a header line per question; the page itself shows only answers,
sources and each architecture's panel.

Follow-ups use a plain list of messages passed directly into the graph. The CLI
keeps this list for the current `./rag chat` process; Streamlit keeps it in
`st.session_state` across widget reruns. Rewriting and generation see the last six
prior messages (roughly three exchanges). No conversation database is used.
Refresh, restart, New chat or switching architecture starts an empty history.
The URL conversation ID identifies the chat; it cannot restore a saved conversation.
Failed turns keep their draft out of history and require an explicit Retry click
or CLI `/retry`. Previously saved SQLite files are left in place but are not read.

Architecture options are owned by the plugin; for example:

```bash
./rag check --architecture graph_rag --option graph_index=outputs/another/graph.json
./rag chat --architecture agentic_rag --option max_searches=3
```

Changed corpus or embedding/chunking settings need a new collection. Changed
graph extraction model, prompt or schema needs a new graph index path. Answer
prompt changes are recorded in report identity and do not re-embed PDFs.
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

## Evaluation

[Ragas](https://docs.ragas.io/) scores 21 Greek and English classroom cases on
the Magenta Moments PDF. See [evals/README.md](evals/README.md) for the metrics.

```bash
uv run --all-extras python -m evals.run --architecture classic_rag --retrieval-only   # free
uv run --all-extras python -m evals.run --architecture all                            # uses Anthropic
```

These are classroom examples, not a held-out benchmark. Report ties and
regressions honestly.
