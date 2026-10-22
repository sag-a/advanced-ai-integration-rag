# Setup and running every example

All commands run from the repository root unless a step says otherwise.

## 1. Prerequisites

| Tool | Why | Check |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/) | Installs Python 3.12 and every dependency | `uv --version` |
| [Docker](https://docs.docker.com/get-docker/) (Docker Desktop on macOS/Windows) | Chroma and PostgreSQL + pgvector | `docker compose version` |
| Git | Clone the repository | `git --version` |
| Anthropic API key | Lessons 3–5 only | — |

You do not need to install Python yourself. `.python-version` pins 3.12 and uv
downloads it when it is missing.

## 2. Install uv

### macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or, with Homebrew:
brew install uv
```

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# if curl is missing:
wget -qO- https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# or, with winget:
winget install --id=astral-sh.uv -e
```

### Any platform with pipx

```bash
pipx install uv
```

Close and reopen the terminal so `uv` is on your `PATH`, then run `uv --version`.
To update later: `uv self update` (or `brew upgrade uv` / `winget upgrade astral-sh.uv`).

## 3. One-time project setup

```bash
git clone git@github.com:sag-a/advanced-ai-integration-rag.git
cd advanced-ai-integration-rag
uv sync --all-extras                             # creates .venv with every dependency
uv run python -m rag_core.embeddings --download  # Nomic v2 MoE weights, about 2 GB, once
uv run --all-extras python -m rag_core.reranker --download  # reranker for --option rerank=true and jev_rag, about 2.3 GB
cp .env.example .env                             # PowerShell: Copy-Item .env.example .env
```

Put your key in `.env` as `ANTHROPIC_API_KEY=...`. Jev RAG also needs `OPENROUTER_API_KEY=...`.
Never commit `.env`; only `.env.example` with empty values is tracked.

Every lesson uses the same pinned Nomic model, so the single download covers all of
them. Ordinary questions never download weights.

Check the installation:

```bash
uv run --all-extras pytest -q      # expect all tests to pass
```

## 4. Lesson 1: embeddings (no API key, no Docker)

```bash
uv run --all-extras python -m modules.c01_embeddings.m01_nomic_embeddings       # sentences → vectors
uv run --all-extras python -m modules.c01_embeddings.m02_contextual_words       # same word, different context
uv run --all-extras python -m modules.c01_embeddings.m03_chunking_comparison    # chunking strategies
uv run --all-extras python -m modules.c01_embeddings.m04_nomic_chunking         # embed the chunks
uv run --all-extras python -m modules.c01_embeddings.m05_nomic_semantic_search  # question → passages
```

`m02` uses cached weights only; add `--allow-download` if they are missing. Its plots
are written to `outputs/contextual_words_v2/`.

## 5. Lesson 2: vector databases (Docker, no API key)

```bash
docker compose -f modules/c02_vector_databases/compose.pgvector.yaml up -d --wait   # port 5434
docker compose -f modules/c02_vector_databases/compose.chroma.yaml up -d --wait     # port 8002

# M01 pgvector: all stages, or setup / insert / index / search / explain
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo all
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo search --query "My Wi-Fi stopped working" --top-k 3
uv run --all-extras python -m modules.c02_vector_databases.m01_pgvector_demo explain --query "My Wi-Fi stopped working" --top-k 3

# M02 Chroma: all stages, or create / add / search / filter / update / delete
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo all
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo filter --query "I was charged twice" --category billing
uv run --all-extras python -m modules.c02_vector_databases.m02_chroma_demo search --method compare \
  --query "temporary card authorization" --top-k 3 --candidate-k 10

# stop the services
docker compose -f modules/c02_vector_databases/compose.pgvector.yaml down
docker compose -f modules/c02_vector_databases/compose.chroma.yaml down
```

## 6. Lessons 3–5: the RAG architectures (Docker + Anthropic key)

```bash
docker compose -f compose.chroma.yaml up -d --wait   # architectures' Chroma, port 8004
./rag architectures                     # list architectures and missing extras
./rag prepare                           # embed the PDF into Chroma, once
./rag check
./rag search "Πότε λήγουν οι Καρδιές;"  # passages only, no model call
```

Classic (base) RAG:

```bash
./rag ask "Πότε λήγουν οι Καρδιές;"
./rag ask --process "Πότε λήγουν οι Καρδιές;"      # show each node and model call
./rag search --option rerank=true "Πότε λήγουν οι Καρδιές;"   # print passages before and after reranking
./rag ask --option rerank=true --option rerank_candidates=12 "Πότε λήγουν οι Καρδιές;"
./rag ask --option answer_style=detailed "Πότε λήγουν οι Καρδιές;"
```

Graph RAG. The graph index is built once and costs about 30 Anthropic calls:

```bash
./rag prepare --architecture graph_rag            # only if outputs/graph_rag/graph.json is missing
./rag check --architecture graph_rag
./rag ask --process --architecture graph_rag "Πότε λήγουν οι Καρδιές;"
```

Agentic RAG:

```bash
./rag ask --process --architecture agentic_rag "Πότε λήγουν οι Καρδιές;"
./rag chat --architecture agentic_rag --option max_searches=3
```

Hybrid RAG (dense + keywords) on the harder tariffs corpus, prepared once (about 25 s):

```bash
./rag prepare --corpus data/corpus-tariffs --collection cosmote_tariffs_v1
./rag search  --corpus data/corpus-tariffs --collection cosmote_tariffs_v1 --architecture classic_rag "Double Play 24 XL 500 free minutes"
./rag search  --corpus data/corpus-tariffs --collection cosmote_tariffs_v1 --architecture hybrid_rag  "Double Play 24 XL 500 free minutes"
```

Jev RAG: rerank, then a Jev gate that abstains before the answer model when the
passages don't contain the answer (needs `OPENROUTER_API_KEY`; about $0.0001 per question):

```bash
./rag ask --process --architecture jev_rag "What is my current Hearts balance?"   # gate abstains, no Claude call
./rag ask --process --architecture jev_rag "Πότε λήγουν οι Καρδιές;"              # gate passes, Claude answers
./rag ask --architecture jev_rag --option reranker=jev --option gate_threshold=0.6 "…"
```

Add `--option route=true` and Jev first acts as a supervisor: it picks the Magenta
Moments collection, the tariffs collection, or "neither" (abstain at once). Both
collections must be prepared:

```bash
./rag ask --process --architecture jev_rag --option route=true "Πότε λήγουν οι Καρδιές;"                 # → magenta
./rag ask --process --architecture jev_rag --option route=true "Which Double Play 24 XL has 500 minutes?" # → tariffs
./rag ask --process --architecture jev_rag --option route=true "What's the weather in Athens tomorrow?"   # → neither
./rag ui --architecture jev_rag --option route=true
```

Terminal conversation:

```bash
./rag chat     # /retry after a failed turn
```

### Streamlit UI

Chroma must be running and the PDF prepared (`./rag check` passes). Then:

```bash
./rag ui                              # http://127.0.0.1:8504
./rag ui --architecture graph_rag     # open with Graph RAG selected
./rag ui --ui-port 8600               # if 8504 is already taken
./rag ui --top-k 3                    # any ./rag ask setting works here too
```

`./rag ui` runs `streamlit run app/streamlit_app.py` bound to `127.0.0.1`, headless
(open the URL yourself) and without file watching, so restart it with Ctrl+C after
code changes. Switch architectures from the sidebar; each switch starts a new chat.
Each turn's steps (nodes, model calls, tool calls) are printed in the terminal
running `./rag ui`, not on the page. Answers show their sources and the architecture's own panel: graph hops for Graph
RAG, research events for Agentic RAG. History lives in the browser session and is
cleared by a refresh or restart.

Prompt-template previews (no API key or model call):

```bash
./rag prompt answer --option answer_style=detailed
./rag prompt research --architecture agentic_rag --option max_searches=3 --option show_skills=true
./rag prompt extraction --architecture graph_rag --option source_language=English
```

Evaluations:

```bash
uv run --all-extras python -m evals.run --architecture classic_rag --retrieval-only   # no model calls
uv run --all-extras python -m evals.run --architecture all                            # uses Anthropic
```

`--process` works on `ask` and `chat`. It prints each step as it happens:

```text
◆ retrieve
  · graph-vector-seeds · 3 seed passages in 9.17s
  · graph-traversal · 4 start entities, 30 facts, 8 new passages
  · graph-passage-selection · 6 passages selected
◆ retrieve done in 9.24s
◆ generate_answer
  ✦ model call · claude-haiku-4-5-20251001
  ✦ model call done in 4.61s · 3981 in / 370 out tokens
```

The first turn in a new process includes loading the Nomic encoder (about 9 s).

## Windows notes

- Replace `./rag` with `uv run --all-extras python -m app`; the arguments stay the same.
- Replace `cp` with `Copy-Item`, and `\` line continuations with a backtick (`` ` ``).
- Run Docker Desktop with the WSL 2 backend.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `uv: command not found` | Open a new terminal, or add `~/.local/bin` (macOS/Linux) to `PATH` |
| `Cannot reach Chroma` | Start the matching compose file: port 8004 for `./rag`, 8002 for lesson 2 |
| `Corpus collection does not exist` | Run `./rag prepare` |
| `Graph index is incompatible` | Use the default `--model`, or `--option graph_index=PATH` for another index |
| `Set ANTHROPIC_API_KEY …` | Add the key to `.env` |
| UI port already in use | `./rag ui --ui-port 8600` |
| Optional packages disappear | Keep `--all-extras` on every `uv run` command |
