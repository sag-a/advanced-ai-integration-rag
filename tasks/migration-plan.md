> Current scope (2026-10-21): the instructor removed the six synthetic PDFs and
> their generation tooling. Use only `data/corpus/`, `magenta_course_pdf_v1` and
> the existing original-PDF graph. Evaluation now reuses ten Magenta-only
> development scenarios. The expanded-corpus review and Step 4 gate below are
> superseded; the remaining content is historical, not a request to recreate it.

# RAG Course Migration Plan

Oct 20, 2026 · @anastasia

Rebuild the Magenta Moments RAG course as one shared core plus four registered architectures, in five steps that each end with a runnable course. The old layout is not preserved.

## Goal and principles

The end state is one repository where a shared `rag_core` owns everything common, and each RAG architecture contributes only its retriever, its views and its own index or tools. Students run all four architectures from one CLI and one Streamlit page, and one eval command compares them by question type.

Principles that apply to every step:

- **Greenfield, not refactor.** Build the new tree fresh and port code in. No compatibility shims, old launchers, old URLs or old checkpoint files are kept.
- **Port the good parts, cut the rest.** Keep the three-node LangGraph, citation validation, streaming draft withdrawal, explicit retries, bounded research budgets and graph provenance. Drop code that only existed for backward compatibility or cross-package isolation.
- **One seam.** Every architecture is a function `query -> RetrievalResult` registered in `rag_core.registry`. Nothing outside an architecture package knows its internals.
- **Runnable after every step.** Each step ends with a concrete exit check; don’t start the next step until it passes.
- **Old numbers don’t matter.** The old reports scored 1.0 almost everywhere on a corpus too small to separate architectures. A new baseline is set in Step 4.
- **Delete as you go.** Old folders are removed at the end of the step that replaces them.

## Target structure

Five top-level parts, each with one responsibility; the dependency direction is always `app`/`evals` → `architectures` → `rag_core`, never the reverse.

```
rag-course/
├── pyproject.toml              # one project; extras: graph, agentic, langfuse, dev
├── .env.example                # placeholders only
├── .gitignore                  # .env, outputs/, __pycache__/, .DS_Store
├── compose.chroma.yaml
├── data/
│   └── corpus/                 # Magenta Moments PDF + new documents (Step 4)
├── rag_core/
│   ├── config.py               # one Settings dataclass + AppError
│   ├── corpus.py               # PDF -> clause-aware, page-aware chunks
│   ├── embeddings.py           # Nomic v2 MoE adapter + Chroma embedding function
│   ├── store.py                # connect, prepare, check, retrieve, read_documents
│   ├── model.py                # AnthropicChat: call() and stream()
│   ├── answers.py              # Answer schema, citation validation, references
│   ├── graph.py                # prepare_query -> retrieve -> generate_answer
│   ├── session.py              # Conversation, SQLite checkpoints, retries, filelock
│   ├── prompts.py              # REWRITE, ANSWER (generic, no eval cases)
│   ├── retrieval.py            # RetrievalResult
│   ├── registry.py             # Architecture dataclass, register(), REGISTRY
│   ├── observability.py        # Langfuse tracing + scrub
│   └── lesson_log.py           # shared Rich panels
├── architectures/
│   ├── classic_rag/            # retriever.py, __init__.py (register)
│   ├── graph_rag/              # schema, index, retrieval, global_search, view
│   ├── agentic_rag/            # tools, guard, prompts, retrieval, view, skills/
│   └── system1_rag/            # fast_path, semantic_cache, confidence, view
├── app/
│   ├── __main__.py             # python -m app <command> --architecture <name>
│   ├── cli.py                  # check, prepare, search, ask, chat
│   └── streamlit_app.py        # architecture picker driven by REGISTRY
├── evals/
│   ├── datasets/cases.jsonl    # one file, type + split per case
│   ├── metrics.py              # retrieval + answer metrics, N/A handling
│   ├── judges.py               # LLM factual/faithfulness judges
│   ├── run.py                  # python -m evals.run --architecture all
│   └── reports/                # committed baseline reports
├── modules/                    # c01_embeddings ... c07_capstone (Step 5)
└── tests/
    ├── core/
    └── architectures/
```

| Part | Owns | May import |
| --- | --- | --- |
| `rag_core` | Everything shared by two or more architectures | Third-party libraries only |
| `architectures/*` | One retriever plus its index, tools and views | `rag_core` |
| `app` | CLI and Streamlit, generic over the registry | `rag_core`, registry entries |
| `evals` | Dataset, metrics, judges, reports | `rag_core`, registry entries |
| `modules` | Lesson READMEs, notebooks, exercises | Everything (teaching material) |

## Old to new

Every existing source file either moves to a new home (and is simplified on the way) or is deleted; the Step column says when.

| Old file | New home | What changes | Step |
| --- | --- | --- | --- |
| `advanced_rag_01_embeddings/local_embeddings.py` | `rag_core/embeddings.py` | Merged with the Chroma embedding function; keeps `CONTRACT`, `MAX_TOKENS` | 2 |
| `advanced_rag_02_vector_databases/nomic_embedding_function.py` | `rag_core/embeddings.py` | Merged as above | 2 |
| `advanced_rag_01_embeddings/m01–m05_*.py`, `display.py`, `data.py` | `modules/c01_embeddings/` | Rewritten as notebooks on the new corpus | 5 |
| `advanced_rag_02_vector_databases/m01–m02_*.py`, `keyword_search.py`, `chroma_hybrid.py`, `sql/` | `modules/c02_vector_databases/` | Same, on the new corpus | 5 |
| `advanced_rag_01_embeddings/telecom_support.txt` | Deleted | One corpus for the whole course | 5 |
| `advanced_rag_01_embeddings/Magenta_Moments_Terms_Conditions_gr.pdf` | `data/corpus/` | Moved | 1 |
| `base_rag_architecture/config.py` | `rag_core/config.py` | One `Settings` with optional architecture fields | 2 |
| `base_rag_architecture/corpus.py` | `rag_core/corpus.py` | Multi-document input (Step 4 corpus) | 2 |
| `base_rag_architecture/store.py` | `rag_core/store.py` | Unchanged logic; Chroma timeout adapter kept | 2 |
| `base_rag_architecture/model.py` | `rag_core/model.py` | Unchanged | 2 |
| `base_rag_architecture/answers.py` | `rag_core/answers.py` | Unchanged | 2 |
| `base_rag_architecture/graph.py` | `rag_core/graph.py` | Optional entry node hook for System1 (Step 5) | 2 |
| `base_rag_architecture/session.py` | `rag_core/session.py` | `filelock` replaces `fcntl`; one SQLite file, architecture in thread ID | 2 |
| `base_rag_architecture/retrieval.py` | `rag_core/retrieval.py` | `as_result` two-tuple path removed | 2 |
| `base_rag_architecture/prompts.py` | `rag_core/prompts.py` | Eval-specific examples removed (Step 4) | 2, 4 |
| `base_rag_architecture/observability.py` | `rag_core/observability.py` | Unchanged | 2 |
| `base_rag_architecture/cli.py` | `app/cli.py` | Generic over the registry | 3 |
| `base_rag_architecture/streamlit_app.py` | `app/streamlit_app.py` | Architecture branches replaced by registry lookups | 3 |
| `base_rag_architecture/evaluate.py`, `experiments.py`, `judges.py` | `evals/run.py`, `metrics.py`, `judges.py` | Rebuilt around the new dataset | 4 |
| `base_rag_architecture/eval_cases.json`, `evaluation/*.json` | Deleted | Replaced by `evals/datasets/cases.jsonl` | 4 |
| `base_rag_architecture/README.md`, `BASELINE.md`, `LANGFUSE.md` | `modules/c03_classic_rag/` | Split into lesson README and instructor notes | 5 |
| `graph_rag_architecture/schema.py`, `index.py`, `retrieval.py`, `view.py` | `architectures/graph_rag/` | Ported; registered | 3 |
| `graph_rag_architecture/tracing.py` | `architectures/graph_rag/` | Uses `rag_core.observability` | 3 |
| `graph_rag_architecture/lesson_logs.py` | `rag_core/lesson_log.py` | Merged with agentic `logs.py` | 3 |
| `graph_rag_architecture/cli.py`, `config.py` | Deleted | Folded into `app/cli.py` and `rag_core/config.py` | 3 |
| `agentic_rag_architectures/tools.py`, `guard.py`, `prompts.py`, `retrieval.py`, `view.py`, `skills/` | `architectures/agentic_rag/` | Ported; registered | 3 |
| `agentic_rag_architectures/logs.py` | `rag_core/lesson_log.py` | Merged | 3 |
| `agentic_rag_architectures/cli.py`, `config.py` | Deleted | Folded into app and core | 3 |
| `*/MEASUREMENTS.md` | `modules/c0X_*/instructor_notes/` | Historical reference only | 5 |
| `__pycache__/`, `__MACOSX/`, `*.pyc` | Deleted | Never committed again | 1 |

## Step 1 — Scaffold the new repo

Create an empty but installable project with the final folder layout, safe credential handling and tooling, so every later step only adds code. Estimated effort: half a day.

**Tasks**

- [ ] Create the tree from *Target structure* with empty `__init__.py` files in `rag_core/`, each `architectures/*/`, `app/` and `evals/`.
- [ ] Write one `pyproject.toml`: core dependencies (`langgraph`, `langchain-anthropic`, `chromadb` pinned to 1.5.9, `pypdf`, `transformers`, `torch`, `pydantic`, `rich`, `python-dotenv`, `filelock`, `streamlit`); extras `graph` (`graphviz`, `networkx`, `leidenalg` or `python-louvain`), `agentic` (`deepagents==0.7.19`), `langfuse`, `dev` (`pytest`, `ruff`).
- [ ] Add `.gitignore`: `.env`, `outputs/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `__MACOSX/`.
- [ ] Add `.env.example` with placeholders only (`ANTHROPIC_API_KEY=`, `LANGFUSE_PUBLIC_KEY=`, `LANGFUSE_SECRET_KEY=`, `LANGFUSE_HOST=`). Document that real keys go in `.env`, which is the default and never committed.
- [ ] Copy `compose.chroma.yaml` and move the Magenta Moments PDF to `data/corpus/`.
- [ ] Configure `ruff` and `pytest` in `pyproject.toml`; add a `tests/test_smoke.py` that imports every package.
- [ ] Write a top-level `README.md` stub: prerequisites, `uv sync --all-extras`, start Chroma, where models come from (the Nomic weights preparation from the old Chapter 1).

**Exit criteria**

- [ ] `uv sync --all-extras` succeeds on macOS, Linux and Windows.
- [ ] `pytest` and `ruff check .` pass.
- [ ] `git status` after a run shows no generated files and no credentials.

## Step 2 — Build rag\_core

Port the shared modules out of `base_rag_architecture` and the two chapter folders into `rag_core`, simplifying as you go, until Classic RAG answers end to end. Estimated effort: 2–3 days.

**Tasks**

- [ ] `embeddings.py`: merge `local_embeddings.py` and `nomic_embedding_function.py`. Keep the encoder contract (768 dims, cosine, document/query prefixes, masked mean pooling, L2 normalisation) and the 512-token ceiling.
- [ ] `config.py`: one frozen `Settings` with the current validation, plus `architecture: str = "classic_rag"` and an `extras: dict` for architecture-specific options (graph index path, research budgets). Default `env_file` to `.env`.
- [ ] `corpus.py` and `store.py`: port clause-aware, page-aware chunking and the `check` / `prepare` / `retrieve` / `read_documents` functions. Keep the identity hashing and the rule that preparation never overwrites a collection. Make `corpus.py` accept a directory of documents, not one PDF path.
- [ ] `model.py`, `answers.py`, `observability.py`: port unchanged.
- [ ] `retrieval.py`: keep `RetrievalResult(documents, elapsed_seconds, details)`; delete `as_result` and the two-tuple path.
- [ ] `graph.py`: port `build_graph(chat, retriever, settings, checkpointer)`. Add an optional `entry` node parameter (default none) so System1 can insert a fast path in Step 5 without editing the core.
- [ ] `session.py`: port `Conversation`, explicit retries and message IDs. Replace `fcntl` with `filelock`. Use one checkpoint file `outputs/conversations.sqlite`; prefix thread IDs with the architecture name (`graph_rag:lesson`) instead of separate files and rejection rules.
- [ ] `prompts.py`: port `REWRITE` and `ANSWER` as-is for now; de-leaking happens in Step 4.
- [ ] `architectures/classic_rag/retriever.py`: a function wrapping `store.retrieve` to return `RetrievalResult` with `details={"architecture": "classic_rag"}`.
- [ ] Port the focused tests for answers, session (retries, stale evidence, history budget) and store into `tests/core/`.
- [ ] Delete `advanced_rag_01_embeddings/local_embeddings.py` and `advanced_rag_02_vector_databases/nomic_embedding_function.py` once nothing imports them.

**Simplifications to make on purpose**

- No separate SQLite per architecture and no “sharing the baseline checkpoint is rejected” checks.
- No legacy URL or old-thread handling.
- One `AppError` type, raised from `rag_core.config`.

**Exit criteria**

- [ ] A temporary script builds the graph with the Classic retriever and answers “Τι είναι οι Καρδιές στο Magenta Moments;” with validated citations.
- [ ] A follow-up “Πότε λήγουν;” on the same thread resolves to Hearts after a process restart.
- [ ] `tests/core/` passes; `grep -r base_rag_architecture rag_core` returns nothing.

## Step 3 — Port Graph and Agentic, add the registry and one app

Introduce the registry, port the two advanced architectures as registered plugins, and replace three CLIs and the hard-wired Streamlit page with one generic app. Estimated effort: 3–4 days.

**The registry contract**

```python
# rag_core/registry.py
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Architecture:
    name: str                                # "graph_rag"
    label: str                               # "Graph RAG" (UI picker)
    make_retriever: Callable                 # (settings, collection, log) -> callable(query) -> RetrievalResult
    prepare: Callable | None = None          # optional offline step, e.g. graph extraction
    check: Callable | None = None            # validates its own prerequisites
    render_details: Callable | None = None   # Streamlit expander for RetrievalResult.details
    log_panels: Callable | None = None       # Rich panels for the terminal
    entry_node: Callable | None = None       # optional node before prepare_query (System1)
    needs_llm_for_retrieval: bool = False    # agentic: excluded from no-API retrieval evals

REGISTRY: dict[str, Architecture] = {}

def register(arch: Architecture) -> None:
    if arch.name in REGISTRY:
        raise ValueError(f"{arch.name} already registered")
    REGISTRY[arch.name] = arch

def load_all() -> None:
    """Import each architecture package; skip ones whose optional extras are missing."""
```

**Tasks**

- [ ] Write `rag_core/registry.py` as above. `load_all()` imports `architectures.*` and records which ones failed on a missing extra, so the UI can say “install the `agentic` extra” instead of crashing.
- [ ] Register `classic_rag` in `architectures/classic_rag/__init__.py`.
- [ ] Port `graph_rag`: `schema.py`, `index.py` (extraction, atomic publish, cache, `--resume`), `retrieval.py` (seeds, BFS, RRF), `view.py`, `tracing.py`. Replace its `Settings` with `settings.extras["graph_index"]`. Keep the default index at `outputs/graph_rag/graph.json`. Register with `prepare`, `check`, `render_details`, `log_panels`.
- [ ] Port `agentic_rag`: `tools.py`, `guard.py`, `prompts.py`, `retrieval.py`, `view.py`, `skills/`. Move budgets (5 searches, 5 clause reads, 14 model calls, 35 tool calls, 2 delegations, 150 s) into `settings.extras`. Register with `needs_llm_for_retrieval=True`.
- [ ] Merge `graph_rag_architecture/lesson_logs.py` and `agentic_rag_architectures/logs.py` into `rag_core/lesson_log.py`: one `LessonLog` with a shared retrieval ID and panel style; each architecture supplies its panels through `log_panels`.
- [ ] Write `app/cli.py` with `check`, `prepare`, `search`, `ask`, `chat`, all taking `--architecture` (default `classic_rag`) and the existing `--thread`, `--message-id`, `--top-k`, `--langfuse`, `--env-file` options.
- [ ] Write `app/streamlit_app.py`: architecture picker from `REGISTRY`, the existing progress → draft → final streaming, Sources expander, `render_details` expander, New chat, resume by ID, explicit retry.
- [ ] Add one launcher script `./rag` that calls `uv run python -m app "$@"` and `./rag ui` for Streamlit with the file watcher disabled.
- [ ] Port graph and agentic tests into `tests/architectures/`; add a registry test that every registered architecture returns a `RetrievalResult` for a fixed query (agentic marked as needing an API key).
- [ ] Delete `base_rag_architecture/`, `graph_rag_architecture/`, `agentic_rag_architectures/` and the `./graph-rag`, `./agentic-rag` launchers.

**Exit criteria**

- [ ] `./rag ask --architecture X "Πότε λήγουν οι Καρδιές;"` works for `classic_rag`, `graph_rag` and `agentic_rag`.
- [ ] The Streamlit picker shows all three; switching starts a new thread; restart restores history and details.
- [ ] No file outside `architectures/<name>/` mentions an architecture by name, except registry tests.

## Step 4 — Evals from scratch

Build a corpus and dataset where the architectures can actually differ, remove test-set leakage from the prompts, and commit a new baseline report. Estimated effort: 4–6 days, most of it writing and reviewing questions.

**4a. Grow the corpus**

The current 5-page PDF yields 30 chunks, so top-k 6 retrieves 20% of the corpus and every retriever succeeds. Target at least 150–300 chunks across documents that relate to each other.

- [ ] Add 4–6 documents to `data/corpus/`, for example an older T&C version, a Magenta Moments FAQ, partner offer terms, a price list or tariff sheet, and a general telecom loyalty policy.
- [ ] Include deliberate overlaps and conflicts (two versions of the same clause, an FAQ that simplifies a rule) so version-aware and multi-hop questions exist.
- [ ] Re-run `./rag prepare` into a new collection `magenta_corpus_v1` and rebuild the graph index.

**4b. One dataset, typed and split**

- [ ] Create `evals/datasets/cases.jsonl` with one case per line: `id`, `language`, `type`, `split`, `question`, optional `history`, `expected_status`, `expected_evidence` (document, page, clause), `expected_facts`.
- [ ] Write 60–80 cases, roughly 60% dev and 40% held-out, Greek and English:

| Type | Tests | Cases | Expected winner |
| --- | --- | --: | --- |
| `factual` | One fact from one clause | 15 | Classic RAG |
| `followup` | Pronoun resolution, clarification | 10 | All equal (shared rewrite) |
| `multihop` | Facts joined across clauses or documents | 15 | Graph or Agentic RAG |
| `global` | Whole-corpus synthesis (“all restrictions on Hearts”) | 10 | Graph RAG (global search) |
| `version` | Which document or version applies | 5 | Agentic RAG |
| `abstain` | Account data, live offers, absent documents | 8 | All equal |
| `repeated` | Near-duplicate FAQ phrasings | 10 | System1 + RAG |

- [ ] Have a Greek-speaking reviewer check held-out questions and expected facts before they are frozen.

**4c. Remove leakage from prompts**

- [ ] Delete the literal “Πότε λήγουν;” Hearts/coupons example from `REWRITE`; replace it with a generic rule about ambiguous pronouns.
- [ ] Delete the named balance, live-offer and privacy-policy rules from `ANSWER`; replace them with “never infer personal account data or content of documents not in the evidence.”
- [ ] If few-shot examples help, load them from dev cases only; assert in a test that no held-out question appears in any prompt.

**4d. Metrics, judges and runner**

- [ ] Port retrieval recall, precision and MRR at k; report actual selected count for agentic.
- [ ] Port factual correctness, required-fact recall, faithfulness and citation support judges.
- [ ] Mark faithfulness and citation support as `null` (not applicable) for intentional abstentions with empty evidence, instead of scoring them 0.
- [ ] Record latency, LLM calls and tokens per stage (rewrite, research, generation, judge) from `RetrievalResult.details` and the answer observation.
- [ ] `python -m evals.run --architecture all --split heldout` writes `evals/reports/<date>.json` and a Markdown table grouped by `type` and `language`.
- [ ] Retrieval-only runs (no API) skip architectures with `needs_llm_for_retrieval=True`.

**Exit criteria**

- [ ] One command produces the per-type comparison for every registered architecture.
- [ ] At least one type shows a measurable gap between architectures (if none does, the corpus or questions are still too easy).
- [ ] The first held-out report is committed as the course baseline.

## Step 5 — System1, Graph global search and course modules

Add the two missing capabilities as registry plugins, then package everything as seven lessons. Estimated effort: 5–7 days.

**5a. System1 + RAG** (`architectures/system1_rag/`)

A fast path answers immediately when it is confident; otherwise the turn falls through to a configurable slow-path architecture (default `classic_rag`).

```text
START -> fast_path --hit--> END
             |
             +--miss--> prepare_query -> retrieve (slow path) -> generate_answer -> cache_write -> END
```

- [ ] `semantic_cache.py`: store only validated `answered` responses with their source map, keyed by the embedded standalone question, in a separate Chroma collection `system1_cache_v1`. Invalidate on corpus or prompt identity change.
- [ ] `confidence.py`: hit if cosine similarity ≥ a threshold tuned on dev `repeated` cases (start at 0.92), plus optional cheap-model self-rating.
- [ ] `fast_path.py`: the `entry_node` registered with the architecture; returns the cached answer with its original citations and `details={"path": "fast", "similarity": ...}`, or passes through.
- [ ] Follow-ups always take the slow path unless rewriting already produced a standalone query.
- [ ] `view.py`: show fast vs slow path, similarity and the cached question it matched.
- [ ] Register as `system1_rag` with `extras["slow_path"]`. Exit check: the second asking of a `repeated` case is served from cache in under 1 s with valid citations.

**5b. Graph global search** (`architectures/graph_rag/global_search.py`)

- [ ] Detect communities over the entity/fact graph (Leiden or Louvain) at prepare time.
- [ ] Generate one summary per community with source chunk IDs; cache it in the index manifest.
- [ ] At query time, route `global`-style questions (classifier or simple rewrite flag) to map-reduce over summaries, then return the original passages the summaries cite, so citation validation is unchanged.
- [ ] Record `details["mode"] = "local" | "global"` and show it in the view. Exit check: `global` cases improve over Classic on fact recall.

**5c. Course modules** (`modules/`)

| Module | Teaches | Built from |
| --- | --- | --- |
| `c01_embeddings` | Embeddings, tokenisation, chunking, similarity | Old Chapter 1 scripts, new corpus |
| `c02_vector_databases` | Chroma, pgvector, keyword and hybrid search | Old Chapter 2 scripts and SQL |
| `c03_classic_rag` | The three-node graph, citations, memory, evals | `rag_core` + `classic_rag` |
| `c04_graph_rag` | Extraction, traversal, RRF, global search | `graph_rag` |
| `c05_agentic_rag` | Skills, tools, budgets, delegation | `agentic_rag` |
| `c06_system1_rag` | Fast path, semantic cache, confidence gating | `system1_rag` |
| `c07_capstone` | Compare all four per question type, cost vs quality | `evals` reports |

- [ ] Each module gets `README.md` (one page: concept, diagram, three commands, what to look for), `notebooks/`, `exercises/` with TODO stubs, and `instructor_notes/` (old MEASUREMENTS/BASELINE content, expected results, common mistakes).
- [ ] Keep `solutions/` on a separate `solutions` branch.
- [ ] Delete `advanced_rag_01_embeddings/` and `advanced_rag_02_vector_databases/` once their scripts are ported.

**Exit criteria**

- [ ] The picker and CLI list four architectures.
- [ ] `c07_capstone` reproduces the held-out comparison table for all four.
- [ ] Nothing from the old layout remains in the repository.

## Risks and open questions

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Larger corpus still too easy | Architectures tie again; capstone teaches nothing | Exit check in Step 4 requires a measurable gap; add harder `multihop`/`global` cases until it appears |
| Graph extraction cost on a 5–10× larger corpus | Old run used \~110k input tokens for 30 chunks | Estimate before running; extract once, cache, ship the prepared index to students |
| Deep Agents 0.7.19 API drift | Agentic port breaks on upgrade | Keep the pin; isolate framework calls in `agentic_rag/retrieval.py` |
| Semantic cache serves a stale or wrong answer | System1 returns confident errors | Cache only validated answers; invalidate on corpus/prompt identity; tune threshold on dev only |
| Local Nomic weights on student machines | Setup failures, especially on Windows | Document one preparation command in `c01`; test setup on all three OSes in Step 1 |
| Held-out leakage creeps back | Inflated scores | Test that no held-out question appears in prompts or skills |

Open questions:

- [ ] Which additional documents can be used for the corpus, and are they licensed for course use?
- [ ] Should students run graph extraction themselves (paid) or receive a prepared index?
- [ ] Is pgvector still taught in `c02`, or is Chroma enough for the rest of the course?
- [ ] Does Langfuse stay optional, or become a required part of `c03` and `c07`?
- [ ] Who is the independent Greek-speaking reviewer for the held-out set?
