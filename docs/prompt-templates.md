# Classroom lesson: prompts as Jinja templates

From the repository root, these previews need no API key, model call, model
download or database connection:

```bash
./rag prompt answer
./rag prompt answer --option answer_style=detailed
./rag prompt answer --option response_language=English --option 'programs=["Magenta Moments"]'
./rag prompt rewrite --option max_subjects=2 --option mention_history_truncation=false
./rag prompt research --architecture agentic_rag --option max_searches=3 --option show_skills=true
./rag prompt research --architecture agentic_rag --option top_k=3
./rag prompt extraction --architecture graph_rag --option source_language=English --option 'entity_types=["program","condition"]'
```

Each command prints the rendered text, its identities, and a two-column comparison
with defaults. Added/changed text is green; replaced default text is red. Shell
quoting matters for JSON lists. `--option top_k=3` selects the same shared passage
budget as `--top-k 3`, including the retriever and agent's enforced budget.

## Walk through the templates

| Jinja feature | File | What students change |
|---|---|---|
| Variable `{{ source_language }}` | `architectures/graph_rag/templates/extraction.j2` | Language named in the extraction instructions |
| Conditional `{% if ... %}` | `rag_core/templates/answer.j2` | Concise/detailed style; user language/fixed language |
| Loop `{% for ... %}` | `rag_core/templates/answer.j2` | Configured program names |
| Filter `{{ response_language \| upper }}` | `rag_core/templates/answer.j2` | Display a configured language in uppercase |
| Loop over structured metadata | `architectures/agentic_rag/templates/research.j2` | Show bundled skill names and descriptions with `show_skills=true` |

Defaults deliberately preserve the original text, including its final newline.
`programs=[]` and `entity_types=[]` add no extra instructions. An empty entity-type
list uses all nine existing schema types. A nonempty list may only select from
those schema types; it cannot invent a new type. Skill names/descriptions are read
from the bundled `skill_files()` front-matter, rather than copied into Python.
The skill list is shown only when requested, preserving the original default.

`max_subjects=null` adds no subject-count restriction. History-truncation guidance
is enabled by default. Research defaults are five searches and two delegations;
the enforcement still lives in `ResearchOptions` and the tool guards. Helper names
are constrained to the existing SDK helper and its legacy label, so a template
option cannot introduce an unconstrained worker.

## Keep configuration separate from runtime data

`rag_core/templating.py` renders only approved configuration variables with
`SandboxedEnvironment`, `StrictUndefined`, disabled autoescaping and explicit
whitespace settings. Unknown names, invalid values and missing template variables
raise readable `AppError`s. Templates load from package resources and ship in the
wheel. The untrusted-data instructions are written directly in each template.

Questions, conversation messages, retrieved passages, source lines and entity
catalogs remain in the existing JSON payload. They never become Jinja variables.
A question containing `{{ 7*7 }}` reaches Anthropic as that literal string, not
`49`. The template engine is not a mechanism for trusting document instructions.

## Identity and compatibility

Two different hashes are necessary:

- `prompt_id` preserves the historical text hash when rendering produces the
  unchanged default text. Existing combined answer/rewrite and research/helper
  hashes retain their previous values too. Changed prompts use the new template
  identity.
- `template_id` always hashes the template name, source and resolved variables.
  Trace/report metadata records
  these under `prompt_template_ids`, alongside the compatible `prompt_id`.

A hash of name/source/variables cannot equal the pre-existing text hash in general.
Keeping both identities avoids invalidating a completed graph merely because its
unchanged prompt now comes from a template. The graph contract keeps its existing
`prompt_id` field and exact shape. A changed extraction prompt gets a new ID and
therefore requires a separately prepared graph index. Previewing it changes no
data. Changing only `answer_style` leaves the extraction contract compatible.

Research has two views: `render("research")` reproduces the old constant, while
the runtime preview reproduces the previously applied constrained-helper name and
final passage-budget suffix. The replacement now happens inside Jinja. Agentic
reports retain their default combined legacy ID and additionally record the
effective runtime prompt identity. The judge in `evals/judges.py` is unchanged.

## Finish the demonstration

Ask the same question twice in separate conversations, first with defaults and
then with `--option answer_style=detailed`. Inspect the prompt comparison before
making paid calls. Style is an instruction, not a guarantee of improved quality;
evidence and citation validation are unchanged.

Each `ask` command starts with empty history and writes no conversation database.
Use `./rag chat` for follow-ups within one terminal process:

```bash
./rag ask --architecture classic_rag "Πότε λήγουν οι Καρδιές στο Magenta Moments;"
./rag ask --architecture classic_rag --option answer_style=detailed "Πότε λήγουν οι Καρδιές στο Magenta Moments;"
```

Windows users can use `uv run --all-extras python -m app` in place of `./rag`.

## Verification on 2026-10-21

The default corpus/collection passed `./rag check` for Classic, Graph and Agentic
RAG without preparation. Five golden prompts, the legacy combined IDs and the
unchanged judge ID passed deterministic checks. A built wheel loaded all five
prompts using package resources.

Four sequential live calls used `claude-haiku-4-5-20251001`, the same Greek Hearts
expiry question and temporary SQLite checkpoints (before the subsequent
simplification to caller-owned message lists). Each returned `answered` with
one citation to the supplied PDF. End-to-end CLI times were 14.45 s (Classic),
12.71 s (Graph), 19.96 s (Agentic) and 12.10 s (Classic with detailed style).
These single calls are smoke checks, not performance or quality comparisons.
The detailed answer was only slightly longer for this narrow question.

The Chroma record hash and all pre-existing file hashes under `outputs/` remained
unchanged. No graph preparation, classroom-history write or judge run was needed.
