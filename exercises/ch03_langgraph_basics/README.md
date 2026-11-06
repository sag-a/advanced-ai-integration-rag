# Chapter 3 exercise: a basic LangGraph with parallel nodes

After lesson 3 (the RAG turn, slide 53) · about 20 minutes · no API key, no Docker

The course runs every question through a LangGraph turn: `prepare_query → retrieve →
generate_answer` (`rag_core/graph.py`). Here you read a small working graph, then fix three
mistakes in a bigger one: an **agent state** and **two search nodes that run in parallel**.

```bash
uv run --all-extras python -m exercises.ch03_langgraph_basics.run      # draw the graphs, trace each question
uv run --all-extras pytest exercises/ch03_langgraph_basics             # 7 tests
```

## Part 1: read the basic graph (given)

[`starter.py`](starter.py) builds `START → retrieve → write_answer → END`.

- **State** (`BasicState`): a `TypedDict` that every node can read. A node returns a dict
  with only the fields it changes, and LangGraph merges that into the state.
- **Nodes**: plain functions, `state → update`.
- **Edges**: the order in which nodes run.

Run `run.py` and paste the printed diagram into [mermaid.live](https://mermaid.live).

## Part 2: fix the agent graph

The documents come from two sources, `BILLING` and `SUPPORT`. A question such as *"I was
charged twice and my router shows a red light"* needs both. The graph should search each
source in its own node, **at the same time**, then merge:

```text
START ─┬─► search_billing ─┬─► merge ─► write_agent_answer ─► END
       └─► search_support ─┘
```

Part 2 is already written but has three mistakes, each marked in `starter.py` with what you
will see and how to fix it. Fix them in order and run `run.py` after each one.

| Fix | What you see in `run.py` | Test |
| --- | --- | --- |
| FIX 1 | `search_billing` in step 1, `search_support` in step 2: one after the other | `-k fix1` |
| FIX 2 | After FIX 1: `InvalidUpdateError: At key 'steps': Can receive only one value per step` | `-k fix2` |
| FIX 3 | The answer to the two-topic question has no router passage | `-k fix3` |

Before FIX 1, predict: what will LangGraph do when two nodes write `steps` in the same step?
FIX 2 answers it. A **reducer** tells LangGraph how to combine two updates to the same key:
`Annotated[list[str], operator.add]` joins the two lists.

Before you fix anything, 2 tests pass and 5 fail. After FIX 1 it is still 5, because the
graph now stops with the error from FIX 2.

## Done when

All 7 tests pass and `run.py` shows, for every question:

```text
  step 1: search_billing, search_support   ← in parallel
  step 2: merge
  step 3: write_agent_answer
```

## Discuss

- Why don't `billing_hits` and `support_hits` need a reducer, but `steps` does?
- The basic graph also finds both sources for the two-topic question. When would one search
  over everything miss a source, and per-source searches not?
- Both searches run even for a pure billing question. What would a router node change?
  (Compare the capstone's Step 8 and the Jev router on slide 84.)
- In the course's turn (`rag_core/graph.py`), which node would you split into parallel nodes?

Stretch: replace `write_agent_answer` with a Claude call. Use `rag_core.config.Settings` for
the model and key, as in the capstone's `generate_answer`.
