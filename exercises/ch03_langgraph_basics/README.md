# Chapter 3 exercise: a basic LangGraph with parallel nodes

After lesson 3 (the RAG turn, slide 53) · about 25 minutes · no API key, no Docker

The course runs every question through a LangGraph turn: `prepare_query → retrieve →
generate_answer` (`rag_core/graph.py`). Here you build a small graph yourself: first read a
working two-node graph, then add an **agent state** and **two search nodes that run in
parallel**.

```bash
uv run --all-extras python -m exercises.ch03_langgraph_basics.run      # draw the graphs, trace each question
uv run --all-extras pytest exercises/ch03_langgraph_basics             # 9 tests
```

## Part 1: read the basic graph (given)

[`starter.py`](starter.py) builds `START → retrieve → write_answer → END`.

- **State** (`BasicState`): a `TypedDict` that every node can read. A node returns a dict
  with only the fields it changes, and LangGraph merges that into the state.
- **Nodes**: plain functions, `state → update`.
- **Edges**: the order in which nodes run.

Run `run.py` and paste the printed diagram into [mermaid.live](https://mermaid.live).

## Part 2: an agent state and two parallel searches

The documents come from two sources, `BILLING` and `SUPPORT`. A question such as *"I was
charged twice and my router shows a red light"* needs both. Instead of one search over
everything, search each source in its own node, **at the same time**, then merge:

```text
START ─┬─► search_billing ─┬─► merge ─► write_agent_answer ─► END
       └─► search_support ─┘
```

1. **TODO 1, the agent state.** Add the fields to `AgentState`: `billing_hits`,
   `support_hits`, `evidence`, `answer`, and `steps`, the list every node appends its name to.
2. **TODO 2, two nodes.** Write `search_billing` and `search_support` with `top_matches`.
   Each returns its own hits and `{"steps": ["<its name>"]}`.
3. **TODO 3, the wiring.** Add both nodes. Two edges from `START` make them run in
   parallel; `graph.add_edge([...both...], "merge")` makes `merge` wait for both.

Predict before you run: what happens when two nodes write `steps` in the same step? Try it
with `steps: list[str]` first. LangGraph stops with:

```text
InvalidUpdateError: At key 'steps': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

A **reducer** tells LangGraph how to combine updates: `Annotated[list[str], operator.add]`
joins the two lists.

## Done when

All 9 tests pass and `run.py` shows, for every question:

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
  (Compare the Jev router on slide 84: it picks one collection, or none.)
- In the course's turn (`rag_core/graph.py`), which node would you split into parallel nodes?

Stretch: replace `write_agent_answer` with a Claude call. Use `rag_core.config.Settings` for
the model and key, as in the capstone's `generate_answer`.

## For instructors

Reference: `LAB_TARGET=solution` for the tests and `run.py`.
