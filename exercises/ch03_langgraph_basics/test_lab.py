"""Grade the LangGraph exercise. Offline: no model, no database."""

from typing import get_type_hints

TWO_TOPICS = "I was charged twice and my router shows a red light. What should I do?"


def run(lab, question):
    return lab.build_agent_graph().invoke({"question": question})


# Part 1: given, so these pass from the start


def test_part1_the_basic_graph_answers(lab):
    result = lab.build_basic_graph().invoke({"question": "My router shows a red light."})
    assert "red router light" in result["answer"]


# TODO 1: the agent state


def test_todo1_agent_state_has_its_fields(lab):
    fields = get_type_hints(lab.AgentState, include_extras=True)
    missing = {"question", "billing_hits", "support_hits", "evidence", "answer", "steps"} - set(fields)
    assert not missing, f"AgentState is missing {sorted(missing)}"


def test_todo1_steps_has_a_reducer(lab):
    steps = get_type_hints(lab.AgentState, include_extras=True).get("steps")
    reducer = getattr(steps, "__metadata__", [None])[0]
    assert callable(reducer), (
        "steps needs a reducer: Annotated[list[str], operator.add]. Two nodes write it in the same step"
    )
    assert reducer(["search_billing"], ["search_support"]) == ["search_billing", "search_support"]


# TODO 2: the two search nodes


def test_todo2_search_billing_reads_only_billing(lab):
    update = lab.search_billing({"question": TWO_TOPICS})
    assert update["steps"] == ["search_billing"]
    assert {hit["source"] for hit in update["billing_hits"]} <= {"refunds.md", "billing.md"}
    assert any("charged twice" in hit["text"] for hit in update["billing_hits"])


def test_todo2_search_support_reads_only_support(lab):
    update = lab.search_support({"question": TWO_TOPICS})
    assert update["steps"] == ["search_support"]
    assert {hit["source"] for hit in update["support_hits"]} == {"internet_support.md"}


# TODO 3: wire them in parallel


def test_todo3_both_searches_start_at_start_and_feed_merge(lab):
    edges = {(e.source, e.target) for e in lab.build_agent_graph().get_graph().edges}
    for search in ("search_billing", "search_support"):
        assert ("__start__", search) in edges, f"connect START to {search}"
        assert (search, "merge") in edges, f"connect {search} to merge"


def test_todo3_the_searches_run_in_the_same_step(lab):
    app = lab.build_agent_graph()
    steps = {chunk["payload"]["name"]: chunk["step"]
             for chunk in app.stream({"question": TWO_TOPICS}, stream_mode="debug") if chunk["type"] == "task"}
    assert steps["search_billing"] == steps["search_support"], "the two searches should start together"
    assert steps["merge"] > steps["search_billing"], "merge must wait for both searches"


# The whole agent graph


def test_a_two_topic_question_uses_both_sources(lab):
    result = run(lab, TWO_TOPICS)
    assert {"billing.md", "internet_support.md"} <= {p["source"] for p in result["evidence"]}
    assert sorted(result["steps"][:2]) == ["search_billing", "search_support"]
    assert result["steps"][2:] == ["merge", "write_agent_answer"]


def test_a_question_the_documents_cannot_answer_is_refused(lab):
    assert run(lab, "What is the customer service phone number?")["answer"] == lab.REFUSAL
