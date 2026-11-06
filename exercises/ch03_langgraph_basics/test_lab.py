"""Grade the LangGraph exercise. Offline: no model, no database."""

from typing import get_type_hints

TWO_TOPICS = "I was charged twice and my router shows a red light. What should I do?"


def run(lab, question):
    return lab.build_agent_graph().invoke({"question": question})


# Part 1: given, so this passes from the start


def test_part1_the_basic_graph_answers(lab):
    result = lab.build_basic_graph().invoke({"question": "My router shows a red light."})
    assert "red router light" in result["answer"]


# FIX 1: start both searches at START


def test_fix1_both_searches_start_at_start_and_feed_merge(lab):
    edges = {(e.source, e.target) for e in lab.build_agent_graph().get_graph().edges}
    assert ("search_billing", "search_support") not in edges, "search_support should not wait for search_billing"
    for search in ("search_billing", "search_support"):
        assert ("__start__", search) in edges, f"connect START to {search}"
        assert (search, "merge") in edges, f"connect {search} to merge"


# FIX 2: a reducer for steps


def test_fix2_steps_has_a_reducer(lab):
    steps = get_type_hints(lab.AgentState, include_extras=True).get("steps")
    reducer = getattr(steps, "__metadata__", [None])[0]
    assert callable(reducer), (
        "steps needs a reducer: Annotated[list[str], operator.add]. Two nodes write it in the same step"
    )
    assert reducer(["search_billing"], ["search_support"]) == ["search_billing", "search_support"]


# FIX 3: merge keeps both searches' hits


def test_fix3_merge_keeps_the_hits_of_both_searches(lab):
    billing, support = {"source": "billing.md"}, {"source": "internet_support.md"}
    update = lab.merge({"billing_hits": [billing], "support_hits": [support]})
    assert update["evidence"] == [billing, support]


# The whole agent graph: passes once all three fixes are in


def test_the_searches_run_in_the_same_step(lab):
    app = lab.build_agent_graph()
    steps = {chunk["payload"]["name"]: chunk["step"]
             for chunk in app.stream({"question": TWO_TOPICS}, stream_mode="debug") if chunk["type"] == "task"}
    assert steps["search_billing"] == steps["search_support"], "the two searches should start together"
    assert steps["merge"] > steps["search_billing"], "merge must wait for both searches"


def test_a_two_topic_question_uses_both_sources(lab):
    result = run(lab, TWO_TOPICS)
    assert {"billing.md", "internet_support.md"} <= {p["source"] for p in result["evidence"]}
    assert sorted(result["steps"][:2]) == ["search_billing", "search_support"]
    assert result["steps"][2:] == ["merge", "write_agent_answer"]


def test_a_question_the_documents_cannot_answer_is_refused(lab):
    assert run(lab, "What is the customer service phone number?")["answer"] == lab.REFUSAL
