"""Chapter 3 exercise: a basic LangGraph, then an agent state with two parallel searches.

Part 1 works: run it and read it. Part 2 is written too, but has three mistakes, marked
FIX 1, FIX 2 and FIX 3. Fix them in that order; each is about one line. After each fix:

    uv run --all-extras python -m exercises.ch03_langgraph_basics.run
    uv run --all-extras pytest exercises/ch03_langgraph_basics -k fix1     (then fix2, fix3)
"""

import operator  # noqa: F401  (FIX 2)
import re
from typing import Annotated, TypedDict  # noqa: F401  (FIX 2)

from langgraph.graph import END, START, StateGraph

# Two document sources: (file, text). A question may need one source or both.
BILLING = [
    ("refunds.md", "Customers can receive a full refund if they cancel within 14 days of purchase."),
    ("refunds.md", "Customers who have already used the service are not eligible for a full refund."),
    ("billing.md", "If you were charged twice, check whether one charge is a temporary card authorization. "
                   "If both charges are completed, contact the billing team."),
    ("billing.md", "A late fee of 5 euros is added when a bill is not paid within 10 days of its due date."),
]
SUPPORT = [
    ("internet_support.md", "A red router light means there is no connection to our network."),
    ("internet_support.md", "If Wi-Fi stops working, restart the router by unplugging it for 30 seconds."),
    ("internet_support.md", "Planned maintenance is announced by SMS at least 48 hours in advance."),
]
REFUSAL = "The documents do not contain this information."
# Common words, plus "customer" and "service", which appear in almost every passage here.
STOP_WORDS = {"a", "an", "and", "are", "can", "do", "for", "i", "if", "in", "is", "it", "my", "of",
              "on", "or", "the", "to", "was", "what", "when", "who", "you", "your", "shows", "should",
              "customer", "customers", "service"}


def words(text):
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9-]+", text.lower()) if w not in STOP_WORDS}


def top_matches(question, documents, k=2):
    """The k documents sharing the most words with the question (none that share no words)."""
    scored = [{"source": s, "text": t, "score": len(words(question) & words(t))} for s, t in documents]
    return sorted((d for d in scored if d["score"]), key=lambda d: -d["score"])[:k]


def answer_from(passages):
    """A grounded answer from the passages, with sources. No model call in this exercise."""
    if not passages:
        return REFUSAL
    return "\n".join(f"- {p['text']} [{p['source']}]" for p in passages)


# ---------------------------------------------------------------------------
# Part 1: a basic graph. START → retrieve → write_answer → END
# ---------------------------------------------------------------------------


class BasicState(TypedDict):
    question: str
    passages: list[dict]
    answer: str


def retrieve(state):
    return {"passages": top_matches(state["question"], BILLING + SUPPORT)}


def write_answer(state):
    return {"answer": answer_from(state["passages"])}


def build_basic_graph():
    graph = StateGraph(BasicState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("write_answer", write_answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "write_answer")
    graph.add_edge("write_answer", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Part 2: an agent state and two searches that run in parallel.
#
#   START ─┬─► search_billing ─┬─► merge ─► write_agent_answer ─► END
#          └─► search_support ─┘
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    question: str
    billing_hits: list[dict]
    support_hits: list[dict]
    evidence: list[dict]
    answer: str
    # FIX 2 · You see (after FIX 1): InvalidUpdateError: At key 'steps': Can receive only one
    #         value per step. Both searches now write steps in the same step.
    #         Fix: give steps a reducer, a rule that joins the two updates:
    #         Annotated[list[str], operator.add]
    steps: list[str]


def search_billing(state):
    return {"billing_hits": top_matches(state["question"], BILLING), "steps": ["search_billing"]}


def search_support(state):
    return {"support_hits": top_matches(state["question"], SUPPORT), "steps": ["search_support"]}


def merge(state):
    """Runs once both searches are done: put their hits together."""
    # FIX 3 · You see: the two-topic question's answer has no router passage.
    #         Fix: the evidence must hold the hits of both searches.
    return {"evidence": state["billing_hits"], "steps": ["merge"]}


def write_agent_answer(state):
    return {"answer": answer_from(state["evidence"]), "steps": ["write_agent_answer"]}


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("search_billing", search_billing)
    graph.add_node("search_support", search_support)
    graph.add_node("merge", merge)
    graph.add_node("write_agent_answer", write_agent_answer)
    graph.add_edge(START, "search_billing")
    # FIX 1 · You see: run.py shows search_billing in step 1 and search_support in step 2.
    #         Fix: start search_support from START too, so both searches run in the same step.
    graph.add_edge("search_billing", "search_support")
    # A list of sources: merge waits until both searches have finished.
    graph.add_edge(["search_billing", "search_support"], "merge")
    graph.add_edge("merge", "write_agent_answer")
    graph.add_edge("write_agent_answer", END)
    return graph.compile()
