from dataclasses import replace
from io import StringIO

import httpx
import pytest
from langchain_core.documents import Document
from rich.console import Console

from architectures.jev_rag import jev, retriever
from rag_core.config import AppError, Settings
from rag_core.retrieval import RetrievalResult
from tests.core.test_conversation import Chat, make_graph, turn


def chunk(identifier, distance):
    return Document(
        id=identifier,
        page_content=f"Text of clause {identifier}.",
        metadata={"source": "t.pdf", "page_start": 1, "page_end": 1, "clause": identifier, "distance": distance},
    )


CANDIDATES = [chunk("1.1", 0.2), chunk("1.2", 0.3), chunk("1.3", 0.4)]


@pytest.fixture
def setup(monkeypatch):
    calls = {"gate": []}
    terminal = StringIO()
    monkeypatch.setattr(retriever, "retrieve", lambda *args: RetrievalResult(CANDIDATES, 0.01))
    monkeypatch.setattr(retriever, "CONSOLE", Console(file=terminal, width=200))
    monkeypatch.setattr(retriever.reranker, "score", lambda q, texts: [0.0, 1.0, 2.0])
    monkeypatch.setattr(retriever.jev, "relevance", lambda s, q, texts: ([0.9, 0.1, 0.5], 0.0003))

    def gate(settings, question, passages):
        calls["gate"].append(passages)
        return calls.get("probability", 0.9), 0.0001

    monkeypatch.setattr(retriever.jev, "gate", gate)
    return calls, terminal


def settings(**extras):
    return replace(Settings(), architecture="jev_rag", top_k=2, extras=extras)


def test_bge_reranks_then_the_gate_sees_only_the_kept_passages(setup):
    calls, terminal = setup
    result = retriever.make_retriever(settings(), object())("Question")
    assert [d.id for d in result.documents] == ["1.3", "1.2"]
    assert calls["gate"] == [["Text of clause 1.3.", "Text of clause 1.2."]]
    assert result.details["gate_passed"] and result.details["reranker"] == "bge"
    assert "Jev gate: P(these 2 passages answer the question, at least in part) = 0.90" in terminal.getvalue()


def test_jev_can_be_the_reranker(setup):
    result = retriever.make_retriever(settings(reranker="jev"), object())("Question")
    assert [d.id for d in result.documents] == ["1.1", "1.3"]
    assert result.details["jev_cost_usd"] == 0.0004


def test_a_low_gate_returns_no_passages_and_the_answer_model_is_never_called(setup):
    calls, terminal = setup
    calls["probability"] = 0.03
    search = retriever.make_retriever(settings(), object())
    state = turn(make_graph(Chat(), search, settings()), "What is my balance?", settings=settings())
    assert state["answer_status"] == "insufficient_evidence"
    assert state["documents"] == []
    assert "abstain: not in the documents" in terminal.getvalue()


@pytest.mark.parametrize(
    "extras, message",
    [({"reranker": "cohere"}, "reranker=bge"), ({"gate_threshold": 1.5}, "gate_threshold"), ({"gate_threshold": True}, "gate_threshold")],
)
def test_invalid_options_fail_before_any_search(extras, message):
    with pytest.raises(AppError, match=message):
        retriever.make_retriever(settings(**extras), object())


def test_decide_sends_typed_questions_and_reads_the_probability(monkeypatch):
    sent = {}

    def post(url, headers, json, timeout):
        sent.update(url=url, auth=headers["Authorization"], body=json)
        return httpx.Response(
            200,
            json={"answers": {"answerable": {"type": "noul", "noul": 0.81}}, "usage": {"cost": 2e-5}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(jev.httpx, "post", post)
    probability, cost = jev.gate(Settings(), "Q?", ["first", "second"])
    assert (probability, cost) == (0.81, 2e-5)
    assert sent["url"].endswith("/api/alpha/decisions") and sent["auth"] == "Bearer test-key"
    assert sent["body"]["model"] == "typesafe/jev-1.14"
    assert sent["body"]["state"] == ["Question: Q?", "Passage 1: first", "Passage 2: second"]
    assert sent["body"]["questions"]["answerable"]["type"] == "noul"


def test_http_errors_become_actionable_and_never_show_the_key(monkeypatch):
    def post(url, **kwargs):
        return httpx.Response(402, json={"error": "no credits"}, request=httpx.Request("POST", url))

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key-value")
    monkeypatch.setattr(jev.httpx, "post", post)
    with pytest.raises(AppError, match="HTTP 402") as error:
        jev.gate(Settings(), "Q?", ["p"])
    assert "secret-key-value" not in str(error.value)


@pytest.fixture
def routed(setup, monkeypatch):
    calls, terminal = setup
    searched = []
    targets = {
        route: (replace(Settings(), architecture="jev_rag", top_k=2, collection=f"{route}_collection"), f"{route}-handle")
        for route in retriever.ROUTES
    }
    monkeypatch.setattr(retriever, "open_routes", lambda settings: targets)

    def fake_retrieve(collection, settings, question):
        searched.append((collection, settings.collection))
        return RetrievalResult(CANDIDATES, 0.01)

    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)

    def route(settings, question, descriptions):
        calls["descriptions"] = descriptions
        return calls["choice"], {calls["choice"]: 0.9}, 0.00002

    monkeypatch.setattr(retriever.jev, "route", route)
    return calls, terminal, searched


def test_the_router_searches_only_the_chosen_collection(routed):
    calls, terminal, searched = routed
    calls["choice"] = "tariffs"
    result = retriever.make_retriever(settings(route=True), "default-handle")("Double Play price?")
    assert searched == [("tariffs-handle", "tariffs_collection")]
    assert result.details["route"] == "tariffs" and result.details["collection"] == "tariffs_collection"
    assert set(calls["descriptions"]) == {"magenta", "tariffs", "neither"}
    assert "Jev router: tariffs 0.90 → tariffs" in terminal.getvalue()


def test_neither_abstains_before_retrieval_and_the_answer_model(routed):
    calls, terminal, searched = routed
    calls["choice"] = "neither"
    search = retriever.make_retriever(settings(route=True), "default-handle")
    state = turn(make_graph(Chat(), search, settings(route=True)), "Weather?", settings=settings(route=True))
    assert state["answer_status"] == "insufficient_evidence"
    assert searched == [] and calls["gate"] == []
    assert "abstain before retrieval" in terminal.getvalue()


def test_route_option_must_be_a_boolean():
    with pytest.raises(AppError, match="route=true"):
        retriever.make_retriever(settings(route="yes"), object())


def test_route_sends_a_choice_question_and_returns_the_distribution(monkeypatch):
    sent = {}

    def post(url, headers, json, timeout):
        sent.update(json)
        answer = {"type": "choice", "choice": "magenta", "probabilities": {"magenta": 0.91, "neither": 0.09}}
        return httpx.Response(200, json={"answers": {"route": answer}, "usage": {"cost": 1.6e-5}}, request=httpx.Request("POST", url))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(jev.httpx, "post", post)
    choice, probabilities, cost = jev.route(Settings(), "Q?", {"magenta": "loyalty terms", "neither": "other"})
    assert (choice, probabilities["magenta"], cost) == ("magenta", 0.91, 1.6e-5)
    assert sent["questions"]["route"]["type"] == "choice"
    assert sent["questions"]["route"]["criteria"] == {"magenta": "loyalty terms", "neither": "other"}
