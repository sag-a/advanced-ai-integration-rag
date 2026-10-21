from dataclasses import replace

import pytest

from rag_core.config import Settings
from rag_core.registry import Architecture, REGISTRY, load_all, register
from rag_core.retrieval import RetrievalResult


def test_registry_load_is_repeatable_and_duplicate_names_fail():
    load_all()
    before = dict(REGISTRY)
    load_all()
    assert REGISTRY == before
    assert {"classic_rag", "graph_rag", "agentic_rag"} <= REGISTRY.keys()
    with pytest.raises(ValueError, match="already registered"):
        register(REGISTRY["classic_rag"])


def test_classic_retriever_returns_the_shared_contract(monkeypatch):
    from architectures.classic_rag import retriever

    monkeypatch.setattr(retriever, "retrieve", lambda *args: RetrievalResult([], 0))
    load_all()
    settings = replace(Settings(), architecture="classic_rag")
    search = REGISTRY[settings.architecture].make_retriever(settings, object(), None)
    result = search("Question without expected facts")
    assert isinstance(result, RetrievalResult)
    assert result.details["architecture"] == settings.architecture


def test_registration_marks_paid_retrieval_and_owns_details():
    load_all()
    assert REGISTRY["agentic_rag"].needs_llm_for_retrieval
    assert not REGISTRY["classic_rag"].needs_llm_for_retrieval
    assert REGISTRY["graph_rag"].prepare
    assert REGISTRY["graph_rag"].render_details


def test_registration_can_be_extended_without_app_changes(monkeypatch):
    prototype = Architecture(
        "test_retriever", "Test", lambda *args: lambda q: RetrievalResult([], 0)
    )
    monkeypatch.setitem(REGISTRY, prototype.name, prototype)
    assert isinstance(
        REGISTRY[prototype.name].make_retriever(None, None, None)("Q"), RetrievalResult
    )


def test_missing_extra_is_reported_but_programming_errors_are_not_hidden(monkeypatch):
    import rag_core.registry as registry
    from rag_core.config import AppError

    original = registry.import_module
    monkeypatch.setattr(registry, "REGISTRY", dict(registry.REGISTRY))
    monkeypatch.setattr(registry, "UNAVAILABLE", {})

    def missing(name):
        if name == "architectures.graph_rag":
            raise registry.MissingExtra("Install the graph extra")
        return original(name)

    monkeypatch.setattr(registry, "import_module", missing)
    registry.load_all()
    assert "graph_rag" not in registry.REGISTRY
    with pytest.raises(AppError, match="graph extra"):
        registry.get("graph_rag")

    def broken(name):
        if name == "architectures.graph_rag":
            raise ValueError("Programming error")
        return original(name)

    monkeypatch.setattr(registry, "import_module", broken)
    with pytest.raises(ValueError, match="Programming error"):
        registry.load_all()
