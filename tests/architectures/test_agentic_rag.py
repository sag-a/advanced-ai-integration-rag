import pytest
from rag_core.config import AppError, Settings
from rag_core.retrieval import RetrievalResult
from architectures.agentic_rag.tools import Research
from tests.architectures.test_graph_rag import documents


def test_outer_graph_allows_the_default_bounded_research_deadline():
    from rag_core.graph import build_graph
    from architectures.agentic_rag.options import options

    settings = Settings()
    graph = build_graph(None, None, settings)
    assert graph.step_timeout > options(settings).deadline_seconds


def test_read_only_tools_select_only_seen_ids_and_enforce_budgets():
    research = Research(
        Settings(extras={"max_searches": 1}),
        lambda q: RetrievalResult(documents()[:1], 0.01),
        documents(),
    )
    tools = {t.name: t for t in research.tools()}
    with pytest.raises(AppError, match="retrieved"):
        tools["select_evidence"].invoke({"chunk_ids": ["foreign"]})
    result = tools["search_pdf"].invoke({"query": "question"})
    assert result["passages"][0]["id"] == "0"
    tools["select_evidence"].invoke({"chunk_ids": ["0"]})
    assert research.selected == ["0"]
    with pytest.raises(AppError, match="budget"):
        tools["search_pdf"].invoke({"query": "another"})


def test_clause_reads_preserve_original_provenance():
    research = Research(Settings(), lambda q: RetrievalResult([], 0), documents())
    tools = {t.name: t for t in research.tools()}
    result = tools["read_clause"].invoke({"clause": "1"})
    assert result["passages"][0]["text"] == documents()[1].page_content
    assert result["passages"][0]["page_start"] == 1
    assert tools["read_clause"].invoke({"clause": "missing"})["passages"] == []


def request(name, args=None):
    from types import SimpleNamespace

    return SimpleNamespace(tool_call={"name": name, "args": args or {}, "id": "call"})


def test_guard_virtual_permissions_skill_gate_and_helper_boundaries():
    from architectures.agentic_rag.guard import ResearchGuard
    from langchain_core.messages import ToolMessage

    r = Research(Settings(), lambda q: RetrievalResult(documents()[:1], 0), documents())
    guard = ResearchGuard(r)

    def handler(req):
        return ToolMessage(content="skill text", tool_call_id="call")

    assert (
        guard.wrap_tool_call(request("search_pdf", {"query": "q"}), handler).status == "error"
    )
    guard.wrap_tool_call(
        request("read_file", {"file_path": "/skills/hearts/SKILL.md"}), handler
    )
    assert "/skills/hearts/SKILL.md" in guard.loaded_skills
    for name, args in [
        ("execute", {"command": "anything"}),
        ("read_file", {"file_path": "/etc/passwd"}),
        ("write_file", {"file_path": "/skills/hearts/SKILL.md"}),
        ("read_file", {"file_path": "/workspace/../../secret"}),
    ]:
        with pytest.raises(AppError):
            guard.wrap_tool_call(request(name, args), handler)
    helper = ResearchGuard(r, helper=True)
    with pytest.raises(AppError):
        helper.wrap_tool_call(request("task"), handler)
    with pytest.raises(AppError):
        helper.wrap_tool_call(request("select_evidence"), handler)


def test_service_failure_is_not_empty_evidence():

    def fail(q):
        raise RuntimeError("private connection details")

    r = Research(Settings(), fail, documents())
    with pytest.raises(AppError, match="service failed"):
        r.tools()[0].invoke({"query": "question"})
    with pytest.raises(AppError, match="service failed"):
        r.check()
    assert "private" not in r.failure


def test_model_budget_shared_by_researchers():
    from types import SimpleNamespace
    from architectures.agentic_rag.guard import ResearchGuard

    r = Research(
        Settings(extras={"max_model_calls": 1}), lambda q: RetrievalResult([], 0), documents()
    )
    req = SimpleNamespace(tools=[], override=lambda **kwargs: None)
    ResearchGuard(r).wrap_model_call(req, lambda req: SimpleNamespace(result=[]))
    with pytest.raises(AppError, match="model-call budget"):
        ResearchGuard(r, helper=True).wrap_model_call(
            req, lambda req: SimpleNamespace(result=[])
        )


def test_actual_deep_agent_reads_skill_and_selects_original_evidence(tmp_path, monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    import architectures.agentic_rag.retrieval as module
    from tests.architectures.test_graph_rag import Collection

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def call(name, args, i):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": i}])

    model = Model(
        responses=[
            call("read_file", {"file_path": "/skills/hearts/SKILL.md"}, "one"),
            call("search_pdf", {"query": "Hearts"}, "two"),
            call("select_evidence", {"chunk_ids": ["0"]}, "three"),
            AIMessage(content="Research complete"),
        ]
    )
    monkeypatch.setattr(
        module, "retrieve", lambda *args: RetrievalResult(documents()[:1], 0.01)
    )
    cfg = Settings()
    result = module.AgenticRetriever(Collection(documents()), cfg, model)("question")
    assert result.documents[0].page_content == documents()[0].page_content
    assert result.details["skills_read"] == ["/skills/hearts/SKILL.md"]
    assert result.details["model_calls"] == 4
    assert result.details["selected_ids"] == ["0"]


def test_missing_selection_fails_and_empty_selection_abstains(monkeypatch):
    from architectures.agentic_rag.retrieval import AgenticRetriever
    from tests.architectures.test_graph_rag import Collection

    class Agent:
        def invoke(self, *a, **k):
            return {}

    missing = AgenticRetriever(
        Collection(documents()), Settings(), None, factory=lambda m, r: Agent()
    )
    with pytest.raises(AppError, match="without selecting"):
        missing("q")

    def factory(m, r):
        r.selected = []
        return Agent()

    assert (
        AgenticRetriever(Collection(documents()), Settings(), None, factory=factory)(
            "unsupported"
        ).documents
        == []
    )


def test_unsupported_refusal_must_finalize_empty_evidence_without_research():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from architectures.agentic_rag.retrieval import AgenticRetriever
    from tests.architectures.test_graph_rag import Collection

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = Model(
        responses=[
            AIMessage(content="No account access."),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "select_evidence", "args": {"chunk_ids": []}, "id": "finalize"}
                ],
            ),
            AIMessage(content="Research complete"),
        ]
    )
    result = AgenticRetriever(Collection(documents()), Settings(), model)(
        "My live account balance?"
    )
    assert result.documents == []
    assert result.details["searches"] == 0
    assert result.details["model_calls"] == 3
    assert any((e["event"] == "selection_required" for e in result.details["events"]))
