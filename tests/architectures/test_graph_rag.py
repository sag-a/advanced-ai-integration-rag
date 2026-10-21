from dataclasses import replace
import json
import pytest
from langchain_core.documents import Document
from rag_core.config import AppError, Settings
from rag_core.retrieval import RetrievalResult
from architectures.graph_rag.index import prepare, load_index, index_path
from architectures.graph_rag.schema import Extraction


def documents():
    return [
        Document(
            id=str(i),
            page_content=text,
            metadata={
                "chunk_index": i,
                "pdf_sha256": "pdf",
                "corpus_id": "corpus",
                "embedding_contract_id": "emb",
                "chunk_config_id": "chunk",
                "source": "test.pdf",
                "page_start": 1,
                "page_end": 1,
                "clause": str(i),
            },
        )
        for i, text in enumerate(
            ["A cannot transfer B.", "B expires in C.", "C lasts one year."]
        )
    ]


def extraction(i):
    labels = [("a", "b"), ("b", "c"), ("c", "d")][i]
    return {
        "entities": [
            {
                "id": x,
                "label": x.upper(),
                "type": "concept",
                "aliases": [x.upper(), "Καρδιές" if x == "b" else x],
            }
            for x in labels
        ],
        "relationships": [
            {
                "source": labels[0],
                "target": labels[1],
                "statement": documents()[i].page_content,
                "polarity": "negative" if i == 0 else "positive",
                "quote": documents()[i].page_content,
            }
        ],
    }


class Chat:
    def __init__(self, fail=None):
        self.calls = 0
        self.fail = fail

    def call(self, schema, system, payload):
        i = int(payload["chunk_id"])
        self.calls += 1
        if i == self.fail:
            raise AppError("interrupted")
        data = extraction(i)
        if schema.__name__ == "ChunkExtraction":
            data["relationships"] = [
                {
                    **{k: v for k, v in r.items() if k != "quote"},
                    "first_line": 1,
                    "last_line": 1,
                }
                for r in data["relationships"]
            ]
        return (schema.model_validate(data), {"usage": {"input_tokens": 1}})


def settings(tmp_path):
    return Settings(architecture="graph_rag", extras={"graph_index": tmp_path / "graph.json"})


def test_repeat_resume_and_identity(tmp_path):
    cfg = settings(tmp_path)
    with pytest.raises(AppError, match="interrupted"):
        prepare(documents(), cfg, Chat(fail=1))
    failures = list((tmp_path / "graph.json.cache").glob("*.failed.json"))
    assert len(failures) == 1
    assert len(json.loads(failures[0].read_text())["attempts"]) == 3
    with pytest.raises(AppError, match="resume"):
        prepare(documents(), cfg, Chat())
    chat = Chat()
    graph = prepare(documents(), cfg, chat, resume=True)
    assert chat.calls == 2
    assert graph["relationships"][0]["polarity"] == "negative"
    assert prepare(documents(), cfg, Chat())["fingerprint"] == graph["fingerprint"]
    assert load_index(documents(), cfg)["fingerprint"] == graph["fingerprint"]
    with pytest.raises(AppError, match="incompatible"):
        load_index(documents(), replace(cfg, model="claude-another"))


def test_reusing_a_complete_index_needs_no_chat_client(tmp_path, monkeypatch):
    import architectures.graph_rag as plugin

    cfg = settings(tmp_path)
    expected = prepare(documents(), cfg, Chat())
    monkeypatch.setattr(
        plugin,
        "AnthropicChat",
        lambda *args, **kwargs: pytest.fail("No model needed to reuse an index"),
    )
    result = plugin.prepare(cfg, Collection(documents()), None)
    assert result["fingerprint"] == expected["fingerprint"]


def test_invalid_extraction(tmp_path):
    from architectures.graph_rag.index import validate_extraction

    data = extraction(0)
    data["relationships"][0]["quote"] = "invented"
    with pytest.raises(AppError, match="quotation"):
        validate_extraction(Extraction.model_validate(data), documents()[0])
    data["relationships"][0]["quote"] = documents()[0].page_content
    data["relationships"][0]["target"] = "missing"
    with pytest.raises(AppError, match="endpoint"):
        validate_extraction(Extraction.model_validate(data), documents()[0])


def test_endpoint_feedback_identifies_the_missing_entity(tmp_path):
    from architectures.graph_rag.index import validate_extraction

    data = extraction(0)
    data["relationships"][0]["target"] = "undeclared_benefit"
    with pytest.raises(AppError, match="undeclared_benefit.*entities array"):
        validate_extraction(Extraction.model_validate(data), documents()[0])


def test_ambiguous_alias_and_atomic_complete(tmp_path):
    graph = prepare(documents(), settings(tmp_path), Chat())
    assert "καρδιέσ" not in graph["ambiguous_aliases"]
    assert json.loads((tmp_path / "graph.json").read_text())["complete"]
    assert not (tmp_path / "graph.json.tmp").exists()


def test_two_hops_cycles_and_bilingual_anchors(tmp_path):
    from architectures.graph_rag.retrieval import expand

    graph = prepare(documents(), settings(tmp_path), Chat())
    details = expand(graph, documents(), [documents()[0]], "Καρδιές", max_anchors=1)
    assert details["anchors"] == ["b"]
    assert "2" in details["candidate_ids"]
    assert len(details["relationships"]) == 3
    limited = expand(graph, documents(), [documents()[0]], "B", max_anchors=1, max_edges=1)
    assert len(limited["relationships"]) == 1
    assert len(limited["candidate_ids"]) <= 12
    graph["entities"]["a"]["aliases"].append("Καρδιές")
    graph["ambiguous_aliases"]["καρδιέσ"] = ["a", "b"]
    ambiguous = expand(graph, documents(), [documents()[0]], "Καρδιές", max_anchors=1)
    assert ambiguous["ambiguous_query_aliases"] == ["καρδιέσ"]


def test_fusion_retains_seed():
    from architectures.graph_rag.retrieval import fuse

    assert fuse(["seed", "x", "y"], ["y", "x"], ["seed"], 2) == ["seed", "y"]


def test_foreign_sources_and_modified_fingerprint_rejected(tmp_path):
    from rag_core.corpus import digest

    cfg = settings(tmp_path)
    graph = prepare(documents(), cfg, Chat())
    graph["relationships"][0]["chunk_id"] = "foreign"
    graph["fingerprint"] = digest(
        {k: graph[k] for k in ("contract", "entities", "relationships")}
    )
    index_path(cfg).write_text(json.dumps(graph))
    with pytest.raises(AppError, match="foreign chunk"):
        load_index(documents(), cfg)


def test_graph_explanation_stays_with_the_accepted_message(tmp_path):
    from tests.core.test_conversation import Chat as AnswerChat, answer, make_graph, turn

    cfg = settings(tmp_path)
    details = {
        "architecture": "graph_rag",
        "relationships": [{"polarity": "negative"}],
        "expansion_count": 1,
    }
    calls = []

    def retrieval(q):
        calls.append(q)
        return RetrievalResult(documents()[:1], 0.1, details)

    result = turn(make_graph(AnswerChat(answer()), retrieval, cfg), "Question", settings=cfg)
    assert result["observation"]["retrieval_details"] == details
    messages = result["messages"]
    assert (
        messages[-1].additional_kwargs["result"]["observation"]["retrieval_details"] == details
    )
    assert len(calls) == 1


class Collection:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []
        self.fail = False

    def get(self, **kwargs):
        return {
            "ids": [d.id for d in self.docs],
            "documents": [d.page_content for d in self.docs],
            "metadatas": [d.metadata for d in self.docs],
        }

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("unavailable")
        ids = kwargs["where"]["$and"][1]["chunk_id"]["$in"]
        return {"ids": [list(reversed(ids))], "distances": [[0.2] * len(ids)]}


def test_retriever_expansion_fallback_and_explicit_errors(tmp_path, monkeypatch):
    import architectures.graph_rag.retrieval as retrieval

    cfg = replace(settings(tmp_path), top_k=3)
    prepare(documents(), cfg, Chat())
    collection = Collection(documents())
    monkeypatch.setattr(
        retrieval, "retrieve", lambda *args: RetrievalResult(documents()[:1], 0.01)
    )
    retriever = retrieval.GraphRetriever(collection, cfg)
    result = retriever("Καρδιές")
    assert result.documents[0].id == "0"
    assert {d.id for d in result.documents} == {"0", "1", "2"}
    assert result.details["expansion_count"] == 2
    assert result.details["selected_expansion_count"] == 2
    assert all(
        (d.page_content == documents()[int(d.id)].page_content for d in result.documents)
    )
    assert collection.calls[0]["where"]["$and"][0] == {"corpus_id": {"$eq": "corpus"}}
    collection.fail = True
    with pytest.raises(AppError, match="candidate ranking failed"):
        retriever("B")
    collection.fail = False
    retriever.graph["relationships"] = []
    fallback = retriever("B")
    assert not fallback.details["graph_used"]
    assert "contributed no additional" in fallback.details["explanation"]
    monkeypatch.setattr(retrieval, "retrieve", lambda *args: RetrievalResult([], 0.01))
    assert retriever("empty").documents == []

    def fail(*args):
        raise AppError("Chroma unavailable")

    monkeypatch.setattr(retrieval, "retrieve", fail)
    with pytest.raises(AppError, match="unavailable"):
        retriever("B")


def test_cycles_alias_types_and_candidate_limits(tmp_path):
    from architectures.graph_rag.retrieval import expand

    graph = prepare(documents(), settings(tmp_path), Chat())
    graph["relationships"].append(
        {**graph["relationships"][0], "id": "cycle", "source": "c", "target": "a"}
    )
    details = expand(graph, documents(), documents()[:1], "B", max_anchors=1, max_candidates=2)
    assert details["anchors"] == ["b"]
    assert len(details["candidate_ids"]) == 2
    assert len({r["id"] for r in details["relationships"]}) == len(details["relationships"])
    assert all((r["hop"] <= 2 for r in details["relationships"]))
    from architectures.graph_rag.index import assemble, contract

    rows = [{"extraction": extraction(i), "observation": {}} for i in range(3)]
    rows[1]["extraction"]["entities"][0]["type"] = "organization"
    with pytest.raises(AppError, match="incompatible types"):
        assemble(documents(), contract(documents(), settings(tmp_path)), rows)
    rows[1]["extraction"]["entities"][0]["type"] = "concept"
    rows[1]["extraction"]["entities"][1]["aliases"].append("Καρδιές")
    graph = assemble(documents(), contract(documents(), settings(tmp_path)), rows)
    assert graph["ambiguous_aliases"]["καρδιέσ"] == ["b", "c"]


def test_graph_tracing_failure_preserves_retrieval(tmp_path, monkeypatch):
    import architectures.graph_rag.retrieval as retrieval
    import architectures.graph_rag.tracing as tracing

    cfg = replace(settings(tmp_path), langfuse=True)
    prepare(documents(), replace(cfg, langfuse=False), Chat())
    monkeypatch.setattr(
        tracing, "get_telemetry", lambda *args: (_ for _ in ()).throw(RuntimeError("secret"))
    )
    monkeypatch.setattr(
        retrieval, "retrieve", lambda *args: RetrievalResult(documents()[:1], 0.01)
    )
    result = retrieval.GraphRetriever(Collection(documents()), cfg)("B")
    assert len(result.documents) == 3
    assert len(result.details["trace_warnings"]) == 3
    assert "secret" not in str(result.details)


def test_streamed_graph_answer_uses_original_passages_and_clears_stale_details(tmp_path):
    from rag_core.graph import run_turn
    from tests.core.test_streaming import StreamingChat
    from tests.core.test_conversation import answer, make_graph, turn

    cfg = settings(tmp_path)
    detail = {"architecture": "graph_rag", "statement": "EXTRACTED-NAVIGATION-ONLY"}
    chat = StreamingChat(
        answer("Validated answer"), {"search_query": "", "clarification": "Which subject?"}
    )
    graph = make_graph(chat, lambda q: RetrievalResult(documents()[:1], 0, detail), cfg)
    events = list(run_turn(graph, "Question", [], cfg, streaming=True))
    assert any(e["type"] == "draft" for e in events)
    state = events[-1]["data"]
    assert (
        state["messages"][-1].additional_kwargs["result"]["observation"]["retrieval_details"]
        == detail
    )
    assert "EXTRACTED-NAVIGATION-ONLY" not in str(chat.calls[0][1])
    assert chat.calls[0][1]["evidence"][0]["text"] == documents()[0].page_content
    state = turn(graph, "And that?", state["messages"], settings=cfg)
    assert state["answer_status"] == "clarification_needed"
    assert "retrieval_details" not in state["observation"] and len(state["messages"]) == 4


def test_trace_initialization_failure_does_not_invalidate_graph_conversation(
    tmp_path, monkeypatch
):
    from tests.core.test_conversation import Chat as AnswerChat, answer, make_graph, turn

    def fail(*args):
        raise ValueError("private initialization detail")

    monkeypatch.setattr("rag_core.observability.get_telemetry", fail)
    cfg = replace(settings(tmp_path), langfuse=True)
    graph = make_graph(AnswerChat(answer()), lambda q: RetrievalResult(documents()[:1], 0), cfg)
    state = turn(graph, "Question", settings=cfg)
    assert state["answer_status"] == "answered"
    assert (
        "initialization failed" in state["trace_warning"]
        and "private" not in state["trace_warning"]
    )
