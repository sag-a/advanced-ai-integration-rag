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
        data["relationships"] = [
            {**{k: v for k, v in r.items() if k != "quote"}, "first_line": 1, "last_line": 1}
            for r in data["relationships"]
        ]
        return (schema.model_validate(data), {"usage": {"input_tokens": 1}})


def settings(tmp_path):
    return Settings(architecture="graph_rag", extras={"graph_index": tmp_path / "graph.json"})


class Collection:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def get(self, **kwargs):
        return {
            "ids": [d.id for d in self.docs],
            "documents": [d.page_content for d in self.docs],
            "metadatas": [d.metadata for d in self.docs],
        }

    def query(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["where"]["$and"][1]["chunk_id"]["$in"]
        return {"ids": [list(reversed(ids))], "distances": [[0.2] * len(ids)]}


def test_prepare_builds_once_and_copies_quotes(tmp_path):
    cfg = settings(tmp_path)
    chat = Chat()
    graph = prepare(documents(), cfg, chat)
    assert chat.calls == 3
    assert len(graph["entities"]) == 4 and len(graph["relationships"]) == 3
    assert graph["relationships"][0]["polarity"] == "negative"
    assert graph["relationships"][0]["quote"] == documents()[0].page_content
    assert graph["entities"]["b"]["chunk_ids"] == ["0", "1"]
    assert json.loads(index_path(cfg).read_text(encoding="utf-8"))["complete"]
    again = Chat()
    assert prepare(documents(), cfg, again)["fingerprint"] == graph["fingerprint"]
    assert again.calls == 0


def test_failed_extraction_retries_then_stops(tmp_path):
    chat = Chat(fail=1)
    with pytest.raises(AppError, match="interrupted"):
        prepare(documents(), settings(tmp_path), chat)
    assert chat.calls == 1 + 3
    assert not index_path(settings(tmp_path)).exists()


def test_reusing_a_complete_index_needs_no_chat_client(tmp_path, monkeypatch):
    import architectures.graph_rag as plugin

    cfg = settings(tmp_path)
    expected = prepare(documents(), cfg, Chat())
    monkeypatch.setattr(
        plugin, "AnthropicChat", lambda *args, **kwargs: pytest.fail("No model needed to reuse an index")
    )
    result = plugin.prepare(cfg, Collection(documents()), None)
    assert result["fingerprint"] == expected["fingerprint"]


def test_load_index_refuses_stale_or_forged_graphs(tmp_path):
    cfg = settings(tmp_path)
    with pytest.raises(AppError, match="missing"):
        load_index(documents(), cfg)
    graph = prepare(documents(), cfg, Chat())
    with pytest.raises(AppError, match="incompatible"):
        load_index(documents(), replace(cfg, model="claude-another"))
    graph["relationships"][0]["quote"] = "invented"
    index_path(cfg).write_text(json.dumps(graph), encoding="utf-8")
    with pytest.raises(AppError, match="quotes text"):
        load_index(documents(), cfg)


def test_invalid_extraction():
    from architectures.graph_rag.index import validate_extraction

    data = extraction(0)
    data["relationships"][0]["quote"] = "invented"
    with pytest.raises(AppError, match="quotation"):
        validate_extraction(Extraction.model_validate(data), documents()[0])
    data["relationships"][0]["quote"] = documents()[0].page_content
    data["relationships"][0]["target"] = "undeclared_benefit"
    with pytest.raises(AppError, match="undeclared_benefit.*entities array"):
        validate_extraction(Extraction.model_validate(data), documents()[0])


def test_assemble_rejects_one_entity_with_two_types(tmp_path):
    from architectures.graph_rag.index import assemble, contract

    extractions = [Extraction.model_validate(extraction(i)) for i in range(3)]
    extractions[1].entities[0].type = "organization"
    with pytest.raises(AppError, match="incompatible types"):
        assemble(documents(), contract(documents(), settings(tmp_path)), extractions)


def retriever(tmp_path, monkeypatch, seeds, top_k=3):
    import architectures.graph_rag.retrieval as retrieval

    cfg = replace(settings(tmp_path), top_k=top_k)
    prepare(documents(), cfg, Chat())
    monkeypatch.setattr(retrieval, "retrieve", lambda *args: RetrievalResult(seeds, 0.01))
    return retrieval.GraphRetriever(Collection(documents()), cfg)


def test_start_entities_prefer_names_in_the_question(tmp_path, monkeypatch):
    graph = retriever(tmp_path, monkeypatch, documents()[:1])
    assert graph.start_entities(documents()[:1], "Καρδιές") == ["b", "a"]
    assert graph.start_entities(documents()[:1], "something else") == ["a", "b"]


def test_walk_follows_two_hops_within_the_fact_limit(tmp_path, monkeypatch):
    import architectures.graph_rag.retrieval as retrieval

    graph = retriever(tmp_path, monkeypatch, documents()[:1])
    facts = graph.walk(["a"])
    assert [(f["id"], f["hop"]) for f in facts] == [("r0", 1), ("r1", 2)]
    monkeypatch.setattr(retrieval, "MAX_FACTS", 1)
    assert len(graph.walk(["a", "b"])) == 1


def test_retriever_keeps_seeds_first_then_adds_graph_passages(tmp_path, monkeypatch):
    graph = retriever(tmp_path, monkeypatch, documents()[:1])
    result = graph("Καρδιές")
    assert [d.id for d in result.documents] == ["0", "2", "1"]
    assert result.details["expansion_count"] == 2
    assert result.details["selected_expansion_count"] == 2
    assert result.details["graph_used"]
    assert all(d.page_content == documents()[int(d.id)].page_content for d in result.documents)
    assert graph.collection.calls[0]["where"]["$and"][0] == {"corpus_id": {"$eq": "corpus"}}


def test_retriever_falls_back_to_vector_results(tmp_path, monkeypatch):
    graph = retriever(tmp_path, monkeypatch, documents()[:1])
    graph.graph["relationships"] = []
    result = graph("B")
    assert [d.id for d in result.documents] == ["0"]
    assert not result.details["graph_used"]
    assert "contributed no additional" in result.details["explanation"]
    empty = retriever(tmp_path, monkeypatch, [])
    assert empty("anything").documents == []


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
