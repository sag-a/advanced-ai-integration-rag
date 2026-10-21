from dataclasses import replace
from pathlib import Path
import shutil

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from rag_core.config import Settings, AppError
from rag_core.corpus import build_chunks, source_manifest
from rag_core.store import prepare
from tests.core.test_conversation import Chat, answer, passage, make_graph, turn
from tests.core.test_corpus import Tokenizer
from tests.core.test_store import Client


def test_multiple_documents_have_distinct_ids_and_physical_page_provenance(tmp_path):
    original = next(Settings().corpus.glob("*.pdf"))
    for name in ("first.pdf", "second.pdf"):
        shutil.copyfile(original, tmp_path / name)
    chunks = build_chunks(tmp_path, Tokenizer())
    assert {d.metadata["source"] for d in chunks} == {"first.pdf", "second.pdf"}
    assert len({d.id for d in chunks}) == len(chunks)
    assert len({d.metadata["corpus_id"] for d in chunks}) == 1
    assert [d.metadata["chunk_index"] for d in chunks] == list(range(len(chunks)))
    assert all(1 <= d.metadata["page_start"] <= d.metadata["page_end"] <= 5 for d in chunks)
    settings = replace(Settings(), corpus=tmp_path)
    client = Client()
    prepare(client, settings, chunks=chunks)
    writes = client.collection.writes
    prepare(client, settings, chunks=chunks)
    assert client.collection.writes == writes
    shutil.copyfile(original, tmp_path / "third.pdf")
    with pytest.raises(AppError, match="incompatible corpus_id"):
        prepare(client, settings, chunks=build_chunks(tmp_path, Tokenizer()))
    assert client.collection.writes == writes


def test_legacy_retrieval_tuple_is_rejected():
    graph = make_graph(Chat(), lambda q: ([passage()], 0))
    with pytest.raises(AppError, match="RetrievalResult"):
        turn(graph, "Question")


def test_optional_entry_can_continue_or_finish_without_core_architecture_branches():
    def continue_turn(state):
        return Command(goto="prepare_query")

    graph = make_graph(Chat(answer()), entry=continue_turn)
    assert turn(graph, "Question")["answer_status"] == "answered"

    def finish_turn(state):
        message = AIMessage(
            id="answer:" + state["messages"][-1].id,
            content="Cached, validated answer [1]",
            additional_kwargs={"result": {"answer_status": "answered", "source_map": []}},
        )
        return Command(
            goto="__end__", update={"messages": [message], "answer_status": "answered"}
        )

    graph = make_graph(
        Chat(), lambda q: pytest.fail("retrieval on fast path"), entry=finish_turn
    )
    result = turn(graph, "Question")
    assert result["messages"][-1].content.startswith("Cached")
    assert len(result["messages"]) == 2


def test_default_env_is_optional_but_shell_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shell-example")
    assert Settings().api_key() == "shell-example"
    Path(".env").write_text("ANTHROPIC_API_KEY=file-example\n")
    assert Settings().api_key() == "shell-example"


def test_empty_corpus_is_actionable(tmp_path):
    with pytest.raises(AppError, match="no PDF"):
        source_manifest(tmp_path)
