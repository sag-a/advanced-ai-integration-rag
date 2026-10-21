"""Follow-ups need a caller-owned message list, not a conversation database."""

import pytest

from rag_core.config import AppError, Settings
from rag_core.retrieval import RetrievalResult
from tests.core.test_conversation import Chat, answer, passage


def test_plain_message_list_supports_followups_without_mutating_input():
    from rag_core.graph import build_graph, run_turn

    chat = Chat(answer(), {"search_query": "Hearts expiry?", "clarification": ""}, answer())
    graph = build_graph(chat, lambda q: RetrievalResult([passage()], 0), Settings())
    history = []
    first = list(run_turn(graph, "What are Hearts?", history, Settings()))[-1]["data"]
    assert history == []
    history = first["messages"]
    second = list(run_turn(graph, "When do they expire?", history, Settings()))[-1]["data"]
    assert len(history) == 2
    assert len(second["messages"]) == 4
    assert second["search_query"] == "Hearts expiry?"
    assert chat.calls[1][1]["history"][0]["text"] == "What are Hearts?"


def test_stream_failure_leaves_caller_history_unchanged_and_retry_is_explicit():
    from rag_core.graph import build_graph, run_turn
    from tests.core.test_streaming import StreamingChat

    chat = StreamingChat(answer(sources=[999]), answer())
    graph = build_graph(chat, lambda q: RetrievalResult([passage()], 0), Settings())
    history, events = [], []
    with pytest.raises(AppError, match="citation"):
        for event in run_turn(graph, "Question", history, Settings(), streaming=True):
            events.append(event)
    assert history == []
    assert any(e["type"] == "draft" for e in events)
    assert not any(e["type"] == "final" for e in events)
    result = list(run_turn(graph, "Question", history, Settings(), streaming=True))[-1]["data"]
    assert len(result["messages"]) == 2
    assert "Live draft" not in result["messages"][-1].content


def test_independent_message_lists_have_no_shared_history():
    from rag_core.graph import build_graph, run_turn

    chat = Chat(answer(), answer())
    graph = build_graph(chat, lambda q: RetrievalResult([passage()], 0), Settings())
    list(run_turn(graph, "First browser", [], Settings()))
    other = list(run_turn(graph, "Second browser", [], Settings()))[-1]["data"]
    assert len(other["messages"]) == 2
    assert len(chat.calls) == 2
