"""Streaming must preserve the buffered path's grounding and retry guarantees."""

import json

import pytest
from langchain_core.messages import AIMessageChunk

from rag_core.config import AppError
from rag_core.graph import Answer
from rag_core.model import AnthropicChat
from tests.core.test_conversation import Chat, answer, make_graph, turn
from rag_core.graph import run_turn
from rag_core.config import Settings


class StreamingChat(Chat):
    def stream(self, schema, system, payload, on_draft):
        on_draft("Live draft")
        return self.call(schema, system, payload)


def test_draft_precedes_final_and_only_final_is_retained():
    graph = make_graph(StreamingChat(answer("Final supported answer")))
    history, seen = [], []
    for event in run_turn(graph, "Question", history, Settings(), streaming=True):
        seen.append(event)
        if event["type"] == "draft":
            assert history == []
    assert [e["data"] for e in seen if e["type"] == "progress"] == [
        "Preparing question",
        "Searching PDF",
        "Generating answer",
    ]
    assert seen[-1]["type"] == "final"
    accepted = seen[-1]["data"]["messages"][-1]
    assert "Final supported answer" in accepted.content and "Live draft" not in accepted.content
    assert accepted.additional_kwargs["result"]["source_map"][0]["text"]


@pytest.mark.parametrize("response", [answer(sources=[999]), AppError("Stream interrupted")])
def test_failed_stream_has_no_final_and_can_be_retried(response):
    graph = make_graph(StreamingChat(response, answer()))
    events, history = [], []
    with pytest.raises(AppError):
        for event in run_turn(graph, "Question", history, Settings(), streaming=True):
            events.append(event)
    assert any(e["type"] == "draft" for e in events)
    assert not any(e["type"] == "final" for e in events) and history == []
    result = list(run_turn(graph, "Question", history, Settings(), streaming=True))[-1]["data"]
    assert len(result["messages"]) == 2


class ChunkModel:
    def __init__(self, arguments, reason="tool_use", fail=False):
        self.arguments, self.reason, self.fail = arguments, reason, fail

    def bind_tools(self, *args, **kwargs):
        return self

    def stream(self, messages):
        yield AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "Answer", "id": "call", "index": 0, "args": ""}],
        )
        for part in self.arguments:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": None, "id": None, "index": 0, "args": part}],
            )
        if self.fail:
            raise ConnectionError("private provider detail")
        yield AIMessageChunk(content="", response_metadata={"stop_reason": self.reason})


def model(arguments, **kwargs):
    chat = AnthropicChat.__new__(AnthropicChat)
    chat.model = ChunkModel(arguments, **kwargs)
    return chat


def test_structured_stream_exposes_only_incremental_answer_text():
    drafts = []
    result, usage = model(
        [
            '{"answer":"Οι Καρ',
            'διές \\"λήγουν\\"',
            '","sources":[1],"status":"answered"}',
        ]
    ).stream(Answer, "system", {}, drafts.append)
    assert drafts[0] == "Οι Καρ"
    assert drafts[-1] == 'Οι Καρδιές "λήγουν"'
    assert result.answer == drafts[-1]
    assert usage["elapsed_seconds"] >= 0
    assert all("sources" not in text for text in drafts)


@pytest.mark.parametrize(
    "arguments,kwargs",
    [
        ([json.dumps(answer())], {"reason": "max_tokens"}),
        (['{"answer":"partial"'], {}),
        ([json.dumps(answer())], {"fail": True}),
    ],
)
def test_malformed_or_interrupted_structured_stream_is_rejected(arguments, kwargs):
    with pytest.raises(AppError) as error:
        model(arguments, **kwargs).stream(Answer, "system", {}, lambda text: None)
    assert "private provider detail" not in str(error.value)


def test_configured_anthropic_wrapper_really_uses_streaming(tmp_path, monkeypatch):
    from dataclasses import replace
    from langchain_anthropic import ChatAnthropic
    from langchain_core.outputs import ChatGenerationChunk
    from rag_core.config import Settings

    def stream(self, messages, **kwargs):
        for chunk in ChunkModel([json.dumps(answer())]).stream(messages):
            yield ChatGenerationChunk(message=chunk)

    def buffered_request(*args, **kwargs):
        raise AssertionError("Streaming silently fell back to buffered generation")

    monkeypatch.setattr(ChatAnthropic, "_stream", stream)
    monkeypatch.setattr(ChatAnthropic, "_generate", buffered_request)
    key_file = tmp_path / "example.env"
    key_file.write_text("ANTHROPIC_API_KEY=example-only\n")
    chat = AnthropicChat(replace(Settings(), env_file=key_file))
    drafts = []
    result, _ = chat.stream(Answer, "system", {}, drafts.append)
    assert result.answer == "Supported" and drafts == ["Supported"]


def test_streamed_follow_up_and_buffered_turn_use_the_same_message_list():
    chat = StreamingChat(
        answer("Hearts"),
        {"search_query": "Πότε λήγουν οι Καρδιές;", "clarification": ""},
        answer("Ένα έτος"),
        {"search_query": "", "clarification": "Hearts or coupons?"},
    )
    graph = make_graph(chat)
    first = turn(graph, "What are Hearts?")
    second = list(
        run_turn(graph, "Πότε λήγουν;", first["messages"], Settings(), streaming=True)
    )[-1]["data"]
    assert second["search_query"] == "Πότε λήγουν οι Καρδιές;"
    result = turn(graph, "What about them?", second["messages"])
    assert result["answer_status"] == "clarification_needed" and len(result["messages"]) == 6


def test_graph_delivers_draft_while_generation_is_still_running():
    from threading import Event, Thread

    wrote, release, received = Event(), Event(), Event()
    failures = []

    class PausedChat(StreamingChat):
        def stream(self, schema, system, payload, on_draft):
            on_draft("Early draft")
            wrote.set()
            assert release.wait(5), "Test did not release generation"
            return self.call(schema, system, payload)

    graph = make_graph(PausedChat(answer()))

    def consume():
        try:
            for event in run_turn(graph, "Question", [], Settings(), streaming=True):
                if event["type"] == "draft":
                    received.set()
        except BaseException as error:
            failures.append(error)

    worker = Thread(target=consume)
    worker.start()
    try:
        assert wrote.wait(3)
        assert received.wait(1), "Draft was buffered until generation returned"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not failures


def test_anthropic_stream_requests_unbuffered_tool_arguments():
    from langchain_anthropic import ChatAnthropic
    from langchain_core.outputs import ChatGenerationChunk
    from rag_core.config import Settings

    captured = []

    def stream(self, messages, **kwargs):
        captured.extend(kwargs["tools"])
        for chunk in ChunkModel([json.dumps(answer())]).stream(messages):
            yield ChatGenerationChunk(message=chunk)

    from unittest.mock import patch

    with (
        patch.object(ChatAnthropic, "_stream", stream),
        patch.object(Settings, "api_key", return_value="example-only"),
    ):
        chat = AnthropicChat(Settings())
        chat.stream(Answer, "system", {}, lambda _: None)
    assert captured[0].get("eager_input_streaming") is True
