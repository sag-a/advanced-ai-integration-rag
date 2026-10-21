from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import pytest
from langchain_core.messages import HumanMessage
from rag_core.config import AppError, Settings
from rag_core.observability import credentials, scrub, trace_metadata
from rag_core.graph import run_turn
from tests.core.test_conversation import Chat, answer, make_graph, turn


def test_telemetry_does_not_hash_mutable_architecture_options(monkeypatch):
    import rag_core.observability as observability

    created = []
    monkeypatch.setattr(
        observability, "Telemetry", lambda settings: created.append(settings) or object()
    )
    settings = replace(Settings(), extras={"max_searches": 3})
    observability.get_telemetry(settings)
    assert created == [settings]


def test_langfuse_configuration_prefers_shell_without_mutating_it(tmp_path, monkeypatch):
    p = tmp_path / "chosen.env"
    p.write_text(
        "LANGFUSE_PUBLIC_KEY=pk-lf-example\nLANGFUSE_SECRET_KEY=example-secret\nLANGFUSE_HOST=https://cloud.langfuse.com\n"
    )
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "shell-secret")
    settings = replace(Settings(), env_file=p)
    config = credentials(settings)
    assert config["secret_key"] == "shell-secret"
    assert config["base_url"] == "https://cloud.langfuse.com"
    monkeypatch.delenv("LANGFUSE_SECRET_KEY")
    assert credentials(settings)["secret_key"] == "example-secret"
    p.write_text("LANGFUSE_PUBLIC_KEY=pk-lf-example\n")
    with pytest.raises(AppError, match="LANGFUSE"):
        credentials(settings)


def test_trace_payload_redacts_credentials_and_bounds_history():
    data = {
        "api_key": "private",
        "authorization": "Bearer private",
        "messages": [HumanMessage(content="a" * 3000)]
        + [HumanMessage(content=f"Recent {n}") for n in range(5)]
        + [HumanMessage(content="Current")],
        "text": "secret-example sk-ant-example-credential",
        "metadata": {"checkpoint_ns": "internal"},
    }
    cleaned = scrub(data, ("secret-example",))
    assert "private" not in str(cleaned)
    assert "secret-example" not in str(cleaned)
    assert "sk-ant-example-credential" not in str(cleaned)
    assert "a" * 3000 not in str(cleaned)
    assert len(cleaned["messages"]) == 6
    assert "Current" in str(cleaned)
    assert "checkpoint_ns" not in str(cleaned)
    assert "ANTHROPIC_API_KEY" not in str(trace_metadata(Settings()))


class FakeTelemetry:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail
        self.outputs = []

    @contextmanager
    def turn(self, question, history, thread_id, message_id):
        self.calls += 1
        yield SimpleNamespace(
            reference={
                "trace_id": "trace",
                "trace_url": "https://cloud.langfuse.com/project/demo/traces/trace",
            },
            callbacks=[],
        )

    def finish(self, trace, result):
        self.outputs.append(result["messages"][-1].content)

    def flush(self):
        return "Langfuse export failed." if self.fail else None


@pytest.mark.parametrize("streaming", [False, True])
def test_trace_reference_stays_in_caller_owned_messages(streaming):
    from tests.core.test_streaming import StreamingChat
    from tests.core.test_conversation import make_graph

    telemetry = FakeTelemetry(fail=True)
    settings = Settings()
    chat = StreamingChat(answer()) if streaming else Chat(answer())
    events = list(
        run_turn(
            make_graph(chat), "Question", [], settings, streaming=streaming, telemetry=telemetry
        )
    )
    result = events[-1]["data"]
    assert result["trace_warning"]
    if streaming:
        assert any(e["type"] == "warning" for e in events)
    messages = result["messages"]
    assert messages[-1].additional_kwargs["result"]["trace_id"] == "trace"
    assert messages[-1].additional_kwargs["result"]["trace_url"].endswith("/trace")
    assert telemetry.calls == 1


def test_sdk_callback_captures_nodes_and_nested_model_calls(tmp_path, monkeypatch):
    from uuid import uuid4
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from rag_core.observability import Telemetry

    exporter = InMemorySpanExporter()
    public_key = "pk-lf-" + uuid4().hex
    client = Langfuse(public_key=public_key, secret_key="example-only", span_exporter=exporter)
    monkeypatch.setattr(
        client,
        "get_trace_url",
        lambda **kwargs: "https://cloud.langfuse.com/project/demo/traces/test",
    )
    settings = Settings()
    telemetry = Telemetry(settings, client=client, public_key=public_key)

    class NestedChat(Chat):
        def call(self, schema, system, payload):
            FakeListChatModel(responses=["Nested model"]).invoke("safe prompt")
            return super().call(schema, system, payload)

    graph = make_graph(NestedChat(answer()))
    result = turn(
        graph,
        "Question",
        settings=settings,
        thread_id="test-session",
        message_id="one",
        telemetry=telemetry,
    )
    client.flush()
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "rag-turn" in names
    assert all(
        (
            name in names
            for name in ["prepare_query", "retrieve", "generate_answer", "FakeListChatModel"]
        )
    )
    assert len({s.context.trace_id for s in spans}) == 1
    assert result["messages"][-1].additional_kwargs["result"]["trace_url"]


def test_tracing_setup_failure_does_not_break_the_answer(tmp_path):

    class Broken(FakeTelemetry):
        @contextmanager
        def turn(self, *args):
            raise ConnectionError("private detail")
            yield

    settings = Settings()
    result = turn(make_graph(Chat(answer())), "Question", settings=settings, telemetry=Broken())
    assert result["answer_status"] == "answered" and result["trace_warning"]
    assert "private detail" not in result["trace_warning"]
