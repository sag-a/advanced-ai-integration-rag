import pytest
from langchain_core.messages import AIMessage
from rag_core.config import Settings, AppError
from rag_core.model import AnthropicChat
from rag_core.graph import Answer


def test_missing_key_is_actionable_and_selected_env_does_not_mutate_shell(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = replace(Settings(), env_file=tmp_path / "missing.env")
    with pytest.raises(AppError, match="env file"):
        settings.api_key()
    p = tmp_path / "chosen.env"
    p.write_text("ANTHROPIC_API_KEY=example-only\n")
    assert replace(settings, env_file=p).api_key() == "example-only"
    import os

    assert "ANTHROPIC_API_KEY" not in os.environ


@pytest.mark.parametrize("reason", ["max_tokens", "pause_turn"])
def test_truncated_model_output_is_rejected(reason):
    class Stub:
        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            return {
                "parsed": Answer(answer="Partial", sources=[1], status="answered"),
                "raw": AIMessage(content="", response_metadata={"stop_reason": reason}),
                "parsing_error": None,
            }

    chat = AnthropicChat.__new__(AnthropicChat)
    chat.model = Stub()
    with pytest.raises(AppError, match="truncated"):
        chat.call(Answer, "system", {})


def test_provider_error_does_not_expose_provider_message():
    class Stub:
        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            raise ConnectionError("private connection value")

    chat = AnthropicChat.__new__(AnthropicChat)
    chat.model = Stub()
    with pytest.raises(AppError, match="Anthropic request failed") as error:
        chat.call(Answer, "system", {})
    assert "private connection" not in str(error.value)



def test_temperature_is_sent_only_to_models_that_still_accept_it(monkeypatch):
    from dataclasses import replace
    from rag_core.model import accepts_temperature

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert accepts_temperature("claude-haiku-4-5-20251001")
    assert not accepts_temperature("claude-haiku-5-5")
    assert not accepts_temperature("claude-opus-4-7")
    assert AnthropicChat(replace(Settings(), temperature=0)).model.temperature == 0
    current = AnthropicChat(replace(Settings(), model="claude-haiku-5-5", temperature=0))
    assert current.model.temperature is None


def test_graph_contract_records_the_temperature_actually_sent():
    from dataclasses import replace
    from architectures.graph_rag.index import contract
    from tests.architectures.test_graph_rag import documents

    assert contract(documents(), Settings())["temperature"] == 0
    assert contract(documents(), replace(Settings(), model="claude-haiku-5-5"))["temperature"] is None
