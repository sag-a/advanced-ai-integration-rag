import pytest
from langchain_core.messages import AIMessage
from rag_core.config import Settings, AppError
from rag_core.model import AnthropicChat
from rag_core.graph import Answer


def test_missing_key_is_actionable_and_selected_env_does_not_mutate_shell(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
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
    with pytest.raises(AppError, match="Claude request failed") as error:
        chat.call(Answer, "system", {})
    assert "private connection" not in str(error.value)



def test_temperature_is_sent_only_to_models_that_still_accept_it(monkeypatch):
    from dataclasses import replace
    from rag_core.model import accepts_temperature

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
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


def empty_env(tmp_path):
    """An empty env file, so the tests never read the real .env."""
    path = tmp_path / "empty.env"
    path.write_text("")
    return path


def test_openrouter_is_the_default_provider(monkeypatch, tmp_path):
    for name in ("LLM_PROVIDER", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    settings = Settings(env_file=empty_env(tmp_path))
    assert settings.provider() == "openrouter" and settings.api_key() == "router-key"
    assert settings.base_url() == "https://openrouter.ai/api"
    chat = AnthropicChat(settings)
    assert chat.model.model == "anthropic/claude-haiku-4.5"
    assert chat.model.anthropic_api_url == "https://openrouter.ai/api"


@pytest.mark.parametrize("model, routed", [
    ("claude-haiku-4-5-20251001", "anthropic/claude-haiku-4.5"),
    ("claude-haiku-5-5", "anthropic/claude-haiku-5.5"),
    ("claude-opus-4-20250514", "anthropic/claude-opus-4"),
    ("claude-fable-5-1", "anthropic/claude-fable-5.1"),
])
def test_openrouter_model_names(monkeypatch, model, routed):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert Settings(model=model).provider_model() == routed


def test_anthropic_provider_uses_the_anthropic_key_and_model_id(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    settings = Settings(env_file=empty_env(tmp_path))
    assert settings.api_key() == "anthropic-key" and settings.base_url() is None
    assert AnthropicChat(settings).model.model == "claude-haiku-4-5-20251001"


def test_a_rejected_key_names_the_key_to_check(monkeypatch, tmp_path):
    class Rejected(Exception):
        status_code = 401

    class Stub:
        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            raise Rejected()

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    chat = AnthropicChat(Settings(env_file=empty_env(tmp_path)))
    chat.model = Stub()
    with pytest.raises(AppError, match="via OpenRouter.*Check OPENROUTER_API_KEY"):
        chat.call(Answer, "system", {})
