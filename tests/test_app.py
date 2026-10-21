"""Exercise the page through Streamlit's real rerun and widget lifecycle."""

import pytest
from streamlit.testing.v1 import AppTest
from types import SimpleNamespace
from rag_core.retrieval import RetrievalResult


from app import streamlit_app as ui
from rag_core.config import AppError
from tests.core.test_conversation import answer
from tests.core.test_streaming import StreamingChat


def resources(chat, passage):
    return SimpleNamespace(
        chat=chat, retrieve=lambda q: RetrievalResult([passage()], 0.0), entry=None
    )


def app(tmp_path):
    return AppTest.from_string(
        "from rag_core.config import Settings\n"
        "from app.streamlit_app import main\n"
        "main(Settings())\n"
    )


@pytest.fixture
def setup_ui(monkeypatch):
    from tests.core.test_conversation import passage

    chat = StreamingChat(
        answer("Οι Καρδιές λήγουν σε ένα έτος."),
        {"search_query": "Hearts expiry?", "clarification": ""},
        answer("Hearts expire after one year."),
    )
    monkeypatch.setattr(ui, "create_resources", lambda settings: resources(chat, passage))
    return chat


def test_followup_survives_reruns_but_refresh_and_new_chat_start_empty(tmp_path, setup_ui):
    at = app(tmp_path).run()
    assert not at.exception
    original_thread = at.query_params["thread"]
    at.chat_input[0].set_value("Πότε λήγουν οι Καρδιές;").run()
    assert not at.exception
    assert len(at.chat_message) == 2
    assert "Sources" in [e.label for e in at.expander]
    assert any("έτος" in item.value for item in at.markdown)
    assert len(setup_ui.calls) == 1
    at.run()
    assert len(at.chat_message) == 2 and len(setup_ui.calls) == 1
    at.chat_input[0].set_value("And in English?").run()
    assert len(at.chat_message) == 4 and len(setup_ui.calls) == 3
    # A fresh browser starts empty, even when it has the old URL identifier.
    refreshed = app(tmp_path)
    refreshed.query_params["thread"] = original_thread
    refreshed.run()
    assert not refreshed.chat_message and len(setup_ui.calls) == 3
    at.button(key="new_chat").click().run()
    assert at.query_params["thread"] != original_thread
    assert not at.chat_message


def test_failed_draft_is_withdrawn_and_retry_is_explicit(tmp_path, monkeypatch):
    from tests.core.test_conversation import passage

    chat = StreamingChat(AppError("Interrupted stream"), answer("Accepted"))
    monkeypatch.setattr(ui, "create_resources", lambda settings: resources(chat, passage))
    at = app(tmp_path).run()
    at.chat_input[0].set_value("Question").run()
    assert not at.exception and len(at.error) == 1
    assert not any("Live draft" in item.value for item in at.markdown)
    assert len(chat.calls) == 1
    at.run()
    assert len(chat.calls) == 1 and at.chat_input[0].disabled
    at.button(key="retry_turn").click().run()
    assert not at.exception
    assert len(chat.calls) == 2
    at.run()
    assert len(at.chat_message) == 2


def test_missing_configuration_is_actionable(tmp_path, monkeypatch):
    def fail(settings):
        raise AppError("Cannot reach Chroma. Start compose.chroma.yaml.")

    monkeypatch.setattr(ui, "create_resources", fail)
    at = app(tmp_path).run()
    assert not at.exception
    assert "Cannot reach Chroma" in at.error[0].value
    assert not at.chat_input


def test_process_steps_go_to_the_terminal_not_the_page(tmp_path, setup_ui, monkeypatch):
    from io import StringIO
    from rich.console import Console

    terminal = StringIO()
    monkeypatch.setattr(ui, "TERMINAL", Console(file=terminal, width=200))
    at = app(tmp_path).run()
    at.chat_input[0].set_value("Question [2024]").run()
    assert not at.exception
    printed = terminal.getvalue()
    assert "Question [2024]" in printed
    for node in ("prepare_query", "retrieve", "generate_answer"):
        assert f"◆ {node} done in" in printed
    assert not any(e.label.startswith("Process") for e in at.expander)
    assert not at.toggle


@pytest.mark.parametrize("name", ["graph_rag", "agentic_rag"])
def test_architecture_switch_starts_an_empty_message_list(tmp_path, monkeypatch, name):
    from tests.core.test_conversation import passage

    chat = StreamingChat(answer("First"), answer("Second"))
    monkeypatch.setattr(ui, "create_resources", lambda settings: resources(chat, passage))
    at = app(tmp_path).run()
    at.chat_input[0].set_value("First question").run()
    first_thread = at.query_params["thread"]
    at.selectbox(key="architecture_selector").select(name).run()
    assert not at.exception and not at.chat_message
    assert at.query_params["thread"] != first_thread
    at.chat_input[0].set_value("Second question").run()
    thread = at.query_params["thread"]
    restored = app(tmp_path)
    restored.query_params.update(architecture=name, thread=thread)
    restored.run()
    assert not restored.exception and not restored.chat_message
    assert len(chat.calls) == 2
    # The same external thread ID names a different conversation in each plugin.
    other = app(tmp_path)
    other.query_params.update(architecture=name, thread=first_thread)
    other.run()
    assert not other.chat_message and len(chat.calls) == 2
