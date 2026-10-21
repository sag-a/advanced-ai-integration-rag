import os
from pathlib import Path
import subprocess

import pytest

from app import cli
from rag_core.registry import Architecture, REGISTRY


def test_ui_command_binds_localhost_and_disables_file_watcher(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda args: calls.append(args) or 0)
    assert cli.main(["ui", "--ui-port", "8512"]) == 0
    arguments = calls[0]
    assert arguments[arguments.index("--server.address") + 1] == "127.0.0.1"
    assert arguments[arguments.index("--server.fileWatcherType") + 1] == "none"
    assert arguments[arguments.index("--server.port") + 1] == "8512"


def test_options_are_shared_without_app_architecture_branches():
    args = cli.parser().parse_args(
        ["check", "--option", "some_index=outputs/index.json", "--option", "max_calls=3"]
    )
    settings = cli.settings_from(args)
    assert settings.extras == {"some_index": "outputs/index.json", "max_calls": 3}
    roundtrip = cli.parser().parse_args(["check", *cli.ui_arguments(settings)])
    assert cli.settings_from(roundtrip) == settings


def test_already_running_ui_can_parse_its_old_history_argument():
    import argparse

    parser = argparse.ArgumentParser()
    cli.add_settings_arguments(parser)
    args = parser.parse_args(["--history-tokens", "2000"])
    settings = cli.settings_from(args)
    assert settings == cli.settings_from(parser.parse_args([]))
    assert "--history-tokens" not in cli.ui_arguments(settings)


def test_check_dispatches_through_a_new_registry_entry(monkeypatch):
    checked = []
    entry = Architecture(
        "test_driver",
        "Test driver",
        lambda *args: None,
        check=lambda cfg, collection: checked.append(cfg.architecture) or {"ready": True},
    )
    monkeypatch.setitem(REGISTRY, entry.name, entry)
    monkeypatch.setattr(cli, "connect", lambda settings: object())
    monkeypatch.setattr(cli, "open_collection", lambda *args: object())
    monkeypatch.setattr(cli, "validate_collection", lambda *args: {"chunks": 1})
    assert cli.main(["check", "--architecture", entry.name]) == 0
    assert checked == [entry.name]


def test_cli_chat_keeps_followup_history_only_in_the_running_process(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rag_core.retrieval import RetrievalResult
    from tests.core.test_conversation import Chat, answer, passage

    chat = Chat(
        answer("Hearts"), {"search_query": "Hearts expiry?", "clarification": ""}, answer()
    )
    inputs = iter(["What are Hearts?", "When do they expire?", "/history", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setattr(
        cli,
        "create_resources",
        lambda settings: SimpleNamespace(
            chat=chat,
            retrieve=lambda q: RetrievalResult([passage()], 0),
            entry=None,
        ),
    )
    monkeypatch.chdir(tmp_path)
    assert cli.main(["chat"]) == 0
    assert chat.calls[1][1]["history"][0]["text"] == "What are Hearts?"
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="PowerShell uses python -m app directly")
def test_single_launcher_forwards_arguments_without_architecture_logic(tmp_path):
    fake = tmp_path / "uv"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    launcher = Path(__file__).resolve().parents[1] / "rag"
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    result = subprocess.check_output(
        [str(launcher), "ask", "Greek question"], env=env, text=True
    ).splitlines()
    assert result == ["run", "--all-extras", "python", "-m", "app", "ask", "Greek question"]
