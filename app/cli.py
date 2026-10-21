"""One CLI for every registered architecture."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys

from rich.console import Console
from rich.text import Text

from rag_core.callbacks import terminal_callback
from rag_core.config import AppError, Settings
from rag_core.lesson_log import LessonLog
from rag_core.registry import REGISTRY, UNAVAILABLE, get, load_all
from rag_core.graph import build_graph, run_turn
from rag_core.store import connect, open_collection, prepare, validate_collection
from rag_core.templating import TEMPLATES, validate_options
from .runtime import create_resources


def add_settings_arguments(parser):
    defaults = Settings()
    for name in (
        "corpus",
        "collection",
        "host",
        "port",
        "model",
        "top_k",
        "env_file",
        "architecture",
    ):
        value = getattr(defaults, name)
        parser.add_argument("--" + name.replace("_", "-"), default=value, type=type(value))
    # Older running Streamlit processes still pass this now-unused argument.
    parser.add_argument("--history-tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Architecture option; values accept JSON numbers, booleans and strings",
    )


def settings_from(args):
    extras = {}
    for item in args.option:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise AppError("Use --option KEY=VALUE.")
        try:
            extras[key] = json.loads(value)
        except json.JSONDecodeError:
            extras[key] = value
    if getattr(args, "resume", False):
        extras["resume"] = True
    values = {
        field.name: getattr(args, field.name)
        for field in fields(Settings)
        if hasattr(args, field.name)
    }
    if "top_k" in extras:
        values["top_k"] = extras.pop("top_k")
    return Settings(**values, extras=extras)


def ui_arguments(settings):
    arguments = []
    for field in fields(Settings):
        name, value = field.name, getattr(settings, field.name)
        if name == "extras":
            for key, option in value.items():
                arguments += ["--option", key + "=" + json.dumps(option, default=str)]
        elif name != "temperature":
            arguments += ["--" + name.replace("_", "-"), str(value)]
    return arguments


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("architectures", help="List available plugins and missing extras")
    prompt = commands.add_parser(
        "prompt", help="Render a prompt and compare with defaults; no model calls"
    )
    prompt.add_argument("name", choices=TEMPLATES)
    add_settings_arguments(prompt)
    for name in ("check", "prepare", "search", "ask", "chat", "ui"):
        command = commands.add_parser(name)
        add_settings_arguments(command)
        if name in {"search", "ask"}:
            command.add_argument("question")
        if name in {"ask", "chat"}:
            command.add_argument(
                "--process",
                action="store_true",
                help="Show graph nodes, model calls and tool calls while the turn runs",
            )
        if name == "prepare":
            command.add_argument("--resume", action="store_true")
        if name == "ui":
            command.add_argument("--ui-port", type=int, default=8504)
    return result


def show(state, log, console):
    console.print(Text(state["messages"][-1].content))
    log.accepted(state)


def process_callbacks(args, console):
    if not args.process:
        return []
    return [terminal_callback(console)]


def chat_loop(graph, settings, args, log, console):
    messages, pending = [], None
    console.print("Ask in Greek or English. /history · /retry · /exit")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "/exit":
            break
        if question == "/history":
            for message in messages:
                console.print(Text(f"{message.type}: {message.content}"))
            continue
        if not question:
            continue
        if question == "/retry":
            if pending is None:
                console.print("No failed question to retry.")
                continue
            question = pending
        try:
            callbacks = process_callbacks(args, console)
            result = list(run_turn(graph, question, messages, settings, callbacks=callbacks))[
                -1
            ]["data"]
            messages, pending = result["messages"], None
            show(result, log, console)
        except AppError as error:
            pending = question
            console.print(Text(str(error), style="red"))


def main(argv=None):
    args = parser().parse_args(argv)
    console = Console(highlight=False)
    try:
        load_all()
        if args.command == "architectures":
            for architecture in REGISTRY.values():
                console.print(Text(f"{architecture.name}: {architecture.label}"))
            for name, reason in UNAVAILABLE.items():
                console.print(Text(f"{name}: unavailable — {reason}"))
            return 0
        settings = settings_from(args)
        architecture = get(settings.architecture)
        validate_options(settings)
        if args.command == "prompt":
            from .prompt_demo import show_prompt

            show_prompt(args.name, settings, console)
            return 0
        log = LessonLog(architecture.log_panels)
        if args.command == "ui":
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(Path(__file__).with_name("streamlit_app.py")),
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    str(args.ui_port),
                    "--server.headless",
                    "true",
                    "--server.fileWatcherType",
                    "none",
                    "--browser.gatherUsageStats",
                    "false",
                    "--",
                    *ui_arguments(settings),
                ]
            )
        if args.command in {"check", "prepare", "search"}:
            client = connect(settings)
            if args.command == "prepare":
                console.print(Text(json.dumps(prepare(client, settings))))
            collection = open_collection(client, settings)
            if args.command == "prepare" and architecture.prepare:
                result = architecture.prepare(settings, collection, log)
                console.print(
                    Text(
                        f"Architecture preparation complete: {result.get('fingerprint', 'ready')}"
                    )
                )
            elif args.command == "check":
                result = validate_collection(collection, settings)
                if architecture.check:
                    result.update(architecture.check(settings, collection))
                console.print(Text(json.dumps(result, ensure_ascii=False)))
            elif args.command == "search":
                if architecture.check:
                    architecture.check(settings, collection)
                if architecture.needs_llm_for_retrieval:
                    console.print("This architecture uses Anthropic for research.")
                result = architecture.make_retriever(settings, collection, log)(args.question)
                log.selected(result)
                for document in result.documents:
                    console.print(Text(document.page_content))
            return 0
        resources = create_resources(settings)
        graph = build_graph(resources.chat, resources.retrieve, settings, entry=resources.entry)
        if args.command == "chat":
            chat_loop(graph, settings, args, log, console)
        else:
            callbacks = process_callbacks(args, console)
            result = list(run_turn(graph, args.question, [], settings, callbacks=callbacks))[
                -1
            ]["data"]
            show(result, log, console)

        return 0
    except AppError as error:
        console.print(Text(str(error), style="red"))
        return 1
    except Exception as error:
        console.print(
            Text(
                f"Operation failed ({type(error).__name__}). No successful result was reported.",
                style="red",
            )
        )
        return 1
