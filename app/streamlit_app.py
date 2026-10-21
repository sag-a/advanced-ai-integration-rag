"""One chat page; registry entries supply architecture-specific views."""

import argparse
from contextlib import closing
from dataclasses import replace
from uuid import uuid4

import streamlit as st

from rag_core.config import AppError
from rag_core.registry import REGISTRY, UNAVAILABLE, get, load_all
from rag_core.graph import build_graph, run_turn
from app.cli import add_settings_arguments, settings_from
from app.runtime import create_resources


def read_settings():
    parser = argparse.ArgumentParser(description=__doc__)
    add_settings_arguments(parser)
    return settings_from(parser.parse_args())


def switch_thread(thread):
    st.query_params["thread"] = thread
    st.session_state.pop("pending_turn", None)
    st.rerun()


def sidebar(settings, architecture, thread):
    with st.sidebar:
        names = list(REGISTRY)
        selected = st.selectbox(
            "Architecture",
            names,
            index=names.index(architecture.name),
            format_func=lambda name: REGISTRY[name].label,
            key="architecture_selector",
        )
        if selected != architecture.name:
            st.query_params["architecture"] = selected
            switch_thread(str(uuid4()))
        for name, reason in UNAVAILABLE.items():
            st.info(f"{name}: {reason}")
        st.header("Conversation")
        if st.button("New chat", key="new_chat", width="stretch"):
            switch_thread(str(uuid4()))
        st.caption("Current conversation ID")
        st.code(thread, language=None)
        st.caption(f"Anthropic · {settings.model}\n\nCollection · {settings.collection}")
        st.caption("History lasts for this browser session. Refresh starts an empty chat.")


def show_sources(result, architecture):
    if result.get("trace_url"):
        st.markdown(f"[View trace in Langfuse]({result['trace_url']})")
    if result.get("trace_warning"):
        st.warning(result["trace_warning"])
    details = result.get("observation", {}).get("retrieval_details", {})
    for warning in details.get("trace_warnings", []):
        st.warning(warning)
    if details and architecture.render_details:
        architecture.render_details(details)
    if result.get("source_map"):
        with st.expander("Sources"):
            for source in result["source_map"]:
                pages = str(source["page_start"])
                if source["page_start"] != source["page_end"]:
                    pages += f"–{source['page_end']}"
                clause = f" · §{source['clause']}" if source.get("clause") else ""
                st.caption(f"[{source['number']}] {source['source']} · PDF p. {pages}{clause}")
                st.text(source["text"])


def stream_answer(graph, messages, settings, pending, architecture):
    with st.chat_message("assistant"):
        progress, label, text = st.empty(), st.empty(), st.empty()
        try:
            with closing(
                run_turn(
                    graph,
                    pending["question"],
                    messages,
                    settings,
                    streaming=True,
                    thread_id=pending["thread"],
                    message_id=pending["id"],
                )
            ) as events:
                for event in events:
                    if event["type"] == "progress":
                        progress.caption(event["data"] + "…")
                    elif event["type"] == "warning":
                        st.warning(event["data"])
                    elif event["type"] == "draft":
                        label.caption("Draft — checking sources")
                        text.markdown(event["data"])
                    elif event["type"] == "final":
                        label.empty()
                        progress.empty()
                        st.session_state.messages = event["data"]["messages"]
                        message = st.session_state.messages[-1]
                        text.markdown(message.content)
                        show_sources(message.additional_kwargs["result"], architecture)
                        st.session_state.pop("pending_turn", None)
            st.rerun()
        except AppError as error:
            text.empty()
            label.empty()
            progress.empty()
            pending["error"] = str(error)
            st.session_state.pending_turn = pending
            st.rerun()


def main(settings=None):
    st.set_page_config(page_title="RAG course", page_icon="💬")
    st.title("RAG course")
    st.caption(
        "Ask in Greek or English. Inspect the evidence and compare retrieval approaches."
    )
    try:
        settings = settings or read_settings()
        load_all()
        name = st.query_params.get("architecture", settings.architecture)
        architecture = get(name)
        settings = replace(settings, architecture=name)
        thread = st.query_params.get("thread", "").strip() or str(uuid4())
        st.query_params.update(architecture=name, thread=thread)
        sidebar(settings, architecture, thread)
        cached = st.session_state.get("rag_resources")
        if cached is None or cached[0] != settings:
            with st.spinner("Checking corpus and model configuration…"):
                st.session_state.rag_resources = (settings, create_resources(settings))
        resources = st.session_state.rag_resources[1]
        key = (name, thread)
        if st.session_state.get("chat_key") != key:
            st.session_state.chat_key = key
            st.session_state.messages = []
            st.session_state.pop("pending_turn", None)
        messages = st.session_state.messages
        for message in messages:
            with st.chat_message("user" if message.type == "human" else "assistant"):
                st.markdown(message.content)
                if message.type == "ai":
                    show_sources(message.additional_kwargs["result"], architecture)
        pending = st.session_state.get("pending_turn")
        retry = False
        if pending:
            with st.chat_message("user"):
                st.markdown(pending["question"])
            st.warning("The previous turn is incomplete. Retry explicitly or start a new chat.")
            if pending.get("error"):
                st.error(pending["error"] + " The draft was withdrawn.")
            retry = st.button("Retry previous question", key="retry_turn")
        question = st.chat_input("Ask about the course documents…", disabled=bool(pending))
        if not question and not retry:
            return
        if question:
            pending = {"thread": thread, "question": question, "id": str(uuid4())}
            st.session_state.pending_turn = pending
            with st.chat_message("user"):
                st.markdown(question)
        graph = build_graph(resources.chat, resources.retrieve, settings, entry=resources.entry)
        stream_answer(graph, messages, settings, pending, architecture)

    except AppError as error:
        st.error(str(error))
    except Exception as error:
        st.error(
            f"RAG operation failed ({type(error).__name__}). No complete answer was accepted."
        )


if __name__ == "__main__":
    main()
