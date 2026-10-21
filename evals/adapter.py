"""Run a registry plugin through the real graph with isolated in-memory history."""

from io import StringIO
import time
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console

from rag_core.config import AppError
from rag_core.lesson_log import LessonLog
from rag_core.observability import optional_trace, scrub
from rag_core.graph import build_graph, run_turn
from .schema import Passage, Question, RunOutput


def execute(
    request: Question,
    settings,
    architecture,
    collection,
    chat=None,
    *,
    retrieval_only=False,
    telemetry=None,
):
    # Explicit Question boundary prevents reference answers/facts reaching RAG.
    if type(request) is not Question:
        raise TypeError("Evaluation architectures receive Question, never Case or gold labels.")
    if retrieval_only and (architecture.needs_llm_for_retrieval or request.history):
        raise AppError(
            "This request requires model research or history rewriting; it is excluded from no-API retrieval evaluation."
        )
    log = LessonLog(architecture.log_panels, console=Console(file=StringIO()))
    retriever = architecture.make_retriever(settings, collection, log)
    started = time.perf_counter()
    if retrieval_only:
        with optional_trace(
            telemetry, request.question, [], "evaluation-" + uuid4().hex, request.id
        ) as tracking:
            result = retriever(request.question)
            reference = tracking.trace.reference if tracking.trace else {}
            if tracking.trace:
                try:
                    tracking.trace.root.update(
                        output={
                            "mode": "retrieval-only",
                            "documents": scrub(result.documents),
                            "retrieval_details": result.details,
                        }
                    )
                except Exception:
                    tracking.warning = "Langfuse retrieval output export failed; local results remain available."
        warning = tracking.warning or (telemetry.flush() if telemetry else None)
        return RunOutput(
            search_query=request.question,
            passages=[Passage.from_document(d) for d in result.documents],
            observation={
                "retrieval_seconds": result.elapsed_seconds,
                "retrieval_details": result.details,
            },
            elapsed_seconds=time.perf_counter() - started,
            retrieval_complete=True,
            trace_id=reference.get("trace_id"),
            trace_url=reference.get("trace_url"),
            trace_warning=warning,
        )
    entry = (
        architecture.entry_node(settings, collection, log) if architecture.entry_node else None
    )
    retrieval_complete = False

    def retrieve(query):
        nonlocal retrieval_complete
        result = retriever(query)
        retrieval_complete = True
        return result

    # Each case supplies its own message list; evaluation never reads classroom history.
    messages = [
        (HumanMessage if turn.role == "human" else AIMessage)(
            id=f"history-{i}", content=turn.text
        )
        for i, turn in enumerate(request.history)
    ]
    graph = build_graph(chat, retrieve, settings, entry=entry)
    error, result = None, {}
    try:
        for event in run_turn(
            graph,
            request.question,
            messages,
            settings,
            thread_id="evaluation-" + uuid4().hex,
            message_id=request.id,
            telemetry=telemetry,
        ):
            if event["type"] in {"state", "final"}:
                result = event["data"]
    except Exception as failure:
        # Retain completed retrieval separately; never accept a failed generation.
        error = (
            f"{type(failure).__name__}: {failure}"
            if isinstance(failure, AppError)
            else type(failure).__name__
        )
    accepted = result.get("messages", [])[-1] if result.get("messages") else None
    summary = (
        accepted.additional_kwargs.get("result", {})
        if accepted and accepted.type == "ai" and accepted.id == "answer:" + request.id
        else {}
    )
    passages = [Passage.from_document(d) for d in result.get("documents", [])]
    cited_ids = {s["id"] for s in summary.get("source_map", [])}
    return RunOutput(
        search_query=result.get("search_query", request.question),
        answer=str(accepted.content) if summary and not error else "",
        status=summary.get("answer_status") if not error else None,
        passages=passages,
        citations=[p for p in passages if p.id in cited_ids],
        observation=result.get("observation", {}),
        elapsed_seconds=time.perf_counter() - started,
        trace_id=summary.get("trace_id"),
        trace_url=summary.get("trace_url"),
        trace_warning=summary.get("trace_warning") or result.get("trace_warning"),
        retrieval_complete=retrieval_complete,
        error=error,
    )
