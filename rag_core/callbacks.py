"""A LangChain callback that reports each step of a turn while it runs.

LangGraph passes the callbacks in a turn's config to every node, model call and
tool call below it, including the Deep Agents research loop. Retrievers that are
not runnables report their own stages with ``report_stage``.
"""

from threading import Lock
import time

from langchain_core.callbacks import BaseCallbackHandler
from rich.text import Text
from langchain_core.callbacks.manager import dispatch_custom_event

NODES = {"entry", "prepare_query", "retrieve", "generate_answer"}
STAGE_EVENT = "rag_stage"


def report_stage(name, summary):
    """Send a retriever stage to the turn's callbacks; a no-op outside a graph run."""
    try:
        dispatch_custom_event(STAGE_EVENT, {"stage": name, "summary": summary})
    except RuntimeError:
        pass  # Called directly, e.g. by `./rag search`: there is no parent run.


def _short(value, limit=80):
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tokens(response):
    for generations in response.generations:
        for generation in generations:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                return usage.get("input_tokens"), usage.get("output_tokens")
    return None, None


class ProcessCallback(BaseCallbackHandler):
    """Turn LangChain run events into short steps and pass each one to ``sink``.

    A step is a dict: ``kind`` (node, model, tool, stage, error), ``label``,
    ``detail``, ``depth`` (nesting under other reported steps) and, for finished
    steps, ``seconds``. Callbacks can arrive from worker threads, so the sink must
    be thread-safe; appending to a list or printing with Rich both are.
    """

    raise_error = False

    def __init__(self, sink):
        self.sink = sink
        self.lock = Lock()
        self.parents, self.shown, self.started = {}, {}, {}

    def _depth(self, run_id):
        depth, parent = 0, self.parents.get(run_id)
        while parent is not None:
            depth += parent in self.shown
            parent = self.parents.get(parent)
        return depth

    def _start(self, run_id, parent_run_id, kind, label, detail=""):
        with self.lock:
            self.parents[run_id] = parent_run_id
            self.shown[run_id] = (kind, label)
            self.started[run_id] = time.perf_counter()
            depth = self._depth(run_id)
        self.sink({"kind": kind, "label": label, "detail": detail, "depth": depth})

    def _end(self, run_id, detail=""):
        with self.lock:
            if run_id not in self.shown:
                return
            kind, label = self.shown[run_id]
            seconds = time.perf_counter() - self.started.pop(run_id)
            depth = self._depth(run_id)
        self.sink(
            {"kind": kind, "label": label, "detail": detail, "depth": depth, "seconds": seconds}
        )

    def _error(self, run_id, error):
        with self.lock:
            label = self.shown.get(run_id, ("error", "step"))[1]
            depth = self._depth(run_id)
        self.sink(
            {"kind": "error", "label": label, "detail": type(error).__name__, "depth": depth}
        )

    # Graph nodes: only the shared turn's own nodes, not every internal chain.
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        name = kwargs.get("name") or ""
        node = (kwargs.get("metadata") or {}).get("langgraph_node")
        if name in NODES and node == name:
            self._start(run_id, parent_run_id, "node", name)
        else:
            with self.lock:
                self.parents[run_id] = parent_run_id

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        if run_id in self.shown:
            self._error(run_id, error)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        metadata = kwargs.get("metadata") or {}
        model = metadata.get("ls_model_name") or (serialized or {}).get("name", "model")
        self._start(run_id, parent_run_id, "model", "model call", model)

    def on_llm_end(self, response, *, run_id, **kwargs):
        tokens_in, tokens_out = _tokens(response)
        detail = f"{tokens_in} in / {tokens_out} out tokens" if tokens_in is not None else ""
        self._end(run_id, detail)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._error(run_id, error)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._start(run_id, parent_run_id, "tool", name, _short(input_str))

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._error(run_id, error)

    def on_custom_event(self, name, data, *, run_id, **kwargs):
        if name != STAGE_EVENT:
            return
        with self.lock:
            depth = self._depth(run_id) + (run_id in self.shown)
        self.sink(
            {"kind": "stage", "label": data["stage"], "detail": data["summary"], "depth": depth}
        )


ICONS = {"node": "◆", "model": "✦", "tool": "⚙", "stage": "·", "error": "✗"}


def describe(step):
    """One line of plain text for a step; used by the CLI and the Streamlit panel."""
    indent = "  " * step["depth"]
    finished = "seconds" in step
    text = f"{indent}{ICONS[step['kind']]} {step['label']}"
    if finished:
        text += f" done in {step['seconds']:.2f}s"
    if step["detail"]:
        text += f" · {step['detail']}"
    return text


def terminal_callback(console):
    """A ProcessCallback that prints each step to a Rich console, e.g. the terminal."""
    return ProcessCallback(lambda step: console.print(Text(describe(step), style="dim")))
