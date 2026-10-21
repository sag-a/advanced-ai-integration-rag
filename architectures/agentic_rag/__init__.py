# ruff: noqa: E402
# Check optional dependencies before importing implementation modules.
from rag_core.model import AnthropicChat
from rag_core.registry import Architecture, register, require_extra

require_extra("agentic", "deepagents")

from .options import options
from .retrieval import AgenticRetriever, skill_files
from .view import show_research, render_panels


def make_retriever(settings, collection, log=None):
    options(settings)
    reporter = (lambda event: log(event["event"], event)) if log else None
    return AgenticRetriever(
        collection, settings, AnthropicChat(settings, max_tokens=2400).model, reporter=reporter
    )


def check(settings, collection):
    options(settings)
    return {"skills": sorted(skill_files()), "question_time_research": True}


ARCHITECTURE = Architecture(
    "agentic_rag",
    "Agentic RAG",
    make_retriever,
    check=check,
    render_details=show_research,
    log_panels=render_panels,
    needs_llm_for_retrieval=True,
)
register(ARCHITECTURE)
