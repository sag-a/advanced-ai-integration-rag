# ruff: noqa: E402
# Check optional dependencies before importing implementation modules.
from dataclasses import replace

from rag_core.model import AnthropicChat
from rag_core.registry import Architecture, register, require_extra
from rag_core.store import read_documents

require_extra("graph", "graphviz", "networkx", "community")

from .index import load_index, index_path, prepare as prepare_index
from .retrieval import GraphRetriever
from .view import show_graph, render_panels


def make_retriever(settings, collection, log=None):
    return GraphRetriever(collection, settings, reporter=log)


def prepare(settings, collection, log):
    if index_path(settings).exists():
        return load_index(read_documents(collection), settings)
    chat = AnthropicChat(replace(settings, temperature=0), max_tokens=2400)
    return prepare_index(
        read_documents(collection),
        settings,
        chat,
        resume=settings.extras.get("resume", False),
        progress=log.progress,
    )


def check(settings, collection):
    graph = load_index(read_documents(collection), settings)
    return {
        "index_fingerprint": graph["fingerprint"],
        "relationships": len(graph["relationships"]),
        "indexing_observations": graph.get("indexing_observations", []),
    }


ARCHITECTURE = Architecture(
    "graph_rag",
    "Graph RAG",
    make_retriever,
    prepare=prepare,
    check=check,
    render_details=show_graph,
    log_panels=render_panels,
)
register(ARCHITECTURE)
