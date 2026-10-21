# ruff: noqa: E402
# Check the optional dependency before importing the implementation.
from rag_core.registry import Architecture, register, require_extra

require_extra("hybrid", "sklearn")

from .retriever import make_retriever

ARCHITECTURE = Architecture("hybrid_rag", "Hybrid RAG", make_retriever)
register(ARCHITECTURE)
