from rag_core.registry import Architecture, register
from .retriever import make_retriever

ARCHITECTURE = Architecture("classic_rag", "Classic RAG", make_retriever)
register(ARCHITECTURE)
