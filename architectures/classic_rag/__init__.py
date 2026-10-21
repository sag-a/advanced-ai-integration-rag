from rag_core.registry import Architecture, register
from .retriever import check, make_retriever

ARCHITECTURE = Architecture("classic_rag", "Classic RAG", make_retriever, check=check)
register(ARCHITECTURE)
