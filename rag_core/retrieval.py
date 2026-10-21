"""The only question-time retrieval contract shared by architectures."""

from dataclasses import dataclass, field
from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    documents: list[Document]
    elapsed_seconds: float
    details: dict = field(default_factory=dict)
