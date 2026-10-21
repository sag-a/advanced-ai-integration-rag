"""One vector search; use the same source passages as every other architecture."""

from rag_core.store import retrieve
from uuid import uuid4


def make_retriever(settings, collection, log=None):
    def search(question):
        result = retrieve(collection, settings, question)
        result.details["architecture"] = settings.architecture
        result.details["retrieval_id"] = uuid4().hex[:8]
        result.details["selected_ids"] = [document.id for document in result.documents]
        return result

    return search
