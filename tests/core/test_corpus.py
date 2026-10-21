import pytest

from rag_core.config import Settings, AppError
from rag_core.corpus import build_chunks, identity
from rag_core.store import validate_collection


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text) + 2))}


def test_actual_pdf_chunks_preserve_pages_duplicates_and_budget():
    settings = Settings()
    chunks = build_chunks(settings.corpus, Tokenizer())
    assert {d.metadata["page_start"] for d in chunks} == {1, 2, 3, 4, 5}
    assert all(len("search_document: " + d.page_content) + 2 <= 512 for d in chunks)
    assert len({d.id for d in chunks}) == len(chunks)
    assert len({d.id for d in chunks if d.metadata["clause"] == "8.2"}) >= 2
    assert any(d.metadata["page_start"] == 3 and d.metadata["page_end"] == 4 for d in chunks)
    assert [d.id for d in chunks] == [d.id for d in build_chunks(settings.corpus, Tokenizer())]


def test_missing_pdf_is_actionable(tmp_path):
    with pytest.raises(AppError, match="not found"):
        identity(tmp_path / "missing.pdf")


def test_collection_preflight_does_not_embed_and_rejects_missing_metadata():
    from rag_core.store import collection_metadata

    settings = Settings()

    class Collection:
        metadata = {**collection_metadata(settings, 1), "prepared": True}
        configuration = {"hnsw": {"space": "cosine"}}

        def count(self):
            return 1

        def get(self, **kwargs):
            return {"ids": ["bad"], "documents": ["text"], "metadatas": [{}]}

    with pytest.raises(AppError):
        validate_collection(Collection(), settings)
