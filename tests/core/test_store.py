from copy import deepcopy

import pytest
from chromadb.errors import NotFoundError

from rag_core.config import Settings, AppError
from rag_core.corpus import build_chunks
from rag_core.store import (
    prepare,
    open_collection,
    validate_collection,
    retrieve,
)
from tests.core.test_corpus import Tokenizer


class Collection:
    def __init__(self, metadata, function):
        self.metadata = metadata
        self.configuration = {
            "hnsw": {"space": "cosine"},
            "embedding_function": function,
        }
        self.rows = {}
        self.writes = 0

    def count(self):
        return len(self.rows)

    def get(self, **kwargs):
        rows = list(self.rows.items())[: kwargs.get("limit", len(self.rows))]
        return {
            "ids": [i for i, _ in rows],
            "documents": [v[0] for _, v in rows],
            "metadatas": [v[1] for _, v in rows],
        }

    def add(self, ids, documents, metadatas):
        self.writes += 1
        for i, t, m in zip(ids, documents, metadatas, strict=True):
            assert i not in self.rows
            self.rows[i] = (t, m)

    def modify(self, metadata):
        self.metadata = metadata


class Client:
    def __init__(self):
        self.collection = None
        self.created = 0

    def get_collection(self, name, **kwargs):
        if self.collection is None:
            raise NotFoundError("missing")
        return self.collection

    def create_collection(self, name, embedding_function, configuration, metadata):
        self.created += 1
        self.collection = Collection(metadata, embedding_function)
        return self.collection


@pytest.fixture
def prepared():
    settings = Settings()
    client = Client()
    chunks = build_chunks(settings.corpus, Tokenizer())
    prepare(client, settings, chunks=chunks)
    return client, settings, chunks


def test_idempotent_ingestion_skips_existing_rows(prepared):
    client, settings, chunks = prepared
    count = client.collection.count()
    writes = client.collection.writes
    prepare(client, settings, chunks=chunks)
    assert (
        client.created == 1
        and client.collection.count() == count
        and client.collection.writes == writes
    )
    assert open_collection(client, settings) is client.collection
    assert client.collection.configuration["embedding_function"].encoder.model is None


@pytest.mark.parametrize(
    "change",
    [
        {"corpus_id": "wrong"},
        {"prepared": False},
        {"expected_chunks": 0},
        {"embedding_contract_id": "wrong"},
    ],
)
def test_bad_collection_metadata_is_actionable(prepared, change):
    client, settings, _ = prepared
    client.collection.metadata.update(change)
    with pytest.raises(AppError):
        open_collection(client, settings)


def test_preflight_rejects_missing_pages_wrong_metric_and_missing_collection(prepared):
    client, settings, _ = prepared
    with pytest.raises(AppError, match="does not exist"):
        open_collection(Client(), settings)
    client.collection.configuration["hnsw"]["space"] = "l2"
    with pytest.raises(AppError, match="cosine"):
        open_collection(client, settings)
    client.collection.configuration["hnsw"]["space"] = "cosine"
    first = next(iter(client.collection.rows))
    client.collection.rows[first][1].pop("page_start")
    with pytest.raises(AppError, match="page"):
        validate_collection(client.collection, settings)


def test_changed_input_never_overwrites_existing_collection(prepared):
    client, settings, chunks = prepared
    before = deepcopy(client.collection.rows)
    changed = [d.model_copy(deep=True) for d in chunks]
    changed[0].page_content = "Changed text"
    with pytest.raises(AppError, match="content differs"):
        prepare(client, settings, chunks=changed)
    assert client.collection.rows == before


def test_retrieval_exception_is_not_empty_evidence(prepared):
    client, settings, _ = prepared

    def broken(**kwargs):
        raise ConnectionError("private details")

    client.collection.query = broken
    with pytest.raises(AppError, match="Retrieval failed") as error:
        retrieve(client.collection, settings, "question")
    assert "private details" not in str(error.value)
