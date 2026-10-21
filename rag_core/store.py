"""Explicit PDF preparation; question-time collection access is read-only."""

import math
import time

from langchain_core.documents import Document

from .embeddings import CONTRACT, MAX_TOKENS
from .embeddings import (
    NomicEmbeddingFunction,
)
from .config import AppError
from .corpus import CHUNKING, build_chunks, digest, source_manifest, corpus_identity
from .retrieval import RetrievalResult


def connect(settings):
    from chromadb.api import ServerAPI
    from chromadb.api.client import Client
    from chromadb.config import Settings as ClientSettings, System
    import httpx

    try:
        # Pinned 1.5.9 HttpClient constructs httpx with timeout=None. Configure
        # its session BEFORE Client initialization performs tenant HTTP checks.
        system = System(
            ClientSettings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=settings.host,
                chroma_server_http_port=settings.port,
                anonymized_telemetry=False,
            )
        )
        server = system.instance(ServerAPI)
        server._session.timeout = httpx.Timeout(30.0)
        system.start()
        client = Client.from_system(system)
        client.heartbeat()
        return client
    except Exception as error:
        raise AppError(
            "Cannot reach Chroma. Start compose.chroma.yaml and check --host/--port."
        ) from error


def collection_metadata(settings, count):
    return {
        "owner": "rag_course",
        "corpus_id": corpus_identity(settings.corpus),
        "embedding_contract_id": digest(CONTRACT),
        "chunk_config_id": digest(CHUNKING),
        "expected_chunks": count,
        "prepared": False,
    }


def validate_collection(collection, settings, *, allow_partial=False):
    expected = collection_metadata(settings, 0)
    metadata = collection.metadata or {}
    for key in ("owner", "corpus_id", "embedding_contract_id", "chunk_config_id"):
        if metadata.get(key) != expected[key]:
            raise AppError(
                f"Collection has incompatible {key}. Prepare a new collection name for changed inputs."
            )
    config = collection.configuration
    if (config.get("hnsw") or {}).get("space") != "cosine":
        raise AppError("Corpus collection must use cosine distance.")
    function = config.get("embedding_function")
    if not isinstance(function, NomicEmbeddingFunction) or function.get_config() != {
        "contract": CONTRACT
    }:
        raise AppError("Collection has an incompatible embedding function/configuration.")
    count = metadata.get("expected_chunks")
    if type(count) is not int or not 1 <= count <= 2000:
        raise AppError("Collection has no valid expected chunk count (limit: 2000).")
    if not allow_partial and (
        metadata.get("prepared") is not True or collection.count() != count
    ):
        raise AppError(
            "Corpus collection is empty or preparation is incomplete. Run prepare explicitly."
        )
    records = collection.get(include=["documents", "metadatas"], limit=count + 1)
    if len(records["ids"]) > count:
        raise AppError(
            "Collection contains unexpected records; select a dedicated corpus version."
        )
    manifest = {row["source"]: row for row in source_manifest(settings.corpus)}
    indexes = []
    for identifier, text, tags in zip(
        records["ids"], records["documents"], records["metadatas"], strict=True
    ):
        if not tags or not text or tags.get("chunk_id") != identifier:
            raise AppError("Record is missing text or chunk identity.")
        for key in ("corpus_id", "embedding_contract_id", "chunk_config_id"):
            if tags.get(key) != expected[key]:
                raise AppError(f"Record has incompatible {key}.")
        source = manifest.get(tags.get("source"))
        if source is None or tags.get("pdf_sha256") != source["pdf_sha256"]:
            raise AppError("Record has an unknown or changed source document.")
        start, end = tags.get("page_start"), tags.get("page_end")
        if (
            type(start) is not int
            or type(end) is not int
            or not 1 <= start <= end <= source["pages"]
        ):
            raise AppError("Record is missing valid physical page metadata.")
        if (
            type(tags.get("token_count")) is not int
            or not 1 <= tags["token_count"] <= MAX_TOKENS
        ):
            raise AppError("Record has no valid Nomic token-budget metadata.")
        index = tags.get("chunk_index")
        if type(index) is not int or not 0 <= index < count or index in indexes:
            raise AppError("Record has a missing, duplicate or invalid chunk index.")
        indexes.append(index)
    return {
        "collection": settings.collection,
        "corpus_id": expected["corpus_id"],
        "chunks": len(records["ids"]),
        "documents": len(manifest),
    }


def open_collection(client, settings):
    from chromadb.errors import NotFoundError

    corpus_identity(settings.corpus)
    try:
        function = NomicEmbeddingFunction()
        collection = client.get_collection(settings.collection, embedding_function=function)
        validate_collection(collection, settings)
        return collection
    except NotFoundError as error:
        raise AppError(
            "Corpus collection does not exist. Run prepare --collection <dedicated-name> first."
        ) from error
    except ValueError as error:
        raise AppError(
            "The saved collection embedding configuration is incompatible. Select a matching collection or prepare a new version."
        ) from error


def prepare(client, settings, *, chunks=None):
    from chromadb.errors import NotFoundError

    chunks = build_chunks(settings.corpus) if chunks is None else chunks
    if not chunks:
        raise AppError("Corpus produced no chunks.")
    expected = collection_metadata(settings, len(chunks))
    function = NomicEmbeddingFunction()
    try:
        collection = client.get_collection(settings.collection, embedding_function=function)
    except NotFoundError:
        collection = client.create_collection(
            settings.collection,
            embedding_function=function,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=expected,
        )
    validate_collection(collection, settings, allow_partial=True)
    if collection.metadata.get("expected_chunks") != len(chunks):
        raise AppError("Chunk count changed. Prepare a new collection name.")
    stored = collection.get(include=["documents", "metadatas"])
    existing = {
        i: (t, m)
        for i, t, m in zip(stored["ids"], stored["documents"], stored["metadatas"], strict=True)
    }
    wanted = {d.id: d for d in chunks}
    if not existing.keys() <= wanted.keys():
        raise AppError("Collection contains foreign chunks; existing records were not changed.")
    for identifier, (text, tags) in existing.items():
        if (text, tags) != (
            wanted[identifier].page_content,
            wanted[identifier].metadata,
        ):
            raise AppError("Stored chunk content differs. Use a new collection version.")
    missing = [d for d in chunks if d.id not in existing]
    for offset in range(0, len(missing), 4):
        batch = missing[offset : offset + 4]
        collection.add(
            ids=[d.id for d in batch],
            documents=[d.page_content for d in batch],
            metadatas=[d.metadata for d in batch],
        )
    if collection.metadata.get("prepared") is not True:
        collection.modify(metadata={**expected, "prepared": True})
    return validate_collection(collection, settings)


def retrieve(collection, settings, query):
    started = time.perf_counter()
    try:
        result = collection.query(
            query_texts=[query],
            n_results=min(settings.top_k, collection.count()),
            where={"corpus_id": {"$eq": corpus_identity(settings.corpus)}},
            include=["documents", "metadatas", "distances"],
        )
        documents = []
        for identifier, text, tags, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            if not math.isfinite(distance):
                raise AppError("Chroma returned an invalid distance.")
            documents.append(
                Document(
                    id=identifier,
                    page_content=text,
                    metadata={**tags, "distance": distance},
                )
            )
        return RetrievalResult(documents, time.perf_counter() - started)
    except AppError:
        raise
    except Exception as error:
        raise AppError(
            "Retrieval failed; no answer was generated. Check Chroma and the local Nomic model."
        ) from error


def read_documents(collection):
    try:
        records = collection.get(include=["documents", "metadatas"])
        from langchain_core.documents import Document

        docs = [
            Document(id=i, page_content=t, metadata=m)
            for i, t, m in zip(
                records["ids"], records["documents"], records["metadatas"], strict=True
            )
        ]
        docs.sort(key=lambda d: d.metadata["chunk_index"])
        if not docs or len({d.metadata["chunk_index"] for d in docs}) != len(docs):
            raise ValueError("missing or duplicate chunk order")
        return docs
    except Exception as error:
        raise AppError(
            "Cannot read ordered PDF passages from the prepared collection."
        ) from error
