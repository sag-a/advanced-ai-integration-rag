"""Unit 2, Part 2: configure Nomic once; let Chroma embed documents and queries."""

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values

from modules.c01_embeddings.local_embeddings import DIMENSIONS, EMBEDDING_MODEL as MODEL

COLLECTION = "rag_chroma_demo_nomic_v2_auto"
DEMO_ID = "mutation-demo"
DEMO_TAG = "unit2-chroma"
CATEGORY_HEADINGS = {
    "Cancelling a subscription": "cancel",
    "Understanding your bill": "billing",
    "Reporting a duplicate charge": "billing",
    "Troubleshooting internet connectivity": "technical",
    "Troubleshooting Wi-Fi": "technical",
    "Preparing a support request": "general",
}


class DemoError(Exception):
    """An actionable problem with this isolated teaching collection."""


def validate_collection(collection) -> None:
    from .nomic_embedding_function import NomicEmbeddingFunction

    metadata = collection.metadata or {}
    if (metadata.get("demo"), metadata.get("model"), metadata.get("dimensions")) != (DEMO_TAG, MODEL, DIMENSIONS):
        raise DemoError("The reserved collection has incompatible metadata; use the isolated compose.chroma.yaml service.")
    configuration = collection.configuration
    if (configuration.get("hnsw") or {}).get("space") != "cosine":
        raise DemoError("This lesson requires a cosine collection; its distance metric cannot be changed by reopening it.")
    embedding_function = configuration.get("embedding_function")
    if not isinstance(embedding_function, NomicEmbeddingFunction):
        raise DemoError("The collection must use the configured course Nomic embedding function.")
    NomicEmbeddingFunction.validate_config(embedding_function.get_config())


def create_collection(client):
    from chromadb.errors import NotFoundError
    from .nomic_embedding_function import NomicEmbeddingFunction

    embedding_function = NomicEmbeddingFunction()
    try:
        collection = client.get_collection(COLLECTION, embedding_function=embedding_function)
        print(f"Collection {COLLECTION!r} already exists; keeping its records.")
    except NotFoundError:
        print("client.create_collection(..., embedding_function=NomicEmbeddingFunction(), configuration={'hnsw': {'space': 'cosine'}})")
        collection = client.create_collection(
            name=COLLECTION,
            embedding_function=embedding_function,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"demo": DEMO_TAG, "model": MODEL, "dimensions": DIMENSIONS},
        )
    validate_collection(collection)
    print(f"Collection: {collection.name} | Metric: cosine | Records: {collection.count()}")
    return collection


def get_collection(client):
    from chromadb.errors import NotFoundError
    from .nomic_embedding_function import NomicEmbeddingFunction

    try:
        collection = client.get_collection(COLLECTION, embedding_function=NomicEmbeddingFunction())
    except NotFoundError:
        raise DemoError("Run the create stage first: m02_chroma_demo create") from None
    validate_collection(collection)
    return collection


def add_documents(collection) -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from modules.c01_embeddings.data import load_support_text

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    chunks = splitter.split_text(load_support_text())
    ids = [f"telecom-{index:03d}" for index in range(1, len(chunks) + 1)]
    # add() does not overwrite existing IDs. Skip them for repeatable rehearsals.
    existing = set(collection.get(ids=ids, include=[])["ids"])
    pending = [(number, record_id, chunk) for number, (record_id, chunk) in enumerate(zip(ids, chunks, strict=True), start=1) if record_id not in existing]
    if not pending:
        print(f"All {len(ids)} fixture IDs already exist; add skipped. Records: {collection.count()}")
        return
    metadatas = []
    for number, _, chunk in pending:
        # These are human-assigned teaching tags, not classifier predictions.
        category = next((tag for heading, tag in CATEGORY_HEADINGS.items() if heading in chunk.splitlines()), "general")
        metadatas.append({"source": "telecom_support.txt", "chunk_number": number, "language": "en", "category": category, "model": MODEL, "record_type": "fixture"})
    texts = [chunk for _, _, chunk in pending]
    print(f"collection.add(ids=..., documents=..., metadatas=...) | Chroma embeds {len(texts)} documents")
    collection.add(ids=[record_id for _, record_id, _ in pending], documents=texts, metadatas=metadatas)
    print(f"Added {len(pending)} | Skipped existing: {len(existing)} | Records: {collection.count()}")


def require_documents(collection) -> None:
    if not collection.count():
        raise DemoError("The collection is empty. Run m02_chroma_demo add first.")


def search(collection, queries, top_k: int, where=None, *, method="dense", candidate_k=10) -> None:
    require_documents(collection)
    if method != "dense":
        from .chroma_hybrid import search as hybrid_search

        hybrid_search(collection, queries, method=method, top_k=top_k,
                      candidate_k=candidate_k, where=where)
        return
    print(f"\ncollection.query(query_texts=..., n_results={top_k}, where={where!r})")
    results = collection.query(query_texts=queries, n_results=min(top_k, collection.count()), where=where,
                               include=["documents", "metadatas", "distances"])
    # Query results are nested: one inner result list for each query vector.
    for query, ids, documents, metadatas, distances in zip(queries, results["ids"], results["documents"], results["metadatas"], results["distances"], strict=True):
        print(f"\nQuery: {query}")
        if not ids:
            print("No records matched the filter.")
        for rank, (record_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances, strict=True), start=1):
            print(f"\n{rank}. {record_id} | Cosine distance: {distance:.4f} | Similarity: {1 - distance:.4f}")
            print(f"Metadata: {metadata}\n{document}")
    print("Cosine distance is lower for closer vectors. Similarity = 1 - distance; it is not a probability.")


def mutation_record(collection):
    record = collection.get(ids=[DEMO_ID], include=["documents", "metadatas", "embeddings"])
    if record["ids"] and (record["metadatas"][0] or {}).get("record_type") != "mutation-demo":
        raise DemoError("The reserved mutation ID belongs to a different record; refusing to overwrite or delete it.")
    return record


def update_document(collection) -> None:
    import numpy as np

    mutation_record(collection)
    original = "My router is offline and none of my devices can connect to the internet."
    updated = "My monthly invoice contains two completed charges; I need the billing team to review them."
    metadata = {"source": "chroma_update_demo", "language": "en", "model": MODEL, "record_type": "mutation-demo", "category": "technical", "revision": 1}
    # Reset only the owned scratch record so repeated update demos start alike.
    print(f"\nPreparing temporary record {DEMO_ID!r} with collection.upsert(...)")
    collection.upsert(ids=[DEMO_ID], documents=[original], metadatas=[metadata])
    before = mutation_record(collection)
    print(f"Before: {before['documents'][0]}\nMetadata: {before['metadatas'][0]}")

    # Supplying changed documents makes Chroma recompute their vectors.
    print("collection.update(ids=..., documents=..., metadatas=...) | Chroma re-embeds the text")
    collection.update(ids=[DEMO_ID], documents=[updated],
                      metadatas=[{**metadata, "category": "billing", "revision": 2}])
    after = mutation_record(collection)
    change = np.linalg.norm(np.asarray(after["embeddings"])[0] - np.asarray(before["embeddings"])[0])
    print(f"After: {after['documents'][0]}\nMetadata: {after['metadatas'][0]}")
    print(f"Embedding changed by L2 distance: {change:.4f} | Records: {collection.count()}")


def delete_document(collection) -> None:
    record = mutation_record(collection)
    if not record["ids"]:
        print(f"Temporary record {DEMO_ID!r} is absent; nothing to delete.")
        return
    before = collection.count()
    print(f"\ncollection.delete(ids=[{DEMO_ID!r}])")
    collection.delete(ids=[DEMO_ID])
    remaining = collection.get(ids=[DEMO_ID], include=[])["ids"]
    print(f"Records: {before} → {collection.count()} | Deleted ID still present: {bool(remaining)}")
    print("The collection remains available for the next rehearsal.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=("create", "add", "search", "filter", "update", "delete", "all"))
    parser.add_argument("--host", help="Chroma host (default: CHROMA_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Chroma port (default: CHROMA_PORT or 8002)")
    parser.add_argument("--query", help="One search question instead of the teaching queries")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum results per question")
    parser.add_argument("--method", choices=("dense", "lexical", "hybrid", "compare"), default="dense",
                        help="Search method; compare prints all three rankings (default: dense)")
    parser.add_argument("--candidate-k", type=int, default=10,
                        help="Candidates per branch before hybrid fusion (default: 10)")
    parser.add_argument("--category", default="billing", help="Metadata category for filtered search (default: billing)")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.candidate_k < 1:
        parser.error("--candidate-k must be at least 1")
    if args.method != "dense" and args.candidate_k < args.top_k:
        parser.error("--candidate-k must be at least --top-k for lexical/hybrid/compare")
    if args.method != "dense" and args.stage not in ("search", "filter", "all"):
        parser.error("--method applies to search, filter, or all")
    if args.query is not None and not args.query.strip():
        parser.error("--query must contain non-whitespace text")
    if not args.category.strip():
        parser.error("--category must contain non-whitespace text")
    env = dotenv_values(Path.cwd() / ".env")
    host = args.host or os.getenv("CHROMA_HOST") or env.get("CHROMA_HOST") or "127.0.0.1"
    try:
        port = args.port if args.port is not None else int(os.getenv("CHROMA_PORT") or env.get("CHROMA_PORT") or "8002")
    except ValueError:
        parser.error("CHROMA_PORT must be an integer")
    if not 1 <= port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        import chromadb
        from chromadb.config import Settings
        from chromadb.errors import ChromaError
    except ImportError:
        raise SystemExit("Install the Chroma extra from the repository root: uv sync --all-extras") from None
    try:
        client = chromadb.HttpClient(host=host, port=port, settings=Settings(anonymized_telemetry=False))
        client.heartbeat()
        print(f"Chroma client: {chromadb.__version__} | Server: {client.get_version()}")
    except Exception:
        raise SystemExit("Cannot reach Chroma. Start modules/c02_vector_databases/compose.chroma.yaml or check CHROMA_HOST and CHROMA_PORT.") from None
    try:
        collection = create_collection(client) if args.stage in ("create", "all") else get_collection(client)
        if args.stage == "create":
            return
        if args.stage == "delete":
            delete_document(collection)
            return
        if args.stage in ("search", "filter"):
            require_documents(collection)

        from modules.c01_embeddings.m05_nomic_semantic_search import QUERIES

        if args.stage in ("add", "all"):
            add_documents(collection)
        if args.stage in ("search", "all"):
            search(collection, [args.query] if args.query is not None else QUERIES, args.top_k,
                   method=args.method, candidate_k=args.candidate_k)
        if args.stage in ("filter", "all"):
            query = args.query or "What should I do if I was billed twice?"
            print("\n=== Compare the SAME question before and after metadata filtering ===")
            search(collection, [query], args.top_k, method=args.method, candidate_k=args.candidate_k)
            search(collection, [query], args.top_k, where={"category": {"$eq": args.category}},
                   method=args.method, candidate_k=args.candidate_k)
        if args.stage in ("update", "all"):
            update_document(collection)
        if args.stage == "all":
            delete_document(collection)
    except (DemoError, ValueError) as error:
        raise SystemExit(str(error)) from None
    except ChromaError as error:
        raise SystemExit(f"Chroma rejected the operation ({type(error).__name__}); check the collection and server versions.") from None


if __name__ == "__main__":
    main()
