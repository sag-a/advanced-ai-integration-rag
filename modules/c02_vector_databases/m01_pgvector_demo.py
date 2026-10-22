"""Unit 2, Part 1: create, populate, index, and search PostgreSQL vectors.

Stages run in this order: setup, insert, index, search, explain, hybrid. "all" runs
every stage and adds an exact-scan baseline before the index is built.
"""

import argparse
import os
import re
from importlib.resources import files

import psycopg
from dotenv import dotenv_values
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from modules.c01_embeddings.data import load_support_text
from modules.c01_embeddings.local_embeddings import EMBEDDING_MODEL as MODEL, LocalEncoder
from modules.c01_embeddings.m05_nomic_semantic_search import QUERIES

TABLE = "rag_pgvector_demo_nomic_v2.documents"
INDEX = "rag_pgvector_demo_nomic_v2.documents_embedding_hnsw"
KEYWORD_INDEX = "rag_pgvector_demo_nomic_v2.documents_content_tsv_gin"
CANDIDATES = 10  # passages each branch contributes before RRF fusion
# Public, local-only teaching credentials; override for any other database.
DEFAULT_URL = "postgresql://rag_demo:rag_demo_local@127.0.0.1:5434/rag_demo"
# Planner settings that steer one search. SET LOCAL resets when its transaction ends.
PLANNER = {
    "normal": [],
    "exact": ["SET LOCAL enable_indexscan = off", "SET LOCAL enable_bitmapscan = off"],
    "hnsw": ["SET LOCAL enable_seqscan = off"],
}


class DemoError(Exception):
    """An actionable setup or data problem, displayed without a traceback."""


def database_url() -> str:
    value = os.getenv("PGVECTOR_DATABASE_URL") or dotenv_values(".env").get("PGVECTOR_DATABASE_URL") or ""
    return value.strip() or DEFAULT_URL


def read_sql(name: str) -> str:
    statement = files("modules.c02_vector_databases").joinpath("sql", name).read_text(encoding="utf-8")
    print(f"\n--- {name} ---\n{statement.strip()}", flush=True)
    return statement


def row_count(conn) -> int:
    if conn.execute("SELECT to_regclass(%s)", (TABLE,)).fetchone()[0] is None:
        raise DemoError("Run the setup stage first: m01_pgvector_demo setup")
    return conn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]


def setup(conn) -> None:
    conn.execute(read_sql("01_setup.sql"))
    postgres = conn.execute("SHOW server_version").fetchone()[0]
    extension = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
    print(f"PostgreSQL: {postgres} | pgvector: {extension}\nTable ready: {TABLE}")


def insert(conn, model) -> None:
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(load_support_text())
    vectors = model.encode(["search_document: " + chunk for chunk in chunks])
    statement = read_sql("02_insert.sql")
    with conn.transaction():
        for number, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
            metadata = {"source": "telecom_support.txt", "chunk_number": number, "language": "en", "model": MODEL}
            conn.execute(statement, {"id": number, "content": chunk, "metadata": Jsonb(metadata), "embedding": Vector(vector)})
            print(f"Bound parameters: id={number}, characters={len(chunk)}, vector={len(vector)}D, metadata={metadata}")
    print(f"Upserted {len(chunks)} passages | Total stored rows: {row_count(conn)}")


def create_index(conn) -> None:
    conn.execute(read_sql("03_index.sql"))
    definition = conn.execute("SELECT pg_get_indexdef(%s::regclass)", (INDEX,)).fetchone()[0]
    print(f"Index definition: {definition}")


def run_search(conn, statement, vector, top_k, mode, explain=False):
    """Return (rows, plan lines) for one query under one planner mode."""
    params = {"query": Vector(vector), "top_k": top_k}
    plan = []
    with conn.transaction():
        for setting in PLANNER[mode]:
            print(setting + ";")
            conn.execute(setting)
        if explain:
            print("EXPLAIN (ANALYZE, BUFFERS) <the search SQL above>")
            plan = [row[0] for row in conn.execute("EXPLAIN (ANALYZE, BUFFERS) " + statement, params)]
        rows = conn.execute(statement, params).fetchall()
    return rows, plan


def search(conn, queries, vectors, top_k, mode="normal") -> None:
    statement = read_sql("04_search.sql")
    for query, vector in zip(queries, vectors, strict=True):
        print(f"\nQuery: {query}\nPlanner mode: {mode} | Query vector: {len(vector)}D normalized Nomic")
        rows, _ = run_search(conn, statement, vector, top_k, mode)
        for rank, (number, content, metadata, distance, similarity) in enumerate(rows, start=1):
            print(f"\n{rank}. Chunk {number} | Cosine distance: {distance:.4f} | Similarity: {similarity:.4f}")
            print(f"Metadata: {metadata}\n{content}")
    print("\nSimilarity is 1 - cosine distance. Scores are not probabilities; top-k always returns candidates.")


def explain(conn, queries, vectors, top_k) -> None:
    if conn.execute("SELECT to_regclass(%s)", (INDEX,)).fetchone()[0] is None:
        raise DemoError("Create the HNSW index first: m01_pgvector_demo index")
    statement = read_sql("04_search.sql")
    for query, vector in zip(queries, vectors, strict=True):
        print(f"\n=== Query plans: {query} ===")
        ids = {}
        for mode in ("exact", "normal", "hnsw"):
            print(f"\nPlanner mode: {mode}")
            rows, plan = run_search(conn, statement, vector, top_k, mode, explain=True)
            # PostgreSQL prints the whole bound vector; shorten it for display only.
            print("\n".join(re.sub(r"'\[[^\]]*\]'::vector", "'<768D query vector>'::vector", line) for line in plan))
            ids[mode] = [row[0] for row in rows]
            print(f"Returned chunk IDs: {ids[mode]}")
        overlap = len(set(ids["exact"]) & set(ids["hnsw"])) / len(ids["exact"])
        print(f"Exact/HNSW top-k overlap for this query: {overlap:.1%} | Same ordering: {ids['exact'] == ids['hnsw']}")
    print("\nA six-row table can legitimately use a sequential scan in normal mode.")
    print("Encouraging an HNSW plan demonstrates index mechanics, not a speed improvement.")
    print("Approximate search can miss neighbors; overlap on three toy queries is not a recall benchmark.")


def hybrid(conn, queries, vectors, top_k) -> None:
    if conn.execute("SELECT to_regclass(%s)", (KEYWORD_INDEX,)).fetchone()[0] is None:
        raise DemoError("Add the keyword column and index first: m01_pgvector_demo setup, then index")
    statement = read_sql("05_hybrid.sql")
    for query, vector in zip(queries, vectors, strict=True):
        terms = conn.execute("SELECT plainto_tsquery('english', %s)::text", (query,)).fetchone()[0].replace("&", "|")
        print(f"\nQuery: {query}\nKeyword terms after stemming, any one can match: {terms or '(none: only stop words)'}")
        params = {"query": Vector(vector), "text": query, "top_k": top_k, "candidate_k": CANDIDATES}
        for rank, (number, score, dense_rank, keyword_rank, highlighted) in enumerate(conn.execute(statement, params), start=1):
            print(f"\n{rank}. Chunk {number} | RRF: {score:.4f} | Dense rank: {dense_rank or 'absent'} | Keyword rank: {keyword_rank or 'absent'}")
            print(highlighted)
    print("\nRRF adds 1/(60 + rank) from each list, so it combines positions, not raw scores.")
    print("[Brackets] mark keyword matches. Stemming lets 'charged' match 'charges'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=("setup", "insert", "index", "search", "explain", "hybrid", "all"))
    parser.add_argument("--query", help="One search question instead of the three Unit 1 teaching questions")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum results per question (default: 3)")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    def wants(stage):
        return args.stage in (stage, "all")

    try:
        # Autocommit makes each conn.transaction() a top-level block, so SET LOCAL resets on exit.
        with psycopg.connect(database_url(), autocommit=True, connect_timeout=5) as conn:
            if wants("setup"):
                setup(conn)
                if args.stage == "setup":
                    return
            stored = row_count(conn)
            register_vector(conn)
            if args.stage == "index":
                create_index(conn)
                return
            if not stored and not wants("insert"):
                raise DemoError("The demo table is empty. Run m01_pgvector_demo insert first.")

            model = LocalEncoder(allow_download=True).load()
            if wants("insert"):
                insert(conn, model)
                if args.stage == "insert":
                    return
            queries = [args.query] if args.query else QUERIES
            vectors = model.encode(["search_query: " + query for query in queries])
            if args.stage == "all":
                print("\n=== Exact baseline (works even if an index exists from an earlier run) ===")
                search(conn, queries, vectors, args.top_k, mode="exact")
                create_index(conn)
            if wants("search"):
                search(conn, queries, vectors, args.top_k)
            if wants("explain"):
                explain(conn, queries, vectors, args.top_k)
            if wants("hybrid"):
                hybrid(conn, queries, vectors, args.top_k)
    except psycopg.OperationalError:
        raise SystemExit("Cannot connect to PostgreSQL. Start compose.pgvector.yaml or check PGVECTOR_DATABASE_URL.") from None
    except psycopg.errors.InsufficientPrivilege:
        raise SystemExit("The demo database user needs permission to create the vector extension, schema, and index.") from None
    except psycopg.Error as error:
        # Do not echo driver messages, which can contain connection credentials.
        raise SystemExit(f"PostgreSQL rejected the operation (SQLSTATE {error.sqlstate}); inspect the SQL printed above.") from None
    except DemoError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
