"""Unit 2, Part 1: create, populate, index, and search PostgreSQL vectors."""

import argparse
import os
import re
from importlib.resources import files
from pathlib import Path

from dotenv import dotenv_values

from modules.c01_embeddings.local_embeddings import EMBEDDING_MODEL as MODEL, LocalEncoder
DIMENSIONS = 768
# Public, local-only teaching credentials; override for any other database.
DEFAULT_URL = "postgresql://rag_demo:rag_demo_local@127.0.0.1:5434/rag_demo"


class DemoError(Exception):
    """An actionable setup or data problem, displayed without a traceback."""


def database_url() -> str:
    value = os.getenv("PGVECTOR_DATABASE_URL", "").strip()
    if not value:
        value = (dotenv_values(Path.cwd() / ".env").get("PGVECTOR_DATABASE_URL") or "").strip()
    return value or DEFAULT_URL


def read_sql(name: str) -> str:
    statement = files("modules.c02_vector_databases").joinpath("sql", name).read_text(encoding="utf-8")
    print(f"\n--- {name} ---\n{statement.strip()}", flush=True)
    return statement


def require_table(conn) -> None:
    if conn.execute("SELECT to_regclass('rag_pgvector_demo_nomic_v2.documents')").fetchone()[0] is None:
        raise DemoError("Run the setup stage first: m01_pgvector_demo setup")
    column = conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'rag_pgvector_demo_nomic_v2.documents'::regclass "
        "AND attname = 'embedding' AND NOT attisdropped"
    ).fetchone()
    if column != (f"vector({DIMENSIONS})",):
        raise DemoError("The demo table must have an embedding vector(768) column; use the supplied schema.")


def setup(conn) -> None:
    with conn.transaction():
        conn.execute(read_sql("01_setup.sql"))
        require_table(conn)
    postgres = conn.execute("SHOW server_version").fetchone()[0]
    extension = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
    print(f"PostgreSQL: {postgres} | pgvector: {extension}\nTable ready: rag_pgvector_demo_nomic_v2.documents")


def create_index(conn) -> None:
    require_table(conn)
    with conn.transaction():
        conn.execute(read_sql("03_index.sql"))
    definition = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = 'rag_pgvector_demo_nomic_v2' "
        "AND indexname = 'documents_embedding_hnsw'"
    ).fetchone()[0]
    print(f"Index definition: {definition}")


def insert_documents(conn, model) -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pgvector import Vector
    from psycopg.types.json import Jsonb

    from modules.c01_embeddings.data import load_support_text

    require_table(conn)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    chunks = splitter.split_text(load_support_text())
    vectors = model.encode(["search_document: " + chunk for chunk in chunks])
    if vectors.shape != (len(chunks), DIMENSIONS):
        raise DemoError("The encoder must produce one 768-dimensional vector per chunk.")
    statement = read_sql("02_insert.sql")
    with conn.transaction():
        for number, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
            metadata = {"source": "telecom_support.txt", "chunk_number": number, "language": "en", "model": MODEL}
            conn.execute(statement, {"id": number, "content": chunk, "metadata": Jsonb(metadata), "embedding": Vector(vector)})
            print(f"Bound parameters: id={number}, characters={len(chunk)}, vector={DIMENSIONS}D, metadata={metadata}")
    count = conn.execute("SELECT count(*) FROM rag_pgvector_demo_nomic_v2.documents").fetchone()[0]
    print(f"Upserted {len(chunks)} passages | Total stored rows: {count}")


def require_documents(conn) -> None:
    require_table(conn)
    count, incompatible = conn.execute(
        "SELECT count(*), count(*) FILTER (WHERE metadata->>'model' IS DISTINCT FROM %s) "
        "FROM rag_pgvector_demo_nomic_v2.documents", (MODEL,)
    ).fetchone()
    if not count:
        raise DemoError("The demo table is empty. Run m01_pgvector_demo insert first.")
    if incompatible:
        raise DemoError("The table contains a different embedding model; keep incompatible embedding spaces in separate tables.")


def planner_mode(conn, mode: str) -> None:
    # All these statements run inside explicit transactions with autocommit on.
    # No ALTER SYSTEM or session-wide settings are needed for a teaching demo.
    settings = {
        "normal": (),
        "exact": ("SET LOCAL enable_indexscan = off", "SET LOCAL enable_bitmapscan = off"),
        "hnsw": ("SET LOCAL enable_seqscan = off",),
    }
    for statement in settings[mode]:
        print(statement + ";")
        conn.execute(statement)


def print_results(rows) -> None:
    for rank, (number, content, metadata, distance, similarity) in enumerate(rows, start=1):
        print(f"\n{rank}. Chunk {number} | Cosine distance: {distance:.4f} | Similarity: {similarity:.4f}")
        print(f"Metadata: {metadata}\n{content}")


def search(conn, queries, vectors, top_k: int, mode: str = "normal") -> None:
    from pgvector import Vector

    require_documents(conn)
    statement = read_sql("04_search.sql")
    for query, vector in zip(queries, vectors, strict=True):
        print(f"\nQuery: {query}\nPlanner mode: {mode} | Query vector: {DIMENSIONS}D normalized Nomic")
        with conn.transaction():
            planner_mode(conn, mode)
            rows = conn.execute(statement, {"query": Vector(vector), "top_k": top_k}).fetchall()
        print_results(rows)
    print("\nSimilarity is 1 - cosine distance. Scores are not probabilities; top-k always returns candidates.")


def explain(conn, queries, vectors, top_k: int) -> None:
    from pgvector import Vector

    require_documents(conn)
    if conn.execute("SELECT to_regclass('rag_pgvector_demo_nomic_v2.documents_embedding_hnsw')").fetchone()[0] is None:
        raise DemoError("Create the HNSW index first: m01_pgvector_demo index")
    statement = read_sql("04_search.sql")
    setting_names = ("enable_seqscan", "enable_indexscan", "enable_bitmapscan")

    def snapshot():
        return {name: conn.execute("SELECT current_setting(%s)", (name,)).fetchone()[0] for name in setting_names}

    before = snapshot()
    for query, vector in zip(queries, vectors, strict=True):
        print(f"\n=== Query plans: {query} ===")
        params = {"query": Vector(vector), "top_k": top_k}
        rankings = {}
        for mode in ("exact", "normal", "hnsw"):
            print(f"\nPlanner mode: {mode}")
            with conn.transaction():
                planner_mode(conn, mode)
                print("EXPLAIN (ANALYZE, BUFFERS) <the search SQL above>")
                plan = [row[0] for row in conn.execute("EXPLAIN (ANALYZE, BUFFERS) " + statement, params).fetchall()]
                # PostgreSQL expands bound vectors in its plan. Abbreviate only
                # their displayed literals; the real query still uses all 768D.
                display_plan = [re.sub(r"'\[[^\]]*\]'::vector", f"'<{DIMENSIONS}D query vector>'::vector", line) for line in plan]
                print("Plan display abbreviates vector literals:")
                print("\n".join(display_plan))
                rows = conn.execute(statement, params).fetchall()
            rankings[mode] = [row[0] for row in rows]
            print(f"Returned chunk IDs: {rankings[mode]}")
            if mode == "hnsw" and not any("Index Scan using documents_embedding_hnsw" in line for line in plan):
                raise DemoError("The encouraged plan did not use documents_embedding_hnsw; inspect the index definition and SQL.")
        exact_ids, hnsw_ids = rankings["exact"], rankings["hnsw"]
        overlap = len(set(exact_ids) & set(hnsw_ids)) / len(exact_ids)
        print(f"Exact/HNSW top-k overlap for this query: {overlap:.1%} | Same ordering: {exact_ids == hnsw_ids}")

    after = snapshot()
    print(f"\nPlanner settings restored: {before == after} | {after}")
    if before != after:
        raise DemoError("Planner settings did not return to their original values.")
    print("A six-row table can legitimately use a sequential scan in normal mode.")
    print("Encouraging an HNSW plan demonstrates index mechanics, not a speed improvement.")
    print("Approximate search can miss neighbors; overlap on three toy queries is not a recall benchmark.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=("setup", "insert", "index", "search", "explain", "all"))
    parser.add_argument("--query", help="One search question instead of the three Unit 1 teaching questions")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum results per question (default: 3)")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.query is not None and not args.query.strip():
        parser.error("--query must contain non-whitespace text")
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError:
        raise SystemExit("Install the database extra from the repository root: uv sync --all-extras") from None
    try:
        # Autocommit makes every explicit transaction below a top-level block.
        # SET LOCAL in the explain stage will therefore reset on block exit.
        with psycopg.connect(database_url(), autocommit=True, connect_timeout=5) as conn:
            if args.stage in ("setup", "all"):
                setup(conn)
                if args.stage == "setup":
                    return
            require_table(conn)
            if args.stage == "index":
                create_index(conn)
                return
            register_vector(conn)
            if args.stage not in ("insert", "all"):
                require_documents(conn)

            from modules.c01_embeddings.m05_nomic_semantic_search import QUERIES

            model = LocalEncoder(allow_download=True).load()
            if args.stage in ("insert", "all"):
                insert_documents(conn, model)
                if args.stage == "insert":
                    return
            queries = [args.query] if args.query is not None else QUERIES
            vectors = model.encode(["search_query: " + query for query in queries])
            if args.stage == "all":
                print("\n=== Exact baseline (works even if an index exists from an earlier run) ===")
                search(conn, queries, vectors, args.top_k, mode="exact")
                create_index(conn)
            if args.stage in ("search", "all"):
                search(conn, queries, vectors, args.top_k)
            if args.stage in ("explain", "all"):
                explain(conn, queries, vectors, args.top_k)
    except psycopg.OperationalError:
        raise SystemExit(
            "Cannot connect to PostgreSQL. Start compose.pgvector.yaml or check PGVECTOR_DATABASE_URL."
        ) from None
    except psycopg.errors.InsufficientPrivilege:
        raise SystemExit("The demo database user needs permission to create the vector extension, schema, and index.") from None
    except psycopg.Error as error:
        # Do not echo driver messages, which can contain connection credentials.
        raise SystemExit(f"PostgreSQL rejected the operation (SQLSTATE {error.sqlstate}); inspect the SQL printed above.") from None
    except DemoError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
