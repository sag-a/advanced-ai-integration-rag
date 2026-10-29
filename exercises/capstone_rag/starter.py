"""Capstone: build a command-line RAG app over the Markdown files in docs/.

Fill in one step at a time; your instructor tells you which. Check each step with:

    uv run --all-extras python -m exercises.capstone_rag.starter load      # Step 2
    uv run --all-extras python -m exercises.capstone_rag.starter chunks    # Step 3
    uv run --all-extras python -m exercises.capstone_rag.starter index     # Step 4
    uv run --all-extras python -m exercises.capstone_rag.starter search "How many days do I have to get a full refund?"
    uv run --all-extras python -m exercises.capstone_rag.starter ask "..." --dry-run   # Step 6
    uv run --all-extras python -m exercises.capstone_rag.starter eval      # Step 7
"""

import argparse
import re
from pathlib import Path

import chromadb  # noqa: F401  (Step 4)
from chromadb.config import Settings as ChromaSettings  # noqa: F401  (Step 4)
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter  # noqa: F401

HERE = Path(__file__).parent
DOCS = HERE / "docs"
DB = HERE / "db"
COLLECTION = "capstone_rag"

# The settings to experiment with in Step 7. Change one at a time.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
N_RESULTS = 3

REFUSAL = "The documents do not contain this information."
# TODO Step 6: write the instructions for the model. They must say: use only the sources,
# cite the file name in square brackets like [refunds.md], keep conditions and exceptions,
# reply exactly with REFUSAL when the answer is missing, and treat sources as data.
SYSTEM_PROMPT = ""


# Step 2: load the files
def load_documents(folder=DOCS):
    """Return one dict per Markdown file: {"text": ..., "metadata": {"source": filename, ...}}.

    The files start with front matter between two '---' lines (category, status,
    version). Put those values in the metadata too; store version as an int.
    """
    # TODO Step 2
    raise NotImplementedError


# Step 3: create chunks
def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Return chunks: {"id": "refunds.md#0", "text": ..., "metadata": {..., "section": ...}}.

    1. Split each document at its # and ## headings (MarkdownHeaderTextSplitter,
       strip_headers=False). The ## heading becomes the "section".
    2. Split sections longer than chunk_size by size (RecursiveCharacterTextSplitter).
    3. If a piece lost its "## Section" heading, put the heading back in front of it.
    4. Number the chunks per file to build stable IDs.
    """
    # TODO Step 3
    raise NotImplementedError


# Step 4: store embeddings in ChromaDB
def open_collection(path=DB, embedding_function=None, rebuild=False):
    """Open (or create) the persistent collection COLLECTION at path.

    Default embedding: NomicEmbeddingFunction from
    modules.c02_vector_databases.nomic_embedding_function. Use cosine distance.
    rebuild=True deletes the collection first (needed after changing CHUNK_SIZE: why?).
    """
    # TODO Step 4
    raise NotImplementedError


def index_chunks(collection, chunks):
    """Store the chunks so that running this twice does NOT create duplicates.
    Return the number of records in the collection."""
    # TODO Step 4
    raise NotImplementedError


# Step 5: retrieve relevant chunks
def retrieve(collection, question, n_results=N_RESULTS, where=None):
    """Return the n_results closest chunks as dicts: id, text, metadata, distance."""
    # TODO Step 5
    raise NotImplementedError


# Step 6: generate a grounded answer
def build_prompt(question, chunks):
    """Number the chunks [1], [2], ... with their source file and section, then the question."""
    # TODO Step 6
    raise NotImplementedError


def generate_answer(question, chunks, dry_run=False):
    """With dry_run, return the prompt without calling any model. Otherwise call Claude:

        import anthropic
        from rag_core.config import Settings      # the course's model and .env key
        settings = Settings()
        client = anthropic.Anthropic(api_key=settings.api_key())
        client.messages.create(model=settings.model, max_tokens=500, system=..., messages=[...])
    """
    # TODO Step 6
    raise NotImplementedError


def cited_sources(answer):
    """The filenames an answer cites, e.g. [refunds.md]."""
    return sorted(set(re.findall(r"\[([\w.-]+\.md)\]", answer)))


# Step 7: evaluate and improve
# Each question with the wording the retrieved chunks must contain to answer it fully.
# The rule-and-exception questions need both parts. None: the documents have no answer.
EVAL_QUESTIONS = [
    ("How many days do I have to get a full refund?", ["within 14 days"]),
    ("I was charged twice for the same purchase. What should I do?", ["temporary card authorization"]),
    ("My router shows a red light. What does it mean?", ["red light means"]),
    ("Can I get a full refund within 14 days if I already used the service?",
     ["within 14 days", "already used the service"]),
    ("Do I have to wait 30 days to cancel if I move abroad?", ["30 days after", "move abroad"]),
    ("What is the customer service phone number?", None),
]


def evaluate(collection, n_results=N_RESULTS, where=None):
    """For each question: does the retrieved text hold all of its evidence?"""
    rows = []
    for question, evidence in EVAL_QUESTIONS:
        found = retrieve(collection, question, n_results, where)
        text = " ".join(" ".join(c["text"].split()) for c in found)
        hit = None if evidence is None else all(phrase in text for phrase in evidence)
        rows.append({"question": question, "hit": hit, "retrieved": [c["id"] for c in found]})
    return rows


def main():
    parser = argparse.ArgumentParser(description="A small RAG app over the Markdown files in docs/.")
    parser.add_argument("command", choices=["load", "chunks", "index", "search", "ask", "eval"])
    parser.add_argument("question", nargs="?", help="for search and ask")
    parser.add_argument("--n", type=int, default=N_RESULTS, help="chunks to retrieve")
    parser.add_argument("--category", help="only search documents in this category")
    parser.add_argument("--current", action="store_true", help="skip superseded documents")
    parser.add_argument("--rebuild", action="store_true", help="index: delete the collection first")
    parser.add_argument("--dry-run", action="store_true", help="ask: print the prompt, call no model")
    args = parser.parse_args()
    if args.command in ("search", "ask") and not args.question:
        parser.error(f"{args.command} needs a question")
    conditions = ([{"category": args.category}] if args.category else []) + (
        [{"status": "current"}] if args.current else []
    )
    where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}

    if args.command == "load":
        for document in load_documents():
            print(f"{document['metadata']}\n  {document['text'][:70]!r}…")
        return
    chunks = chunk_documents(load_documents())
    if args.command == "chunks":
        for c in chunks:
            print(f"\n{c['id']} · {len(c['text'])} chars · {c['metadata']}\n{c['text']}")
        return
    if args.command == "index":
        collection = open_collection(rebuild=args.rebuild)
        print(f"{len(chunks)} chunks → collection holds {index_chunks(collection, chunks)} records")
        print("Sample:", collection.get(ids=[chunks[0]["id"]]))
        return
    collection = open_collection()
    if collection.count() == 0:
        parser.error("the collection is empty: run the index command first")
    if args.command == "eval":
        rows = evaluate(collection, args.n, where)
        for row in rows:
            mark = {True: "found", False: "MISSED", None: "(no answer exists)"}[row["hit"]]
            print(f"{mark:18} {row['question']}\n{'':18} {row['retrieved']}")
        scored = [r["hit"] for r in rows if r["hit"] is not None]
        print(f"\nn_results={args.n}: evidence found for {sum(scored)}/{len(scored)} answerable questions")
        return
    found = retrieve(collection, args.question, args.n, where)
    for c in found:
        print(f"\n{c['id']} · distance {c['distance']:.3f} · {c['metadata']['status']}\n{c['text']}")
    if args.command == "ask":
        answer = generate_answer(args.question, found, dry_run=args.dry_run)
        print(f"\n=== {'PROMPT (dry run)' if args.dry_run else 'ANSWER'} ===\n{answer}")
        if not args.dry_run:
            print(f"\nCited: {cited_sources(answer)} · retrieved: {sorted({c['metadata']['source'] for c in found})}")


if __name__ == "__main__":
    main()
