"""Retrieve support passages and compare similarity metrics (slides 16, 24–26)."""

import argparse

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modules.c01_embeddings.local_embeddings import (
    EMBEDDING_MODEL as MODEL,
    LocalEncoder,
)
from sklearn.metrics.pairwise import cosine_similarity

from .data import load_support_text

QUERIES = [
    "How do I cancel my subscription?",
    "What should I do if I was billed twice?",
    "My Wi-Fi stopped working but my wired connection works. What can I try?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query", help="Search with one question instead of the three teaching queries"
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of passages to print per question"
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.query is not None and not args.query.strip():
        parser.error("--query must contain non-whitespace text")

    # Larger chunks keep each heading with its explanation in this document.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len
    )
    chunks = splitter.split_text(load_support_text())
    queries = [args.query] if args.query is not None else QUERIES
    model = LocalEncoder(allow_download=True).load()
    # Queries and passages use the SAME encoder, with different task prefixes.
    documents = model.encode(["search_document: " + chunk for chunk in chunks])
    questions = model.encode(["search_query: " + query for query in queries])
    print(
        f"Model: {MODEL}\nDocument vectors: {documents.shape} | Query vectors: {questions.shape}"
    )
    print("All vectors are L2-normalized; search uses the full embedding space.")

    for query, vector in zip(queries, questions, strict=True):
        cosine = cosine_similarity(vector[None, :], documents)[0]
        dot = documents @ vector
        euclidean = np.linalg.norm(documents - vector, axis=1)
        # Similarity: higher first. Distance: lower first. Stable sorting keeps
        # source order for exact ties; rounding can affect nearly equal scores.
        cosine_order = np.argsort(-cosine, kind="stable")
        dot_order = np.argsort(-dot, kind="stable")
        euclidean_order = np.argsort(euclidean, kind="stable")
        print(f"\nQuery: {query}")
        print(f"Full cosine ranking (chunk IDs): {(cosine_order + 1).tolist()}")
        print(f"Full dot-product ranking:        {(dot_order + 1).tolist()}")
        print(f"Full Euclidean ranking:          {(euclidean_order + 1).tolist()}")
        # For unit vectors: cosine = dot; squared Euclidean = 2 - 2 * dot.
        print(f"Max |cosine - dot|: {np.max(np.abs(cosine - dot)):.2e}")
        print(
            f"Max |distance² - (2 - 2 × dot)|: {np.max(np.abs(euclidean**2 - (2 - 2 * dot))):.2e}"
        )

        for rank, index in enumerate(cosine_order[: args.top_k], start=1):
            print(
                f"\nResult {rank} | Chunk {index + 1} | Cosine: {cosine[index]:.3f} | Dot: {dot[index]:.3f} | Euclidean: {euclidean[index]:.3f}"
            )
            print(chunks[index])

    print(
        "\nScores are similarities/distances, not probabilities or guaranteed relevance."
    )
    print("Top-k returns candidates even for an unrelated query; inspect the results.")
    print(
        "This demo retrieves text without generating an answer or using a vector database."
    )


if __name__ == "__main__":
    main()
