"""Create local Nomic embeddings and compare sentence similarities."""

from modules.c01_embeddings.local_embeddings import (
    EMBEDDING_MODEL as MODEL,
    LocalEncoder,
)

from .data import SENTENCES, SENTENCE_LABELS
from .display import print_similarity, print_vectors


def main() -> None:
    model = LocalEncoder(allow_download=True).load()
    # Keep the SAME prefix on both sides of every similarity comparison.
    inputs = ["search_document: " + text for text in SENTENCES]
    vectors = model.encode(inputs)
    print_vectors(MODEL, SENTENCES, vectors)
    print_similarity(SENTENCE_LABELS, SENTENCES, vectors)
    print(
        "\nNomic v2 is multilingual; inspect the Greek result rather than assuming retrieval quality."
    )


if __name__ == "__main__":
    main()
