"""Vector and similarity output helpers for the numbered local demos."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def print_vectors(model_name: str, texts: list[str], vectors: np.ndarray) -> None:
    print(f"\nModel: {model_name}\nVectors: {len(vectors)} | Shape: {vectors.shape}")
    for index, (text, vector) in enumerate(zip(texts, vectors, strict=True), start=1):
        preview = np.array2string(vector[:5], precision=5, suppress_small=True)
        print(f"{index}. {text}\n   First five: {preview} | L2 norm: {np.linalg.norm(vector):.6f}")


def print_similarity(labels: list[str], texts: list[str], vectors: np.ndarray) -> None:
    # Raw coordinates mean nothing on their own. Compare vectors from ONE model.
    # Semantically similar texts generally have higher cosine similarity.
    scores = cosine_similarity(vectors)
    print("\nCosine similarity (same model for both sides):")
    print(" " * 16 + "".join(f"{label:>16}" for label in labels))
    for label, row in zip(labels, scores, strict=True):
        print(f"{label:>16}" + "".join(f"{score:16.3f}" for score in row))
    print("\nSentence legend:")
    for label, text in zip(labels, texts, strict=True):
        print(f"  {label}: {text}")
