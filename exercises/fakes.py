"""A tiny stand-in embedding so exercise tests run offline, without the Nomic model."""

import re
import zlib

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

STOP_WORDS = {"a", "an", "and", "can", "do", "for", "get", "have", "i", "if", "is", "my",
              "of", "on", "the", "they", "to", "was", "who", "you"}


class WordOverlap(EmbeddingFunction[Documents]):
    """Texts sharing more content words get closer vectors. Deterministic, no download."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vector = np.zeros(512, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if word not in STOP_WORDS:
                    vector[zlib.crc32(word.rstrip("s").encode()) % 512] += 1
            vectors.append(vector / (np.linalg.norm(vector) or 1))
        return vectors

    @staticmethod
    def name() -> str:
        return "exercise-word-overlap"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "WordOverlap":
        return WordOverlap()
