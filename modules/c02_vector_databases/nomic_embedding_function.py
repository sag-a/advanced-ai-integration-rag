"""Chroma adapter: keep Nomic's document/query prefixes behind configuration."""

from copy import deepcopy

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from modules.c01_embeddings.local_embeddings import CONTRACT, PREFIXES, LocalEncoder


# Chroma calls __call__ for documents and embed_query for query_texts.
# https://docs.trychroma.com/docs/embeddings/embedding-functions
@register_embedding_function
class NomicEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        # Loading is deferred until Chroma actually needs an embedding.
        self.encoder = LocalEncoder(allow_download=True)

    def __call__(self, input: Documents) -> Embeddings:
        return list(self.encoder.encode([PREFIXES["document"] + text for text in input]))

    def embed_query(self, input: Documents) -> Embeddings:
        return list(self.encoder.encode([PREFIXES["query"] + text for text in input]))

    @staticmethod
    def name() -> str:
        return "course-nomic-v2-moe"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    def get_config(self) -> dict:
        return {"contract": deepcopy(CONTRACT)}

    @staticmethod
    def validate_config(config: dict) -> None:
        if config != {"contract": CONTRACT}:
            raise ValueError("Incompatible Nomic configuration; use a new collection and re-embed.")

    @staticmethod
    def build_from_config(config: dict) -> "NomicEmbeddingFunction":
        NomicEmbeddingFunction.validate_config(config)
        return NomicEmbeddingFunction()

    def validate_config_update(self, old_config: dict, new_config: dict) -> None:
        self.validate_config(old_config)
        self.validate_config(new_config)
