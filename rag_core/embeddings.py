"""Pinned local Nomic encoder and configured Chroma embedding function."""

from pathlib import Path
from copy import deepcopy
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function


EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
REVISION = "1066b6599d099fbb93dfcb64f9c37a7c9e503e85"
CODE_REPOSITORY = "nomic-ai/nomic-bert-2048"
CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
DIMENSIONS = 768
MAX_TOKENS = 512
PREFIXES = {"document": "search_document: ", "query": "search_query: "}
CONTRACT = {
    "model": EMBEDDING_MODEL,
    "revision": REVISION,
    "code_repository": CODE_REPOSITORY,
    "code_revision": CODE_REVISION,
    "dimensions": DIMENSIONS,
    "pooling": "attention-mask-mean+l2",
    "prefixes": PREFIXES,
    "max_tokens": MAX_TOKENS,
    "truncation": False,
}


def load_model(*, allow_download=False):
    """Load pinned weights AND official MoE code, validating every weight key.

    The native Transformers Nomic class is a dense model, not this MoE model.
    The upstream custom from_pretrained method can ignore revision/strictness
    options, so resolve the snapshot first and load its safetensors strictly.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    snapshot = snapshot_download(
        EMBEDDING_MODEL,
        revision=REVISION,
        local_files_only=not allow_download,
        allow_patterns=["*.json", "*.safetensors", "*.model"],
    )
    model_class = get_class_from_dynamic_module(
        CODE_REPOSITORY + "--modeling_hf_nomic_bert.NomicBertModel",
        EMBEDDING_MODEL,
        code_revision=CODE_REVISION,
        local_files_only=not allow_download,
    )

    class CpuNomicModel(model_class):
        # The pinned upstream forward still calls this Transformers 4 helper.
        # Transformers 5 removed it. This encoder needs only a 2D padding mask.
        def get_extended_attention_mask(self, attention_mask, input_shape):
            if attention_mask.ndim != 2:
                raise ValueError("Expected a batch-by-token attention mask.")
            mask = attention_mask[:, None, None, :].to(dtype=self.dtype)
            return (1.0 - mask) * torch.finfo(self.dtype).min

    config = model_class.config_class.from_json_file(str(Path(snapshot) / "config.json"))
    model = CpuNomicModel(config, add_pooling_layer=False)
    weights = load_file(str(Path(snapshot) / "model.safetensors"), device="cpu")
    model.load_state_dict(weights, strict=True)
    model = model.to("cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    return tokenizer, model


class LocalEncoder:
    """CPU inference with mean pooling and L2 normalization; no truncation."""

    def __init__(self, *, allow_download=False):
        self.allow_download = allow_download
        self.tokenizer = self.model = None

    def load(self):
        if self.model is None:
            self.tokenizer, self.model = load_model(allow_download=self.allow_download)
        return self

    def encode(self, texts):
        import torch
        import torch.nn.functional as functional

        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Provide nonempty embedding inputs.")
        self.load()
        encoded = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > MAX_TOKENS:
            raise ValueError(
                f"Input exceeds {MAX_TOKENS} model tokens including the prefix; split it explicitly."
            )
        with torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return functional.normalize(pooled, p=2, dim=1).cpu().numpy()


# Chroma calls __call__ for documents and embed_query for query_texts.
# https://docs.trychroma.com/docs/embeddings/embedding-functions
@register_embedding_function
class NomicEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        # Loading is deferred until Chroma actually needs an embedding.
        self.encoder = LocalEncoder(allow_download=False)

    def __call__(self, input: Documents) -> Embeddings:
        return list(self.encoder.encode([PREFIXES["document"] + text for text in input]))

    def embed_query(self, input: Documents) -> Embeddings:
        return list(self.encoder.encode([PREFIXES["query"] + text for text in input]))

    @staticmethod
    def name() -> str:
        return "rag-course-nomic-v2-moe"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    def get_config(self) -> dict:
        return {"contract": deepcopy(CONTRACT)}

    @staticmethod
    def validate_config(config: dict) -> None:
        if config != {"contract": CONTRACT}:
            raise ValueError(
                "Incompatible Nomic configuration; use a new collection and re-embed."
            )

    @staticmethod
    def build_from_config(config: dict) -> "NomicEmbeddingFunction":
        NomicEmbeddingFunction.validate_config(config)
        return NomicEmbeddingFunction()

    def validate_config_update(self, old_config: dict, new_config: dict) -> None:
        self.validate_config(old_config)
        self.validate_config(new_config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare/check pinned local Nomic weights and code."
    )
    parser.add_argument(
        "--download", action="store_true", help="Explicitly allow downloading pinned files"
    )
    options = parser.parse_args()
    LocalEncoder(allow_download=options.download).load()
    print("Pinned Nomic model and tokenizer are ready on CPU.")
