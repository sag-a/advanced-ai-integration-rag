"""Pinned local cross-encoder that re-scores retrieved passages against the question.

BAAI/bge-reranker-v2-m3 is free (Apache-2.0), multilingual including Greek, and runs
on CPU. Unlike the embedder, it reads the question and a passage together, so it
can judge relevance more precisely; it is too slow to run over the whole corpus,
which is why it only reorders the vector search's candidates.
"""

import argparse
from functools import lru_cache

from rag_core.config import AppError

MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
FILES = [
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
MAX_TOKENS = 512
DOWNLOAD = "uv run --all-extras python -m rag_core.reranker --download"


@lru_cache(maxsize=2)
def load(*, allow_download=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging

    try:
        path = snapshot_download(
            MODEL, revision=REVISION, allow_patterns=FILES, local_files_only=not allow_download
        )
    except Exception as error:
        raise AppError(f"Reranker weights are not downloaded (about 2.3 GB). Run: {DOWNLOAD}") from error
    logging.disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tokenizer, model


def score(question, texts):
    """One relevance score per passage; higher means more relevant to the question."""
    import torch

    tokenizer, model = load()
    batch = tokenizer(
        [question] * len(texts),
        texts,
        padding=True,
        truncation="only_second",  # never cut the question
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    with torch.no_grad():
        return model(**batch).logits.view(-1).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the pinned reranker weights.")
    parser.add_argument("--download", action="store_true", required=True)
    parser.parse_args()
    load(allow_download=True)
    print(f"{MODEL} @ {REVISION[:8]} is ready.")
