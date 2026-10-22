"""Measure decomposition against one plain search (retrieval only).

    uv run --all-extras python -m exercises.lab02_query_decomposition.measure          # rules, free
    uv run --all-extras python -m exercises.lab02_query_decomposition.measure --llm    # ~30 Haiku calls, < $0.01
"""

import sys

from exercises.common import Corpus, implementation, print_table, recall
from rag_core.config import Settings

DEPTH = 10  # passages fetched per search before merging


def main(use_llm=False):
    lab = implementation("exercises.lab02_query_decomposition")
    chat = None
    if use_llm:
        from rag_core.model import AnthropicChat

        chat = AnthropicChat(Settings(), max_tokens=400)
    rows = []
    for name in ("magenta", "tariffs"):
        corpus = Corpus(name)
        sums = {"dense@3": 0.0, "split@3": 0.0, "dense@6": 0.0, "split@6": 0.0}
        split = 0
        for case in corpus.cases:
            question, gold = case["question"], corpus.gold(case)
            parts = lab.llm_decompose(question, chat) if chat else lab.split_question(question)
            full = corpus.dense(question, DEPTH)
            searches = [corpus.dense(p, DEPTH) for p in parts] if len(parts) > 1 else []
            split += len(parts) > 1
            for k in (3, 6):
                sums[f"dense@{k}"] += recall(full[:k], gold)
                sums[f"split@{k}"] += recall(lab.merge(full, searches, k), gold)
        rows.append({"corpus": name, "split": f"{split}/{len(corpus.cases)}",
                     **{m: v / len(corpus.cases) for m, v in sums.items()}})
    print(f"Decomposition: {'LLM' if chat else 'rules'}")
    print_table(rows, list(rows[0]))


if __name__ == "__main__":
    main(use_llm="--llm" in sys.argv)
