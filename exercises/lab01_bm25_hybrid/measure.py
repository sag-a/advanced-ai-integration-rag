"""Measure your BM25 and RRF on the course corpora (retrieval only, no API key).

    uv run --all-extras python -m exercises.lab01_bm25_hybrid.measure
    LAB_TARGET=solution uv run --all-extras python -m exercises.lab01_bm25_hybrid.measure
"""

from exercises.common import Corpus, implementation, print_table, recall

CANDIDATES = 20


def main(k=3):
    lab = implementation("exercises.lab01_bm25_hybrid")
    rows = []
    for name in ("magenta", "tariffs"):
        corpus = Corpus(name)
        ids = [d.id for d in corpus.documents]
        bm25 = lab.BM25([d.page_content for d in corpus.documents])
        scores = {"dense": [], "bm25": [], "hybrid": []}
        for case in corpus.cases:
            gold = corpus.gold(case)
            dense = corpus.dense(case["question"], CANDIDATES)
            keywords = [ids[i] for i in bm25.top(case["question"], CANDIDATES)]
            scores["dense"].append(recall(dense[:k], gold))
            scores["bm25"].append(recall(keywords[:k], gold))
            scores["hybrid"].append(recall(lab.rrf([dense, keywords], top_k=k), gold))
        rows.append({"corpus": name, "cases": len(corpus.cases),
                     **{f"{m} recall@{k}": sum(v) / len(v) for m, v in scores.items()}})
    print_table(rows, list(rows[0]))
    print(f"\nCourse baselines at k={k}: TF-IDF hybrid on tariffs 0.731, dense on tariffs 0.564.")


if __name__ == "__main__":
    main()
