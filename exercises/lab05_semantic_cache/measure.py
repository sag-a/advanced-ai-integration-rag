"""Try your cache with the course encoder: paraphrases should hit, near-misses must not.

Offline: uses the local Nomic encoder, no API key and no Docker.

    uv run --all-extras python -m exercises.lab05_semantic_cache.measure
"""

from exercises.common import implementation, print_table
from rag_core.embeddings import PREFIXES, LocalEncoder

# Questions already answered and validated (the answer is just a label here).
SEEDS = {
    "expiry": "Πότε λήγουν οι Καρδιές;",
    "transfer": "Can I transfer my Hearts to someone else?",
    "join": "How do I join Magenta Moments?",
    "24xl-500": "Which Double Play 24 XL plan has 500 free minutes and what does it cost?",
    "24s-12m": "How much does COSMOTE Double Play 24 S cost per month on a 12-month contract?",
    "stacking": "Can I combine a Magenta Moments offer with another discount?",
}
# (question, the seed it should reuse, or None when it needs a fresh answer)
PROBES = [
    ("When do Hearts expire?", "expiry"),
    ("Μέχρι πότε ισχύουν οι Καρδιές μου;", "expiry"),
    ("Is it possible to give my Hearts to a friend?", "transfer"),
    ("How can I sign up for Magenta Moments?", "join"),
    ("What's the monthly price of Double Play 24 S with a one-year commitment?", "24s-12m"),
    ("Can I stack a rewards-app deal with another coupon?", "stacking"),
    # Near-misses: similar words, a different answer.
    ("Which Double Play 24 XL plan has 420 free minutes and what does it cost?", None),
    ("How much does COSMOTE Double Play 24 S cost per month on an 18-month contract?", None),
    ("How do I leave Magenta Moments?", None),
    ("When do my coupons expire?", None),
    ("Can I transfer my discount code to someone else?", None),
]
THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def main():
    lab = implementation("exercises.lab05_semantic_cache")
    encoder = LocalEncoder()

    def embed(texts):
        return [list(map(float, v)) for v in encoder.encode([PREFIXES["query"] + t for t in texts])]

    nearest = lab.SemanticCache(embed, threshold=-1.0)
    for label, question in SEEDS.items():
        nearest.add(question, label)
    print("Closest cached question for each probe:")
    for question, want in PROBES:
        hit = nearest.lookup(question)
        print(f"  {hit['similarity']:.3f}  {'should reuse ' + want if want else 'must miss':22} → {hit['answer']:9} {question}")

    rows = []
    for guard_name, guard in (("none", None), ("numbers", lab.numbers_match)):
        for threshold in THRESHOLDS:
            cache = lab.SemanticCache(embed, threshold=threshold, guard=guard)
            for label, question in SEEDS.items():
                cache.add(question, label)
            hits = [(want, cache.lookup(question)) for question, want in PROBES]
            served = [(want, hit) for want, hit in hits if hit]
            correct = sum(want == hit["answer"] for want, hit in served)
            rows.append({"guard": guard_name, "threshold": threshold, "correct reuse": f"{correct}/6",
                         "wrong answer served": len(served) - correct})
    print()
    print_table(rows, list(rows[0]))


if __name__ == "__main__":
    main()
