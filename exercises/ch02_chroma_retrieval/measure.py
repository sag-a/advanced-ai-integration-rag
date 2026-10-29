"""Run the three exercise questions with the real Nomic encoder and show what comes back.

In-memory Chroma: no Docker server, no API key. Loading the encoder takes about 9 s.
"""

from exercises.common import implementation


def show(title, setting, chunks, verdict):
    print(f"\n=== {title} · {setting}")
    for number, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        print(f"  {number}. [{meta['category']}, v{meta['version']}, {meta['status']}] {chunk['text']}")
    print(f"  → {verdict}")


def main():
    lab = implementation("exercises.ch02_chroma_retrieval")
    collection = lab.make_collection()

    chunks = lab.missing_information(collection)
    found = {c["id"] for c in chunks}
    both = {"refund-rule-v2", "refund-exception-v2"} <= found
    show("1. Missing information", f"N_RESULTS={lab.N_RESULTS}", chunks,
         "OK: the rule and its exception are both here" if both
         else "MISSING: the answer needs the refund rule AND its exception")

    chunks = lab.wrong_category(collection)
    categories = sorted({c["metadata"]["category"] for c in chunks})
    show("2. Wrong category", f"where={lab.BILLING_FILTER}", chunks,
         "OK: billing results only" if categories == ["billing"]
         else f"WRONG: a billing question answered from {categories} documents")

    chunks = lab.outdated_information(collection)
    outdated = [c["id"] for c in chunks if c["metadata"]["status"] != "current"]
    show("3. Outdated information", f"where={lab.POLICY_FILTER}", chunks,
         f"OUTDATED: {outdated} contradicts the current policy" if outdated
         else "OK: current policies only")


if __name__ == "__main__":
    main()
