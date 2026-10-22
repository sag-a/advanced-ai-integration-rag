"""Instructor script: record the Jev gate's probability for every lab question.

Runs Jev RAG's real retrieval (20 dense candidates → bge top 6 → Jev gate) and writes
gate_scores.json, which students use offline. About 45 Jev calls, under $0.01.

    uv run --all-extras python -m exercises.lab04_gate_calibration.record
"""

import json
from pathlib import Path

from architectures.jev_rag import jev
from architectures.jev_rag.retriever import CANDIDATES
from evals.run import load_cases
from exercises.common import CORPORA, Corpus
from rag_core import reranker

OUT = Path(__file__).with_name("gate_scores.json")
# On-topic questions the documents do not answer; checked against the extracted text.
UNANSWERABLE = {
    "magenta": [
        ("u01-support-email", "What is the customer service email address for Magenta Moments?"),
        ("u02-minimum-age", "Ποια είναι η ελάχιστη ηλικία για να γίνω μέλος του Magenta Moments;"),
        ("u03-spin-cost", "How many Hearts does one spin of the wheel cost?"),
        ("u04-partners-thessaloniki", "Ποιοι συνεργάτες του προγράμματος υπάρχουν στη Θεσσαλονίκη;"),
        ("u05-start-date", "When did the Magenta Moments programme start?"),
        ("u06-hearts-per-euro", "How many Hearts do I get for each euro I spend at Germanos?"),
    ],
    "tariffs": [
        ("u07-installation-fee", "How much does installation of a COSMOTE broadband line cost?"),
        ("u08-1000-mbps", "What does the COSMOTE Double Play 1000 Mbps plan cost?"),
        ("u09-roaming-italy", "What are the roaming charges in Italy?"),
        ("u10-upload-speed", "What is the upload speed of Fiberspeed 200?"),
        ("u11-netflix", "Is Netflix included in any Double Play plan?"),
        ("u12-free-router", "Does the 24 L plan come with a free Wi-Fi router?"),
    ],
}
EXISTING_UNANSWERABLE = {"s09-private-account", "s10-external-policy", "t14-tv-price"}


def gate_probability(corpus, question):
    ids = corpus.dense(question, CANDIDATES)
    by_id = {d.id: d for d in corpus.documents}
    texts = [by_id[i].page_content for i in ids]
    scores = reranker.score(question, texts)
    kept = [texts[i] for i in sorted(range(len(texts)), key=lambda i: -scores[i])[: corpus.settings.top_k]]
    probability, cost = jev.gate(corpus.settings, question, kept)
    return probability, cost


def main():
    rows, total = [], 0.0
    for name in ("magenta", "tariffs"):
        corpus = Corpus(name)
        everything = {c["id"]: c for c in load_cases(Path(CORPORA[name][1]))}
        questions = [(c["id"], c["question"], True) for c in corpus.cases]
        questions += [(i, everything[i]["question"], False) for i in sorted(EXISTING_UNANSWERABLE & everything.keys())]
        questions += [(i, q, False) for i, q in UNANSWERABLE[name]]
        for identifier, question, answerable in questions:
            probability, cost = gate_probability(corpus, question)
            total += cost
            rows.append({"id": identifier, "corpus": name, "question": question,
                         "answerable": answerable, "gate_probability": round(probability, 4)})
            print(f"{identifier:36} answerable={answerable!s:5} P={probability:.2f}")
    # Alternate within each label so dev and held-out get the same mix.
    for label in (True, False):
        for position, row in enumerate(r for r in rows if r["answerable"] is label):
            row["split"] = "dev" if position % 2 == 0 else "heldout"
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} questions → {OUT.name}; Jev cost ${total:.4f}")


if __name__ == "__main__":
    main()
