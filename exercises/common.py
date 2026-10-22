"""Helpers for the labs' measure scripts: corpora, cases, dense search and recall.

Measuring needs Chroma running and both collections prepared, but no API key.
"""

from dataclasses import replace
import importlib
import os
from pathlib import Path

from evals.run import evidence_ids, load_cases
from rag_core.config import Settings
from rag_core.store import connect, open_collection, read_documents, retrieve

CORPORA = {
    "magenta": (Settings(), "evals/datasets/cases.jsonl"),
    "tariffs": (
        Settings(corpus=Path("data/corpus-tariffs"), collection="cosmote_tariffs_v1"),
        "evals/datasets/tariffs.jsonl",
    ),
}


def implementation(package):
    """The student's starter, or the solution with LAB_TARGET=solution."""
    return importlib.import_module(f"{package}.{os.environ.get('LAB_TARGET', 'starter')}")


class Corpus:
    """One prepared collection plus its answerable, history-free evaluation cases."""

    def __init__(self, name):
        self.name = name
        self.settings, cases_path = CORPORA[name]
        self.collection = open_collection(connect(self.settings), self.settings)
        self.documents = read_documents(self.collection)
        self.cases = [
            case
            for case in load_cases(Path(cases_path))
            if case["expected_status"] == "answered" and not case["history"]
        ]

    def gold(self, case):
        return set(evidence_ids(case, self.documents))

    def dense(self, question, k):
        """Chunk IDs from one vector search, best first."""
        settings = replace(self.settings, top_k=k)
        return [d.id for d in retrieve(self.collection, settings, question).documents]


def recall(selected, gold):
    return len(set(selected) & gold) / len(gold)


def print_table(rows, columns):
    widths = [max(len(str(c)), *(len(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in columns]
    print("  ".join(str(c).ljust(w) for c, w in zip(columns, widths)))
    for row in rows:
        cells = [f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        print("  ".join(cell.ljust(w) for cell, w in zip(cells, widths)))
