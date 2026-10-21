"""Shared sentences and support document for the numbered local demos."""

from importlib.resources import files

SENTENCES = [
    "My internet connection stopped working.",
    "I can no longer connect to the internet.",
    "My internet connection is very slow.",
    "I want to cancel my subscription.",
    "Η σύνδεσή μου στο διαδίκτυο σταμάτησε να λειτουργεί.",
]
SENTENCE_LABELS = ["EN original", "EN paraphrase", "EN related", "EN unrelated", "EL paraphrase"]

def load_support_text() -> str:
    """Read the same bundled UTF-8 document even outside the source directory."""
    return files("modules.c01_embeddings").joinpath("telecom_support.txt").read_text(encoding="utf-8")
