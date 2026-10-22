"""Lab 1: BM25 keyword scoring and reciprocal rank fusion. Fill in the TODOs."""

import math  # noqa: F401  (you will need it for the IDF)
import re
import unicodedata  # noqa: F401  (you will need it for accent folding)
from collections import Counter  # noqa: F401  (handy for term counts)

# Keep prices ("0,3072"), plan sizes ("S", "XL") and numbers ("13888") as tokens.
TOKEN = re.compile(r"\w(?:[\w,.]*\w)?")


def tokenize(text):
    """Lowercase, remove accents (Καρδιές → καρδιες), split into word tokens.

    Hint: unicodedata.normalize("NFKD", ...) separates letters from their accents;
    drop the characters for which unicodedata.combining(ch) is true.
    """
    # TODO 1: return TOKEN.findall(...) over the lowercased, accent-free text.
    raise NotImplementedError


class BM25:
    """Okapi BM25 over a fixed list of documents.

    score(t, d) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
    idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
    """

    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        # TODO 2: tokenize every document and store, per document, its term counts
        # and its length. Compute the average length and an IDF value per term.
        raise NotImplementedError

    def score(self, query):
        """One score per document, in document order. 0 means no shared term."""
        # TODO 3: sum the BM25 contribution of every query term in every document.
        raise NotImplementedError

    def top(self, query, k):
        """Indexes of the k best-scoring documents, best first; documents scoring 0 are left out."""
        # TODO 4: sort by score (ties: lower index first) and drop zero scores.
        raise NotImplementedError


def rrf(rankings, k=60, top_k=6):
    """Reciprocal rank fusion: an item gains 1 / (k + rank) from every ranking it is in.

    Ranks start at 1. Ties keep the order in which items were first seen.
    """
    # TODO 5: add up the scores across rankings and return the top_k items.
    raise NotImplementedError
