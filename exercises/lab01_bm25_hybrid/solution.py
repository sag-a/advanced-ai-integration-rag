"""Reference solution: BM25 keyword scoring and reciprocal rank fusion."""

import math
import re
import unicodedata
from collections import Counter

# Keep prices ("0,3072"), plan sizes ("S", "XL") and numbers ("13888") as tokens.
TOKEN = re.compile(r"\w(?:[\w,.]*\w)?")


def tokenize(text):
    """Lowercase, remove accents (Καρδιές → καρδιες), split into word tokens."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return TOKEN.findall(plain)


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(text)) for text in documents]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average_length = sum(self.lengths) / len(self.docs)
        frequency = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        # Lucene's non-negative IDF: rare terms weigh more, common terms never go below 0.
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}

    def score(self, query):
        """One score per document, in document order. 0 means no shared term."""
        terms = tokenize(query)
        scores = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.average_length)
            total = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    total += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            scores.append(total)
        return scores

    def top(self, query, k):
        """Indexes of the k best-scoring documents, best first; documents scoring 0 are left out."""
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [i for i in order[:k] if scores[i] > 0]


def rrf(rankings, k=60, top_k=6):
    """Reciprocal rank fusion: an item gains 1 / (k + rank) from every ranking it is in.

    Ties keep the order in which items were first seen, so the first ranking wins.
    """
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])[:top_k]
