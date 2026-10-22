"""Reference solution: a semantic cache for validated answers.

`embed(texts)` returns one vector per text. The cache normalizes them itself, so any
encoder works; cosine similarity is then a dot product.
"""

import math
import re

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def normalize(vector):
    length = math.sqrt(sum(x * x for x in vector))
    if length == 0:
        raise ValueError("cannot normalize a zero vector")
    return [x / length for x in vector]


def cosine(a, b):
    """Cosine similarity of two already-normalized vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def numbers_match(question, cached_question):
    """Every number in the new question must also be in the cached one.

    "24 XL with 420 minutes" must not reuse the answer about 500 minutes, however
    similar the vectors are. A question without numbers passes.
    """
    return set(NUMBER.findall(question)) <= set(NUMBER.findall(cached_question))


class SemanticCache:
    def __init__(self, embed, threshold=0.9, guard=None):
        """guard(question, cached_question) -> bool can veto a match, e.g. numbers_match."""
        self.embed, self.threshold, self.guard = embed, threshold, guard
        self.entries = []  # (question, normalized vector, answer)

    def add(self, question, answer, status="answered"):
        """Store only validated answers. Refusals and clarifications depend on context."""
        if status != "answered":
            return False
        vector = normalize(self.embed([question])[0])
        self.entries.append((question, vector, answer))
        return True

    def lookup(self, question):
        """The most similar stored entry that reaches the threshold and passes the guard.

        Returns {"answer", "cached_question", "similarity"}, or None.
        """
        if not self.entries:
            return None
        vector = normalize(self.embed([question])[0])
        ranked = sorted(((cosine(vector, v), q, a) for q, v, a in self.entries), reverse=True)
        for similarity, cached_question, answer in ranked:
            if similarity < self.threshold:
                return None
            if self.guard is None or self.guard(question, cached_question):
                return {"answer": answer, "cached_question": cached_question, "similarity": similarity}
        return None
