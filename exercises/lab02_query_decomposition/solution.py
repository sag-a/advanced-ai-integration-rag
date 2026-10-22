"""Reference solution: split multi-part questions and merge the searches fairly."""

import re

from pydantic import BaseModel, Field

MAX_PARTS = 4
MIN_WORDS = 3
# Separators between the parts of a question, in English and Greek.
SEPARATOR = re.compile(
    r"\s*(?:[;:?]|,\s*(?:and|και)\s+|,\s+|\s+(?:and|και)\s+(?=(?:how|what|can|is|whose|τι|πώς|πότε|μπορ)))\s*",
    re.IGNORECASE,
)


def split_question(question):
    """Rule-based decomposition. Returns [question] when there is nothing to split.

    Parts shorter than MIN_WORDS are dropped (they rarely make sense alone), and at
    most MAX_PARTS are kept.
    """
    parts = [p.strip(" .?;") for p in SEPARATOR.split(question)]
    parts = [p for p in parts if len(p.split()) >= MIN_WORDS][:MAX_PARTS]
    return parts if len(parts) > 1 else [question]


def merge(full, parts, k):
    """Round-robin: the full question's best hit, then each part's best hit, then
    everyone's second hit, and so on. Duplicates are skipped; at most k IDs."""
    rankings = [full, *parts]
    merged = []
    for rank in range(max(map(len, rankings), default=0)):
        for ranking in rankings:
            if rank < len(ranking) and ranking[rank] not in merged:
                merged.append(ranking[rank])
    return merged[:k]


class SubQuestions(BaseModel):
    questions: list[str] = Field(description="Standalone sub-questions, in the user's language")


DECOMPOSE_PROMPT = (
    "Split the user's question into the separate questions it asks, so each can be searched on "
    "its own. Every sub-question must stand alone: repeat the subject (for example 'the discount "
    "code', 'Hearts', 'COSMOTE Double Play 24 L') instead of using pronouns. Keep the user's "
    "language. If the question asks only one thing, return it unchanged as the only item. "
    "Return at most 4."
)


def llm_decompose(question, chat):
    """Ask the model for standalone sub-questions; fall back to [question] on anything odd.

    `chat.call(schema, system, payload)` returns (parsed object, usage).
    """
    parsed, _ = chat.call(SubQuestions, DECOMPOSE_PROMPT, {"question": question})
    parts = list(dict.fromkeys(p.strip() for p in parsed.questions if p.strip()))[:MAX_PARTS]
    return parts if len(parts) > 1 else [question]
