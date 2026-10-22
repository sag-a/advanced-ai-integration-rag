"""Lab 2: split multi-part questions and merge the searches fairly. Fill in the TODOs."""

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

    Split on SEPARATOR, strip " .?;" from each part, drop parts shorter than
    MIN_WORDS words, keep at most MAX_PARTS. One part or none → [question].
    """
    # TODO 1
    raise NotImplementedError


def merge(full, parts, k):
    """Round-robin merge of ranked chunk-ID lists.

    Take the full question's best hit, then each part's best hit, then everyone's
    second hit, and so on. Skip IDs already taken. Return at most k IDs.
    """
    # TODO 2
    raise NotImplementedError


class SubQuestions(BaseModel):
    questions: list[str] = Field(description="Standalone sub-questions, in the user's language")


# TODO 3: write the instructions for the model. It must return standalone
# sub-questions: no pronouns ("they", "it", "τον"), repeat the subject instead.
# Say what to do with a question that asks only one thing, and set a maximum.
DECOMPOSE_PROMPT = "TODO"


def llm_decompose(question, chat):
    """Ask the model for standalone sub-questions; fall back to [question] on anything odd.

    `chat.call(SubQuestions, DECOMPOSE_PROMPT, {"question": question})` returns
    (parsed SubQuestions, usage). Strip blanks, drop duplicates (keep order), keep
    at most MAX_PARTS. One part or none → [question].
    """
    # TODO 4
    raise NotImplementedError
