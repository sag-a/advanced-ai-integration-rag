"""Lab 3: validate an answer's citations and check its numbers. Fill in the TODOs."""

import re

# A citation is [n] or [n, m, ...] with one or two digits. "[2024]" is a year, not a citation.
# TODO 1: write the pattern. The group should capture "3" from "[3]" and "1, 3" from "[1, 3]".
CITATION = re.compile(r"TODO")
# Numbers as written in the documents: "46,05", "0,3072", "1.600", "24", "13888".
# A clock time such as "12:00" is one token, so it can't vouch for "12 months".
NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


class CitationError(ValueError):
    """The answer can't be accepted; the message says why."""


def inline_citations(text):
    """Source numbers cited inline, in order of first appearance, without repeats."""
    # TODO 2
    raise NotImplementedError


def validate(text, sources, status, available):
    """Return the cited source numbers, or raise CitationError with a reason.

    text: the answer; sources: the numbers the model listed separately; status:
    "answered", "insufficient_evidence" or "clarification_needed"; available: the
    numbers of the passages it was given.

    Reject, with a message containing the word in brackets:
    an empty answer [empty]; repeated numbers in `sources` [duplicate]; a cited
    number that isn't available [unknown]; "answered" with no citation [at least one];
    "clarification_needed" with any citation [clarification].
    Cited = `sources` plus inline citations, in that order, without repeats.
    """
    # TODO 3
    raise NotImplementedError


def strip_citations(text):
    """Remove citation markers and the space before them; leave [2024] alone.

    "one year [1]." → "one year."
    """
    # TODO 4
    raise NotImplementedError


def unsupported_numbers(text, passages):
    """Numbers in the answer that appear in none of the cited passages.

    Treat "46,05" and "46.05" as the same number. Ignore citation markers. Return
    each unsupported number once, in order of appearance, as written in the answer.
    """
    # TODO 5
    raise NotImplementedError
