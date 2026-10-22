"""Reference solution: validate an answer's citations and check its numbers."""

import re

# A citation is [n] or [n, m, ...] with one or two digits. "[2024]" is a year, not a citation.
CITATION = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")
# Numbers as written in the documents: "46,05", "0,3072", "1.600", "24", "13888".
# A clock time such as "12:00" is one token, so it can't vouch for "12 months".
NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


class CitationError(ValueError):
    """The answer can't be accepted; the message says why."""


def inline_citations(text):
    """Source numbers cited inline, in order of first appearance."""
    numbers = []
    for group in CITATION.findall(text):
        for part in group.split(","):
            n = int(part)
            if n not in numbers:
                numbers.append(n)
    return numbers


def validate(text, sources, status, available):
    """Return the cited source numbers, or raise CitationError.

    text: the answer; sources: the numbers the model listed separately; status:
    "answered", "insufficient_evidence" or "clarification_needed"; available: the
    numbers of the passages it was given.
    """
    if not text.strip():
        raise CitationError("empty answer")
    if len(set(sources)) != len(sources):
        raise CitationError("duplicate source numbers")
    cited = list(dict.fromkeys([*sources, *inline_citations(text)]))
    unknown = [n for n in cited if n not in available]
    if unknown:
        raise CitationError(f"unknown source numbers: {unknown}")
    if status == "answered" and not cited:
        raise CitationError("an answer must cite at least one source")
    if status == "clarification_needed" and cited:
        raise CitationError("a clarification question must not cite sources")
    return cited


def strip_citations(text):
    """Remove citation markers; leave other bracketed text such as [2024] alone."""
    return re.sub(r"\s+([.,;:!?])", r"\1", CITATION.sub("", text)).strip()


def _normalized(number):
    # "46,05" and "46.05" are the same price; "1.600" is one thousand six hundred.
    return number.replace(",", ".")


def unsupported_numbers(text, passages):
    """Numbers in the answer that appear in none of the cited passages.

    A cheap grounding check: citation membership says nothing about support, but a
    price or a day count the passages never mention is almost certainly wrong.
    Citation markers are ignored.
    """
    found = {_normalized(n) for passage in passages for n in NUMBER.findall(passage)}
    seen = []
    for n in NUMBER.findall(CITATION.sub("", text)):
        if _normalized(n) not in found and n not in seen:
            seen.append(n)
    return seen
