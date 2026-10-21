"""Validate source numbers and format the accepted answer's PDF references."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import AppError


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    sources: list[int]
    status: Literal["answered", "insufficient_evidence", "clarification_needed"]


def insufficient(question):
    # A Greek offer name inside an English question must not switch the refusal
    # language. Use the dominant script; very short mixed queries remain heuristic.
    greek = len(re.findall(r"[\u0370-\u03ff\u1f00-\u1fff]", question))
    latin = len(re.findall(r"[A-Za-z]", question))
    if greek > latin:
        return "Τα παρεχόμενα έγγραφα δεν δίνουν επαρκείς πληροφορίες για να απαντήσω σε αυτή την ερώτηση."
    return "The supplied material does not provide enough evidence to answer this question."


def format_answer(answer: Answer, source_map: list[dict], question: str):
    """Reject unknown references, then build a footer from original metadata.

    This checks citation membership, not whether a passage supports a claim.
    """
    allowed = {source["number"] for source in source_map}
    inline = {int(number) for number in re.findall(r"\[(\d+)\]", answer.answer)}
    used = set(answer.sources) | inline
    if (
        not answer.answer.strip()
        or len(set(answer.sources)) != len(answer.sources)
        or not used <= allowed
        or (answer.status == "answered" and not used)
        or (answer.status == "clarification_needed" and used)
    ):
        raise AppError(
            "Invalid answer or citation references; no unsupported answer was accepted."
        )
    cited = [source for source in source_map if source["number"] in used]
    # A fixed refusal prevents unsupported advice from reaching the final answer.
    if answer.status == "insufficient_evidence":
        text = insufficient(question)
    else:
        text = re.sub(r"\[\d+\]", "", answer.answer).strip()

    # Valid inline references become metadata-backed footers. Unknown ones were
    # rejected above; they are never silently removed.
    for source in cited:
        pages = str(source["page_start"])
        if source["page_start"] != source["page_end"]:
            pages += f"–{source['page_end']}"
        text += f"\n[{source['number']}] {source['source']} · PDF p. {pages}"
        if source["clause"]:
            text += f" · §{source['clause']}"
    return text, cited
