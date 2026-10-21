"""Evaluation labels stay here; architectures receive Question, never Case."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["answered", "insufficient_evidence", "clarification_needed"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class History(StrictModel):
    role: Literal["human", "ai"]
    text: str = Field(min_length=1)


class Question(StrictModel):
    id: str
    question: str = Field(min_length=1)
    history: list[History] = Field(default_factory=list)


class Evidence(StrictModel):
    document: str
    page: int = Field(ge=1)
    clause: str


class Case(Question):
    language: Literal["el", "en"]
    type: Literal["factual", "followup", "multihop", "global", "version", "abstain", "repeated"]
    split: Literal["dev", "heldout"]
    expected_status: Status
    expected_evidence: list[Evidence]
    expected_facts: list[str]

    def request(self):
        return Question(id=self.id, question=self.question, history=self.history)


class Passage(StrictModel):
    id: str
    text: str
    source: str
    pdf_sha256: str
    page_start: int
    page_end: int
    clause: str

    @classmethod
    def from_document(cls, document):
        tags = document.metadata
        return cls(
            id=document.id,
            text=document.page_content,
            **{
                key: tags[key]
                for key in ("source", "pdf_sha256", "page_start", "page_end", "clause")
            },
        )


class RunOutput(StrictModel):
    search_query: str
    answer: str = ""
    status: Status | None = None
    passages: list[Passage] = Field(default_factory=list)
    citations: list[Passage] = Field(default_factory=list)
    observation: dict = Field(default_factory=dict)
    elapsed_seconds: float = 0
    trace_id: str | None = None
    trace_url: str | None = None
    trace_warning: str | None = None
    retrieval_complete: bool = False
    error: str | None = None


class FactLabel(StrictModel):
    index: int = Field(ge=0)
    status: Literal["met", "missing", "contradicted"]
    quote: str = Field(
        default="",
        description="Exact answer quote for met/contradicted; empty when missing. Never quote the reference or required fact instead of the answer.",
    )


class ClaimLabel(StrictModel):
    quote: str = Field(
        min_length=1,
        description="Exact verbatim span of a factual claim in the answer, excluding citation footers.",
    )
    correctness: Literal["correct", "incorrect", "unverifiable"]
    grounded: bool
    cited: bool


class Judgment(StrictModel):
    facts: list[FactLabel]
    claims: list[ClaimLabel]
