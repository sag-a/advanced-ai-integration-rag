"""Extraction output: quotations refer only to the current source passage."""

from rag_core.templating import ENTITY_TYPES
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Entity(Record):
    id: str = Field(
        min_length=1,
        description="Stable canonical English identifier, e.g. hearts, magenta_moments, coupon. Reuse the supplied entity catalog.",
    )
    label: str = Field(min_length=1)
    type: str = Field(
        min_length=1,
        description="One of " + ", ".join(ENTITY_TYPES),
    )
    aliases: list[str] = Field(
        description="Greek and English labels actually denoting this entity"
    )


class Relationship(Record):
    source: str
    target: str
    statement: str = Field(
        min_length=1,
        description="Complete directed fact including all restrictions, exceptions and conditions",
    )
    polarity: Literal["positive", "negative", "conditional"]
    quote: str = Field(
        min_length=1,
        description="Exact contiguous quotation copied from this passage, including the restriction/condition",
    )


class Extraction(Record):
    entities: list[Entity]
    relationships: list[Relationship]


class SelectedRelationship(Record):
    source: str = Field(
        description="Exactly an id in THIS response's entities array. Include the entity even when it is already in the supplied catalog."
    )
    target: str = Field(
        description="Exactly an id in THIS response's entities array, never a label or an undeclared concept. Include the target entity in entities."
    )
    statement: str = Field(
        min_length=1,
        description="Complete fact including restrictions, exceptions and conditions",
    )
    polarity: Literal["positive", "negative", "conditional"]
    first_line: int = Field(
        ge=1, description="First 1-based source line supporting the entire fact"
    )
    last_line: int = Field(
        ge=1,
        description="Last inclusive source line; include all conditions and restrictions",
    )


class ChunkExtraction(Record):
    entities: list[Entity] = Field(
        description="Declare EVERY source and target used in relationships, including concepts and catalog entries. No endpoint may be absent from this array."
    )
    relationships: list[SelectedRelationship]
