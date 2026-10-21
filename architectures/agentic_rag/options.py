"""Interpret research budgets inside the architecture, not in the shared core."""

from dataclasses import dataclass, fields
import math

from rag_core.config import AppError


@dataclass(frozen=True)
class ResearchOptions:
    top_k: int = 6
    max_searches: int = 5
    max_clause_reads: int = 5
    max_model_calls: int = 14
    max_tool_calls: int = 35
    max_delegations: int = 2
    deadline_seconds: float = 150

    def __post_init__(self):
        limits = {
            "max_searches": (1, 8),
            "max_clause_reads": (1, 8),
            "max_model_calls": (1, 20),
            "max_tool_calls": (1, 50),
            "max_delegations": (0, 3),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise AppError(f"Research option {name} must be an integer {low}–{high}.")
        if not math.isfinite(self.deadline_seconds) or not 30 <= self.deadline_seconds <= 300:
            raise AppError("Research deadline must be 30–300 seconds.")


def options(settings):
    values = {
        f.name: settings.extras[f.name]
        for f in fields(ResearchOptions)
        if f.name in settings.extras
    }
    values["top_k"] = settings.top_k
    return ResearchOptions(**values)
