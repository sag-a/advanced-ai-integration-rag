"""Trusted research configuration and packaged skill descriptions."""

from importlib.resources import files

from rag_core.config import AppError
from rag_core.templating import for_settings, render
from .options import options

RESEARCH = render("research").text
HELPER = render("helper").text


def skill_files():
    root = files("architectures").joinpath("agentic_rag", "skills")
    return {
        f"/skills/{name}/SKILL.md": root.joinpath(name, "SKILL.md").read_text(encoding="utf-8")
        for name in ("membership", "hearts", "coupons")
    }


def skill_catalog():
    catalog = []
    for path, text in skill_files().items():
        parts = text.split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            raise AppError(f"Missing skill front-matter: {path}")
        metadata = dict(line.split(":", 1) for line in parts[1].splitlines() if ":" in line)
        if not all(metadata.get(key, "").strip() for key in ("name", "description")):
            raise AppError(f"Missing skill name/description: {path}")
        catalog.append({key: metadata[key].strip() for key in ("name", "description")})
    return catalog


def research_prompts(settings, *, runtime=False):
    budget = options(settings)
    return {
        "research": for_settings(
            "research",
            settings,
            top_k=budget.top_k,
            max_searches=budget.max_searches,
            max_delegations=budget.max_delegations,
            skills=skill_catalog(),
            runtime=runtime,
        ),
        "helper": for_settings("helper", settings),
    }
