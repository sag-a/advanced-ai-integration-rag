"""Render trusted prompt configuration. Runtime questions stay in JSON payloads."""

from copy import deepcopy
from dataclasses import dataclass

from jinja2 import PackageLoader, PrefixLoader, StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment

from .config import AppError
from .corpus import digest


ENV = SandboxedEnvironment(
    loader=PrefixLoader(
        {
            "core": PackageLoader("rag_core", "templates"),
            "agentic": PackageLoader("architectures", "agentic_rag/templates"),
            "graph": PackageLoader("architectures", "graph_rag/templates"),
        }
    ),
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)

TEMPLATES = {
    "rewrite": "core/rewrite.j2",
    "answer": "core/answer.j2",
    "research": "agentic/research.j2",
    "helper": "agentic/helper.j2",
    "extraction": "graph/extraction.j2",
}

DEFAULTS = {
    "rewrite": {"max_subjects": None, "mention_history_truncation": True},
    "answer": {"answer_style": "concise", "response_language": "match_user", "programs": []},
    "research": {
        "helper_name": "policy-researcher",
        "top_k": 6,
        "max_searches": 5,
        "max_delegations": 2,
        "skills": [],
        "show_skills": False,
        "runtime": False,
    },
    "helper": {},
    "extraction": {"source_language": "Greek", "entity_types": []},
}

# Public --option variables; skills and runtime are derived by trusted application code.
PROMPT_OPTIONS = set().union(*(values.keys() for values in DEFAULTS.values())) - {
    "skills",
    "runtime",
}
ARCHITECTURE_OPTIONS = {
    "graph_index",
    "resume",
    "max_clause_reads",
    "max_model_calls",
    "max_tool_calls",
    "deadline_seconds",
}
ENTITY_TYPES = (
    "program",
    "person",
    "currency",
    "coupon",
    "action",
    "condition",
    "duration",
    "organization",
    "concept",
)


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    prompt_id: str
    template_id: str


def validate_options(settings):
    """Catch classroom option typos before connecting to services."""
    if settings.architecture not in {"classic_rag", "graph_rag", "agentic_rag"}:
        return  # Other plugins own their option contracts.
    unknown = settings.extras.keys() - PROMPT_OPTIONS - ARCHITECTURE_OPTIONS
    if unknown:
        raise AppError("Unknown --option: " + ", ".join(sorted(unknown)))
    _validate({key: value for key, value in settings.extras.items() if key in PROMPT_OPTIONS})


def _validate(values):
    for name, value in values.items():
        if name == "max_subjects":
            valid = value is None or (type(value) is int and 1 <= value <= 10)
        elif name in {"mention_history_truncation", "show_skills", "runtime"}:
            valid = type(value) is bool
        elif name in {"top_k", "max_searches", "max_delegations"}:
            low, high = {"top_k": (1, 20), "max_searches": (1, 8), "max_delegations": (0, 3)}[
                name
            ]
            valid = type(value) is int and low <= value <= high
        elif name == "answer_style":
            valid = value in ("concise", "detailed")
        elif name == "helper_name":
            valid = value in (
                "policy-researcher",
                "general-purpose",
                "general-purpose (the constrained PDF researcher)",
            )
        elif name in {"programs", "entity_types"}:
            valid = isinstance(value, list) and all(
                isinstance(v, str) and v.strip() for v in value
            )
            if valid and name == "entity_types":
                valid = set(value) <= set(ENTITY_TYPES)
        elif name == "skills":
            valid = isinstance(value, list) and all(
                isinstance(v, dict)
                and set(v) == {"name", "description"}
                and all(isinstance(s, str) and s.strip() for s in v.values())
                for v in value
            )
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise AppError(
                f"Invalid or missing prompt variable {name!r}: check its type and allowed values."
            )


def render(name, **variables):
    """Return text, a compatible prompt ID, and the source/configuration identity.

    Unchanged text retains its legacy text digest. Changed text uses template_id,
    which hashes name, source and resolved variables.
    """
    if name not in TEMPLATES:
        raise AppError(f"Unknown prompt {name!r}. Choose: {', '.join(TEMPLATES)}.")
    unknown = variables.keys() - DEFAULTS[name].keys()
    if unknown:
        raise AppError(f"Unknown variables for {name}: {', '.join(sorted(unknown))}")
    values = {**deepcopy(DEFAULTS[name]), **variables}
    _validate(values)
    try:
        path = TEMPLATES[name]
        source = ENV.loader.get_source(ENV, path)[0]
        template = ENV.get_template(path)
        text = template.render(**values)
        default_text = template.render(**DEFAULTS[name])
        sources = {path: source}
        template_id = digest({"name": name, "source": sources, "variables": values})
        return RenderedPrompt(
            text, digest(text) if text == default_text else template_id, template_id
        )
    except TemplateError as error:
        raise AppError(f"Cannot render prompt {name!r}: {error}") from error


def for_settings(name, settings, **derived):
    validate_options(settings)
    values = {key: value for key, value in settings.extras.items() if key in DEFAULTS[name]}
    return render(name, **{**values, **derived})


def bundle_identity(prompts):
    """Preserve the existing combined text digest when all prompts are unchanged."""
    unchanged = all(prompt.prompt_id == digest(prompt.text) for prompt in prompts.values())
    return {
        "prompt_id": digest(
            {name: p.text if unchanged else p.prompt_id for name, p in prompts.items()}
        ),
        "prompt_template_ids": {name: p.template_id for name, p in prompts.items()},
    }
