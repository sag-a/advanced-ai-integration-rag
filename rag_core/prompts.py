"""Packaged Jinja prompts; runtime data belongs in the separate JSON payload."""

from .templating import for_settings as configured, render

REWRITE = render("rewrite").text
ANSWER = render("answer").text


def for_settings(settings):
    return {name: configured(name, settings) for name in ("rewrite", "answer")}
