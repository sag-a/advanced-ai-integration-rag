"""Model-free classroom preview: trusted configuration becomes prompt text."""

from dataclasses import replace
from difflib import SequenceMatcher
from itertools import zip_longest

from rich.table import Table
from rich.text import Text

from rag_core.templating import for_settings


def preview(name, settings):
    if name in {"research", "helper"}:
        from architectures.agentic_rag.prompts import research_prompts

        return research_prompts(settings, runtime=True)[name]
    return for_settings(name, settings)


def show_prompt(name, settings, console):
    default = preview(name, replace(settings, extras={}, top_k=6))
    customized = preview(name, settings)
    console.print(
        Text(
            f"Prompt: {name}\nprompt_id: {customized.prompt_id}\ntemplate_id: {customized.template_id}"
        )
    )
    console.print(Text(customized.text))
    table = Table(title="Default / Customized", expand=True)
    table.add_column("Default", ratio=1)
    table.add_column("Customized", ratio=1)
    left, right = default.text.splitlines(), customized.text.splitlines()
    for operation, a, b, c, d in SequenceMatcher(a=left, b=right).get_opcodes():
        for old, new in zip_longest(left[a:b], right[c:d], fillvalue=""):
            table.add_row(
                Text(old, style="red" if operation != "equal" else ""),
                Text(new, style="green" if operation != "equal" else ""),
            )
    console.print(table)
