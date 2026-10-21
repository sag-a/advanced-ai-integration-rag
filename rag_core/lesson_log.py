"""One terminal presentation style; plugins own architecture-specific panels."""

import re

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .redact import scrub


def plain(value, limit=160):
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(scrub(str(value))))
    return Text(text[:limit] + ("…" if len(text) > limit else ""))


def passage_table(documents, origins=None):
    table = Table(expand=True, header_style="bold cyan", box=None)
    for column in ("#", "Evidence", "Source", "PDF page", "Clause", "Distance"):
        table.add_column(column)
    for number, document in enumerate(documents, 1):
        tags = document.metadata
        distance = tags.get("distance")
        table.add_row(
            str(number),
            plain((origins or {}).get(document.id, "Retrieved")),
            plain(tags["source"], 50),
            f"{tags['page_start']}–{tags['page_end']}",
            plain(tags.get("clause", "—")),
            f"{distance:.4f}" if distance is not None else "—",
        )
    return table


class LessonLog:
    def __init__(self, renderer=None, console=None):
        self.renderer = renderer
        self.console = console or Console(stderr=True, highlight=False)

    def panel(self, title, content, *, style="cyan", request_id=""):
        subtitle = Text(f"retrieval {request_id}", style="dim") if request_id else None
        self.console.print(
            Panel(
                content,
                title=Text(title, style="bold " + style),
                subtitle=subtitle,
                border_style=style,
            )
        )

    def progress(self, message):
        self.console.print(Text("  PREPARE  ", style="bold magenta"), plain(message, 240))

    def selected(self, result):
        self.panel(
            "Retrieved passages",
            Group(
                Text(f"{len(result.documents)} passages · {result.elapsed_seconds:.2f}s"),
                passage_table(result.documents),
            ),
            request_id=result.details.get("retrieval_id", ""),
        )

    def accepted(self, state):
        result = state["messages"][-1].additional_kwargs["result"]
        label = "Saved result replayed" if state.get("replayed") else "Validated answer saved"
        self.panel(
            "Answer and memory",
            Text(
                f"{label} · {result['answer_status']} · {len(result.get('source_map', []))} cited passages"
            ),
            style="green",
        )

    def __call__(self, event, data):
        if self.renderer:
            self.renderer(self, event, data)
