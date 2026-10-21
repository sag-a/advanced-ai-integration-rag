"""Saved public research activity; no hidden reasoning or scratch files."""

import streamlit as st
from rag_core.lesson_log import plain
from rag_core.redact import scrub


def show_research(details):
    with st.expander("Agentic research"):
        st.caption(
            "Deep Agents selects original PDF evidence using skills and tools. The separate answer step validates source references."
        )
        st.write(
            f"Searches: {details['searches']} · clause reads: {details['clause_reads']} · research model calls: {details['model_calls']} · delegations: {details['delegations']}"
        )
        st.write(
            "Skills read: "
            + (
                ", ".join(dict.fromkeys(path.split("/")[-2] for path in details["skills_read"]))
                or "none"
            )
        )
        rows = []
        for number, event in enumerate(details["events"], 1):
            fields = []
            for key, value in event.items():
                if key in {"event", "actor"}:
                    continue
                shown = f"{value:.2f}s" if key == "elapsed_seconds" else str(value)
                fields.append(f"{key.replace('_', ' ')}: {shown}")
            rows.append(
                {
                    "Step": number,
                    "Worker": event.get("actor", "coordinator"),
                    "Action": event["event"].replace("_", " "),
                    "Details": " · ".join(fields),
                }
            )
        st.dataframe(rows, hide_index=True)
        st.caption(
            f"Budget: {details['max_searches']} searches, {details['max_model_calls']} research model calls, {details['max_delegations']} sequential delegations. Research usage is separate from final generation usage."
        )
        st.caption(
            "Skills and scratch files are virtual and isolated per turn. The timeline persists with this accepted answer."
        )


def render_panels(log, event, data):
    data = scrub(data)
    fields = " · ".join(
        f"{key}={value}" for key, value in data.items() if key not in {"event", "request_id"}
    )
    log.panel(
        event.replace("_", " ").title(),
        plain(fields or "Continue within shared budget.", 500),
        request_id=data["request_id"],
    )
