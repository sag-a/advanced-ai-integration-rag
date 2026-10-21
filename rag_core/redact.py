"""Remove credentials and internal graph state before values reach a screen or log."""

import re

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from .graph import history_view


def scrub(value, secrets=()):
    """Keep prompt-sized history; remove internal state and known credentials."""
    if isinstance(value, BaseMessage):
        return {"role": value.type, "text": scrub(value.content, secrets)}
    if isinstance(value, Document):
        return {
            "id": value.id,
            "text": scrub(value.page_content, secrets),
            **scrub(value.metadata, secrets),
        }
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            label = str(key).lower()
            if re.search(
                r"api.?key|secret|password|authorization|cookie|token$", label
            ) and not label.endswith("_tokens"):
                clean[key] = "[REDACTED]"
            elif label.startswith(("checkpoint", "__pregel")) or label in {
                "configurable",
                "callbacks",
            }:
                continue
            elif (
                key == "messages"
                and isinstance(item, list)
                and all(isinstance(m, BaseMessage) for m in item)
            ):
                clean[key] = scrub(history_view(item), secrets)
            else:
                clean[key] = scrub(item, secrets)
        return clean
    if isinstance(value, (list, tuple)):
        return [scrub(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"\b(?:sk-|pk-lf-)[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value
