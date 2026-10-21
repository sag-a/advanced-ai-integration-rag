"""Bind a registry entry to the shared collection and answer model."""

from dataclasses import dataclass

from rag_core.lesson_log import LessonLog
from rag_core.model import AnthropicChat
from rag_core.registry import Architecture, get
from rag_core.store import connect, open_collection


@dataclass
class Resources:
    architecture: Architecture
    chat: object
    retrieve: object
    entry: object
    log: LessonLog


def create_resources(settings):
    architecture = get(settings.architecture)
    collection = open_collection(connect(settings), settings)
    if architecture.check:
        architecture.check(settings, collection)
    log = LessonLog(architecture.log_panels)
    entry = (
        architecture.entry_node(settings, collection, log) if architecture.entry_node else None
    )
    return Resources(
        architecture,
        AnthropicChat(settings),
        architecture.make_retriever(settings, collection, log),
        entry,
        log,
    )
