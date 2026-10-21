"""Discover retriever plugins without importing their implementation into the app."""

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
import pkgutil
from typing import Callable

from .config import AppError


@dataclass(frozen=True)
class Architecture:
    name: str
    label: str
    make_retriever: Callable
    prepare: Callable | None = None
    check: Callable | None = None
    render_details: Callable | None = None
    log_panels: Callable | None = None
    entry_node: Callable | None = None
    needs_llm_for_retrieval: bool = False


REGISTRY: dict[str, Architecture] = {}
UNAVAILABLE: dict[str, str] = {}


class MissingExtra(ImportError):
    pass


def require_extra(extra, *modules):
    for name in modules:
        if find_spec(name) is None:
            raise MissingExtra(f"Install the {extra!r} extra: uv sync --extra {extra}")


def register(architecture: Architecture):
    if architecture.name in REGISTRY:
        raise ValueError(f"{architecture.name} already registered")
    REGISTRY[architecture.name] = architecture


def load_all():
    package = import_module("architectures")
    for info in pkgutil.iter_modules(package.__path__):
        if not info.ispkg:
            continue
        try:
            module = import_module(f"{package.__name__}.{info.name}")
        except MissingExtra as error:
            UNAVAILABLE[info.name] = str(error)
            REGISTRY.pop(info.name, None)
            continue
        architecture = getattr(module, "ARCHITECTURE", None)
        if architecture and architecture.name not in REGISTRY:
            register(architecture)
        UNAVAILABLE.pop(info.name, None)


def get(name):
    load_all()
    if name in UNAVAILABLE:
        raise AppError(UNAVAILABLE[name])
    if name not in REGISTRY:
        raise AppError(f"Unknown architecture {name!r}. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name]
