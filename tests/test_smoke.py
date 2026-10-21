import importlib
from pathlib import Path

import pytest
from dotenv import dotenv_values


@pytest.mark.parametrize(
    "package",
    [
        "rag_core",
        "architectures.classic_rag",
        "architectures.graph_rag",
        "architectures.agentic_rag",
        "architectures.system1_rag",
        "app",
        "evals",
        "modules.c01_embeddings.local_embeddings",
        "modules.c02_vector_databases.nomic_embedding_function",
    ],
)
def test_packages_import(package):
    importlib.import_module(package)


def test_example_contains_only_empty_credentials():
    values = dotenv_values(Path(__file__).parents[1] / ".env.example")
    assert set(values) == {"ANTHROPIC_API_KEY"}
    assert all(value == "" for value in values.values())
