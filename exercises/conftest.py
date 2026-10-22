"""Grade your starter by default; LAB_TARGET=solution grades the reference solution."""

import importlib
import os

import pytest

TARGET = os.environ.get("LAB_TARGET", "starter")


@pytest.fixture
def lab(request):
    """The module under test: `starter` or `solution` next to the test file."""
    package = request.module.__name__.rsplit(".", 1)[0]
    return importlib.import_module(f"{package}.{TARGET}")
