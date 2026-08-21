"""Make legacy path-based smoke tests independent of the pytest launch cwd."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _legacy_repository_working_directory():
    previous = Path.cwd()
    repository = Path(__file__).resolve().parents[2]
    os.chdir(repository)
    try:
        yield
    finally:
        os.chdir(previous)
