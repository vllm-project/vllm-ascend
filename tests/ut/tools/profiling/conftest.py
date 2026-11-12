"""Keep profiling tests independent of the runner's environment."""

import os

import pytest


@pytest.fixture(autouse=True)
def profile_environment(monkeypatch):
    for name in os.environ:
        if name.startswith("ASCEND_PROFILE_"):
            monkeypatch.delenv(name)
