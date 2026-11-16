import os

import pytest

os.environ["VLLM_BATCH_INVARIANT"] = "1"


@pytest.fixture(autouse=True)
def enable_batch_invariant_mode(monkeypatch: pytest.MonkeyPatch):
    """Run every case under this directory in batch-invariant mode."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
