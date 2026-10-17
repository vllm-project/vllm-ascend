"""Logging capture scoped to AscendStore v1 tests."""

from collections.abc import Iterator

import pytest
from vllm.logger import logger


@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.LogCaptureFixture]:
    """Capture v1 logs even when vLLM does not propagate them to the root logger."""
    # Capture at the emitting logger; disable propagation to avoid duplicate records.
    monkeypatch.setattr(logger, "propagate", False)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
