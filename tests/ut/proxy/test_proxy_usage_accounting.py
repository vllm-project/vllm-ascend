"""Unit tests for the PD proxy's prefiller cached-token accounting.

The HTTP-level tests in ``test_load_balance_proxy_server.py`` exercise the
proxy end to end. These tests import the example module directly to pin the
fallback and the one-shot diagnostic that a missing
``--enable-prompt-tokens-details`` flag triggers.
"""

import importlib.util
import logging
from pathlib import Path

import pytest

PROXY_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "disaggregated_prefill_v1"
    / "load_balance_proxy_server_example.py"
)


@pytest.fixture(scope="module")
def proxy_module():
    spec = importlib.util.spec_from_file_location("load_balance_proxy_server_example", PROXY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_cached_tokens_uses_reported_value(proxy_module):
    """A reported prefiller count wins over any decoder value."""
    response = {"usage": {"prompt_tokens_details": {"cached_tokens": 128}}}
    assert proxy_module.extract_cached_tokens(response) == 128


def test_extract_cached_tokens_falls_back_to_zero(proxy_module):
    """No prompt-token details means the cold-cache value, not None."""
    proxy_module._missing_prompt_tokens_details_warned = False
    assert proxy_module.extract_cached_tokens({"usage": {}}) == 0
    assert proxy_module.extract_cached_tokens({}) == 0


def test_extract_cached_tokens_warns_once_when_details_absent(proxy_module, caplog):
    """The missing-flag diagnostic is emitted once per proxy process."""
    proxy_module._missing_prompt_tokens_details_warned = False
    with caplog.at_level(logging.WARNING, logger=proxy_module.logger.name):
        proxy_module.extract_cached_tokens({"usage": {}})
        proxy_module.extract_cached_tokens({"usage": {}})
    warnings = [record for record in caplog.records if "enable-prompt-tokens-details" in record.getMessage()]
    assert len(warnings) == 1