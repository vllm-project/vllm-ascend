# SPDX-License-Identifier: Apache-2.0
"""Unit tests for host normalization and backend identity in the proxy example."""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLE_PATH = (
    Path(__file__).parents[3] / "examples" / "disaggregated_prefill_v1" / "load_balance_proxy_server_example.py"
)
SPEC = importlib.util.spec_from_file_location("load_balance_proxy_server_example_under_test", EXAMPLE_PATH)
assert SPEC is not None and SPEC.loader is not None
proxy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = proxy
SPEC.loader.exec_module(proxy)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", "0.0.0.0"),
        ("127.0.0.1", "0.0.0.0"),
        ("0.0.0.0", "0.0.0.0"),
        ("localhost.example.com", "localhost.example.com"),
        ("127.0.0.1.example.com", "127.0.0.1.example.com"),
        ("example.com", "example.com"),
        ("192.168.1.10", "192.168.1.10"),
        ("::1", "::1"),
    ],
)
def test_normalize_host_only_rewrites_exact_loopback_spellings(host, expected):
    assert proxy.normalize_host(host) == expected


def test_loopback_aliases_share_one_server_key():
    keys = {proxy.server_key(host, 9001) for host in ("localhost", "127.0.0.1", "0.0.0.0")}
    assert keys == {"0.0.0.0:9001"}


def test_hosts_containing_loopback_substrings_keep_distinct_keys():
    entries = [
        ("localhost.example.com", 9001),
        ("127.0.0.1.example.com", 9001),
        ("0.0.0.0.example.com", 9001),
    ]
    keys = {proxy.server_key(host, port) for host, port in entries}
    assert len(keys) == 3


def test_same_host_with_different_ports_keeps_distinct_keys():
    keys = {proxy.server_key("localhost", port) for port in (9001, 9002)}
    assert keys == {"0.0.0.0:9001", "0.0.0.0:9002"}
