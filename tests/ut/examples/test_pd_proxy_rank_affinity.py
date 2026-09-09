# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def proxy(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3] / "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
    )
    spec = importlib.util.spec_from_file_location("pd_proxy_rank_affinity_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    policy = asyncio.get_event_loop_policy()
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        asyncio.set_event_loop_policy(policy)


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"remote_engine_id": "engine_dp0"}, 0),
        ({"remote_engine_id": "engine_dp12"}, 12),
        ({"remote_engine_id": "engine_dp1_tail"}, None),
        ({"remote_engine_id": "engine_dp-1"}, None),
        ({"remote_engine_id": 1}, None),
        ({}, None),
        (None, None),
        ("invalid", None),
    ],
)
def test_decoder_rank_from_metadata(proxy, params, expected):
    assert proxy.decoder_data_parallel_rank({"kv_transfer_params": params}) == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_rank_affinity_is_opt_in(proxy, monkeypatch, enabled):
    monkeypatch.setattr(sys, "argv", ["proxy"] + (["--decode-rank-affinity"] if enabled else []))
    assert proxy.parse_args().decode_rank_affinity is enabled


@pytest.mark.parametrize(
    "enabled,engine_id,expected", [(False, "engine_dp2", None), (True, "engine_dp2", "2"), (True, "engine", None)]
)
def test_decode_stream_forwards_rank_header(proxy, monkeypatch, enabled, engine_id, expected):
    monkeypatch.setattr(proxy, "get_global_args", lambda: SimpleNamespace(decode_rank_affinity=enabled))
    captured = {}

    class Stream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def raise_for_status(self):
            pass

        async def aiter_bytes(self):
            yield b"data: test\n\n"

    class Client:
        def stream(self, method, endpoint, **kwargs):
            captured.update(kwargs["headers"])
            return Stream()

    async def consume():
        return [
            chunk
            async for chunk in proxy.stream_service_response(
                Client(), "/chat/completions", {"kv_transfer_params": {"remote_engine_id": engine_id}}, "req-test"
            )
        ]

    assert asyncio.run(consume()) == [b"data: test\n\n"]
    assert captured.get("X-data-parallel-rank") == expected
