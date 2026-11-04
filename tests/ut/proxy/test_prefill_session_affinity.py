# SPDX-License-Identifier: Apache-2.0
"""Conversation routing and accounting without model or accelerator imports."""

import asyncio
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


def make_scheduler(endpoints=None):
    return proxy.SharedProxyScheduler(
        endpoints or [("10.0.0.1", port) for port in range(8000, 8004)],
        [("10.0.0.2", 9000), ("10.0.0.2", 9001)],
    )


def chat():
    return {
        "model": "test-model",
        "messages": [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Fix this bug"}],
    }


def test_growing_history_and_sampling_do_not_change_key():
    request = chat()
    key = proxy.conversation_affinity_key(request)
    request["messages"] += [
        {"role": "assistant", "content": "Inspect the file"},
        {"role": "tool", "content": "file contents"},
        {"role": "user", "content": "Continue"},
    ]
    request.update(temperature=0.5, max_tokens=1024, request_id="another-request")
    assert proxy.conversation_affinity_key(request) == key


@pytest.mark.parametrize("field", ["model", "instructions", "first_user"])
def test_fallback_distinguishes_model_and_prefix(field):
    request = chat()
    changed = copy.deepcopy(request)
    if field == "model":
        changed["model"] = "other-model"
    else:
        changed["messages"][0 if field == "instructions" else 1]["content"] = "different"
    assert proxy.conversation_affinity_key(request) != proxy.conversation_affinity_key(changed)


def test_explicit_session_survives_history_compaction():
    original = {**chat(), "user": "task-1:agent-0"}
    compacted = {"model": "test-model", "user": "task-1:agent-0", "messages": []}
    assert proxy.conversation_affinity_key(original) == proxy.conversation_affinity_key(compacted)
    compacted["user"] = "task-2:agent-0"
    assert proxy.conversation_affinity_key(original) != proxy.conversation_affinity_key(compacted)


def test_structured_unicode_content_and_dictionary_order():
    request = {"model": "model", "messages": [{"role": "user", "content": [{"type": "text", "text": "你好"}]}]}
    reordered = {"messages": [{"content": [{"text": "你好", "type": "text"}], "role": "user"}], "model": "model"}
    assert proxy.conversation_affinity_key(request) == proxy.conversation_affinity_key(reordered)


@pytest.mark.parametrize("req_data", [{"prompt": "hello"}, {"messages": []}, {"messages": [{"role": "system"}]}])
def test_missing_session_and_user_message_has_no_affinity(req_data):
    assert proxy.conversation_affinity_key(req_data) is None


def test_cli_affinity_is_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["proxy"])
    assert not proxy.parse_args().enable_prefill_affinity
    monkeypatch.setattr(sys, "argv", ["proxy", "--enable-prefill-affinity"])
    assert proxy.parse_args().enable_prefill_affinity


def test_affinity_ignores_load_but_preserves_accounting_and_decode_balancing():
    scheduler = make_scheduler()
    first = scheduler.begin_request(10, "session")
    second = scheduler.begin_request(20, "session")
    recompute = scheduler.reserve_prefill_kv(5, "session")
    assert first["key"] == second["key"] == recompute["key"]
    assert scheduler.request_num == 2
    assert scheduler.prefillers[first["key"]].active_kv_cache == 35
    scheduler.release_prefill_kv(first["key"], 35)
    assert scheduler.prefillers[first["key"]].active_kv_cache == 0
    assert scheduler.pick_decoder(10)["key"] != scheduler.pick_decoder(10)["key"]


def test_default_scheduler_still_balances_load():
    scheduler = make_scheduler()
    assert scheduler.begin_request(10)["key"] != scheduler.begin_request(10)["key"]


def test_removal_only_remaps_affected_sessions():
    scheduler = make_scheduler()
    before = {str(i): scheduler.reserve_prefill_kv(0, str(i))["key"] for i in range(100)}
    assert len(set(before.values())) == 4
    removed = before["0"]
    host, port = removed.rsplit(":", 1)
    scheduler.remove_instances(proxy.ServerRole.PREFILL, [(host, int(port))])
    for session, previous in before.items():
        current = scheduler.reserve_prefill_kv(0, session)["key"]
        assert current != removed
        if previous != removed:
            assert current == previous


def test_tainted_endpoint_is_excluded_and_all_tainted_fails_without_leaking_load():
    scheduler = make_scheduler()
    first = scheduler.begin_request(10, "session")
    host, port = first["key"].rsplit(":", 1)
    scheduler.remove_instances(proxy.ServerRole.PREFILL, [(host, int(port))])
    assert scheduler.reserve_prefill_kv(5, "session")["key"] != first["key"]
    scheduler.remove_instances(proxy.ServerRole.PREFILL, [("10.0.0.1", p) for p in range(8000, 8004)])
    count = scheduler.request_num
    with pytest.raises(RuntimeError, match="No available prefill"):
        scheduler.begin_request(10, "session")
    assert scheduler.request_num == count


def test_endpoint_order_and_python_hash_seed_do_not_affect_routes():
    # Load the proxy module by file path: the child does not get pytest's
    # sys.path injection, and a regular `examples` package on PYTHONPATH can
    # otherwise shadow the repo's namespace package.
    example_path = str(Path(proxy.__file__).resolve())
    code = (
        "import importlib.util, json, sys; "
        f"spec = importlib.util.spec_from_file_location('lb_proxy', {example_path!r}); "
        "m = importlib.util.module_from_spec(spec); "
        "sys.modules['lb_proxy'] = m; spec.loader.exec_module(m); "
        "s=m.SharedProxyScheduler([('10.0.0.1',n) for n in PORTS], [('10.0.0.2',9000)]); "
        "print(json.dumps([s.reserve_prefill_kv(0,str(i))['key'] for i in range(20)]))"
    )
    routes = []
    for seed, ports in [("1", [8000, 8001, 8002, 8003]), ("2", [8003, 8002, 8001, 8000])]:
        output = subprocess.check_output(
            [sys.executable, "-c", code.replace("PORTS", repr(ports))],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        routes.append(json.loads(output))
    assert routes[0] == routes[1]


@pytest.mark.parametrize("enabled", [False, True])
def test_session_header_precedence_only_when_enabled(monkeypatch, enabled):
    received = []

    async def receive():
        return {"type": "http.request", "body": json.dumps({**chat(), "user": "body-session"}).encode()}

    async def capture_assignment(api, req_data, request_length, *, is_initial_request):
        received.append(req_data)
        raise RuntimeError("stop after inspecting routing input")

    monkeypatch.setattr(proxy, "get_runtime", lambda: None)
    monkeypatch.setattr(proxy, "get_global_args", lambda: SimpleNamespace(enable_prefill_affinity=enabled))
    monkeypatch.setattr(proxy, "assign_instances", capture_assignment)
    request = Request({"type": "http", "headers": [(b"x-session-id", b"header-session")]}, receive)
    with pytest.raises(RuntimeError, match="stop after inspecting"):
        asyncio.run(proxy.handle_completions_impl("/v1/chat/completions", request))
    assert received[0]["user"] == ("header-session" if enabled else "body-session")


@pytest.mark.parametrize("initial", [False, True])
def test_assignment_propagates_affinity_on_initial_and_recompute(monkeypatch, initial):
    runtime = SimpleNamespace(schedule=AsyncMock(side_effect=RuntimeError("captured")))
    monkeypatch.setattr(proxy, "get_runtime", lambda: runtime)
    monkeypatch.setattr(proxy, "get_global_args", lambda: SimpleNamespace(enable_prefill_affinity=True))
    request = chat()
    with pytest.raises(RuntimeError, match="captured"):
        asyncio.run(proxy.assign_instances("/v1/chat/completions", request, 100, is_initial_request=initial))
    runtime.schedule.assert_awaited_once_with(
        "begin_request" if initial else "reserve_prefill_kv",
        proxy.calculate_prefill_score(100),
        proxy.conversation_affinity_key(request),
    )
