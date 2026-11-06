# SPDX-License-Identifier: Apache-2.0
"""Conversation routing and accounting without model or accelerator imports."""

import asyncio
import copy
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy


def make_scheduler(
    endpoints=None,
    *,
    policy="rendezvous",
    max_groups=proxy.DEFAULT_PREFILL_AFFINITY_MAX_GROUPS,
):
    return proxy.SharedProxyScheduler(
        endpoints or [("10.0.0.1", port) for port in range(8000, 8004)],
        [("10.0.0.2", 9000), ("10.0.0.2", 9001)],
        prefill_affinity_policy=policy,
        prefill_affinity_max_groups=max_groups,
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
    args = proxy.parse_args()
    assert not args.enable_prefill_affinity
    assert args.prefill_affinity_policy == "rendezvous"
    assert args.prefill_affinity_prefix_chars == 0
    assert args.prefill_affinity_max_groups == proxy.DEFAULT_PREFILL_AFFINITY_MAX_GROUPS

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "proxy",
            "--enable-prefill-affinity",
            "--prefill-affinity-policy",
            "group-balanced",
            "--prefill-affinity-prefix-chars",
            "256",
            "--prefill-affinity-max-groups",
            "42",
        ],
    )
    args = proxy.parse_args()
    assert args.enable_prefill_affinity
    assert args.prefill_affinity_policy == "group-balanced"
    assert args.prefill_affinity_prefix_chars == 256
    assert args.prefill_affinity_max_groups == 42


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--prefill-affinity-policy", "group-balanced"], "requires prefill affinity"),
        (["--prefill-affinity-prefix-chars", "1"], "requires prefill affinity"),
        (["--enable-prefill-affinity", "--prefill-affinity-prefix-chars", "-1"], "non-negative"),
        (["--prefill-affinity-max-groups", "0"], "positive"),
    ],
)
def test_cli_rejects_invalid_affinity_configuration(monkeypatch, argv, message):
    monkeypatch.setattr(sys, "argv", ["proxy", *argv])
    with pytest.raises(ValueError, match=message):
        proxy.parse_args()


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
    monkeypatch.setattr(
        proxy,
        "get_global_args",
        lambda: SimpleNamespace(
            enable_prefill_affinity=True,
            prefill_affinity_prefix_chars=2,
        ),
    )
    request = chat()
    with pytest.raises(RuntimeError, match="captured"):
        asyncio.run(proxy.assign_instances("/v1/chat/completions", request, 100, is_initial_request=initial))
    runtime.schedule.assert_awaited_once_with(
        "begin_request" if initial else "reserve_prefill_kv",
        proxy.calculate_prefill_score(100),
        proxy.conversation_affinity_key(request, 2),
    )
    assert request == chat()


def test_string_prefix_groups_suffixes_without_mutating_request():
    request = chat()
    request["messages"][1]["content"] = "shared-prefix:first suffix"
    other = copy.deepcopy(request)
    other["messages"][1]["content"] = "shared-prefix:second suffix"
    original = copy.deepcopy(request)

    assert proxy.conversation_affinity_key(request, 13) == proxy.conversation_affinity_key(other, 13)
    assert proxy.conversation_affinity_key(request) != proxy.conversation_affinity_key(other)
    assert request == original


@pytest.mark.parametrize("prefix_chars", [1, 2, 3])
def test_prefix_length_counts_unicode_characters(prefix_chars):
    request = chat()
    request["messages"][1]["content"] = "你好🙂shared suffix"
    expected = copy.deepcopy(request)
    expected["messages"][1]["content"] = "你好🙂"[:prefix_chars] + "different suffix"
    assert proxy.conversation_affinity_key(request, prefix_chars) == proxy.conversation_affinity_key(
        expected, prefix_chars
    )


def test_prefix_longer_than_content_preserves_full_content():
    request = chat()
    request["messages"][1]["content"] = "你好🙂shared suffix"
    other = copy.deepcopy(request)
    other["messages"][1]["content"] = "你好🙂different suffix"
    assert proxy.conversation_affinity_key(request, 100) == proxy.conversation_affinity_key(request)
    assert proxy.conversation_affinity_key(request, 100) != proxy.conversation_affinity_key(other, 100)


def test_structured_prefix_truncates_only_first_text_part():
    request = chat()
    request["messages"][1]["content"] = [
        {"type": "image_url", "image_url": {"url": "https://example.invalid/one.png"}},
        {"type": "text", "text": "shared text:first suffix"},
        {"type": "text", "text": "later text remains significant"},
    ]
    other = copy.deepcopy(request)
    other["messages"][1]["content"][1]["text"] = "shared text:second suffix"
    original = copy.deepcopy(request)

    assert proxy.conversation_affinity_key(request, 11) == proxy.conversation_affinity_key(other, 11)
    other["messages"][1]["content"][0]["image_url"]["url"] = "https://example.invalid/two.png"
    assert proxy.conversation_affinity_key(request, 11) != proxy.conversation_affinity_key(other, 11)
    other = copy.deepcopy(request)
    other["messages"][1]["content"][2]["text"] = "changed later text"
    assert proxy.conversation_affinity_key(request, 11) != proxy.conversation_affinity_key(other, 11)
    assert request == original


def test_prefix_does_not_override_explicit_user_identity():
    request = {**chat(), "user": "stable-task-id"}
    changed = copy.deepcopy(request)
    changed["messages"][1]["content"] = "an unrelated prompt"
    assert proxy.conversation_affinity_key(request, 1) == proxy.conversation_affinity_key(changed, 1)
    changed["user"] = "different-task-id"
    assert proxy.conversation_affinity_key(request, 1) != proxy.conversation_affinity_key(changed, 1)


@pytest.mark.parametrize("policy", ["unsupported", ""])
def test_scheduler_rejects_unknown_policy(policy):
    with pytest.raises(ValueError, match="Unsupported prefill affinity policy"):
        make_scheduler(policy=policy)


def test_default_rendezvous_remains_stateless_and_ignores_group_capacity():
    scheduler = make_scheduler(max_groups=1)
    routes = [scheduler.reserve_prefill_kv(0, str(i))["key"] for i in range(20)]
    assert len(set(routes)) > 1
    assert not scheduler._prefill_affinity_backend
    assert not scheduler._prefill_affinity_groups


def test_group_balanced_assignments_are_sticky_and_balanced():
    scheduler = make_scheduler(policy="group-balanced")
    routes = {}
    for i in range(100):
        key = str(i)
        first = scheduler.reserve_prefill_kv(0, key)["key"]
        scheduler.prefillers[first].active_kv_cache = 10_000
        assert scheduler.reserve_prefill_kv(0, key)["key"] == first
        routes[key] = first

    counts = {backend: list(routes.values()).count(backend) for backend in scheduler.prefillers}
    assert set(counts.values()) == {25}
    assert len(scheduler._prefill_affinity_backend) == 100
    assert sum(len(groups) for groups in scheduler._prefill_affinity_groups.values()) == 100


def test_group_balanced_ties_are_deterministic_across_endpoint_order():
    endpoints = [("10.0.0.1", 8000), ("10.0.0.1", 8001)]
    first = make_scheduler(endpoints, policy="group-balanced")
    second = make_scheduler(list(reversed(endpoints)), policy="group-balanced")
    assert [first.reserve_prefill_kv(0, str(i))["key"] for i in range(20)] == [
        second.reserve_prefill_kv(0, str(i))["key"] for i in range(20)
    ]


def test_group_balanced_addition_does_not_migrate_healthy_groups():
    scheduler = make_scheduler([("10.0.0.1", 8000), ("10.0.0.1", 8001)], policy="group-balanced")
    before = {str(i): scheduler.reserve_prefill_kv(0, str(i))["key"] for i in range(20)}
    scheduler.add_instances(proxy.ServerRole.PREFILL, [("10.0.0.1", 8002)])
    scheduler.activate_waiting_instance(proxy.ServerRole.PREFILL, "10.0.0.1", 8002)

    assert {key: scheduler.reserve_prefill_kv(0, key)["key"] for key in before} == before
    assert scheduler.reserve_prefill_kv(0, "new-group")["key"] == "10.0.0.1:8002"


def test_group_balanced_taint_only_remaps_affected_groups():
    scheduler = make_scheduler(policy="group-balanced")
    before = {str(i): scheduler.begin_request(1, str(i))["key"] for i in range(20)}
    removed = before["0"]
    unaffected = next(key for key, backend in before.items() if backend != removed)
    host, port = removed.rsplit(":", 1)
    scheduler.remove_instances(proxy.ServerRole.PREFILL, [(host, int(port))])

    assert scheduler.begin_request(1, unaffected)["key"] == before[unaffected]
    assert scheduler.begin_request(1, "0")["key"] != removed
    assert "0" not in scheduler._prefill_affinity_backend or (scheduler._prefill_affinity_backend["0"] != removed)


def test_group_balanced_finalize_and_readd_preserves_unaffected_groups():
    scheduler = make_scheduler(
        [("10.0.0.1", 8000), ("10.0.0.1", 8001)],
        policy="group-balanced",
    )
    before = {str(i): scheduler.begin_request(1, str(i))["key"] for i in range(20)}
    removed = before["0"]
    unaffected = next(key for key, backend in before.items() if backend != removed)
    host, port = removed.rsplit(":", 1)
    scheduler.remove_instances(proxy.ServerRole.PREFILL, [(host, int(port))])

    for backend in before.values():
        scheduler.finish_request(backend, 1, None, 0, True)
    scheduler.finalize_tainted_instances()

    assert removed not in scheduler.prefillers
    assert all(backend != removed for backend in scheduler._prefill_affinity_backend.values())
    assert scheduler.reserve_prefill_kv(0, unaffected)["key"] == before[unaffected]

    scheduler.add_instances(proxy.ServerRole.PREFILL, [(host, int(port))])
    scheduler.activate_waiting_instance(proxy.ServerRole.PREFILL, host, int(port))
    assert scheduler.reserve_prefill_kv(0, "new-after-readd")["key"] == removed


def test_group_balanced_no_healthy_prefiller_does_not_create_state():
    scheduler = make_scheduler([("10.0.0.1", 8000)], policy="group-balanced")
    scheduler.begin_request(1, "existing")
    scheduler.remove_instances(proxy.ServerRole.PREFILL, [("10.0.0.1", 8000)])
    with pytest.raises(RuntimeError, match="No available prefill"):
        scheduler.begin_request(1, "new")
    assert "new" not in scheduler._prefill_affinity_backend


def test_group_balanced_lru_eviction_updates_both_indexes():
    scheduler = make_scheduler(policy="group-balanced", max_groups=2)
    scheduler.reserve_prefill_kv(0, "first")
    scheduler.reserve_prefill_kv(0, "second")
    first_backend = scheduler.reserve_prefill_kv(0, "first")["key"]
    scheduler.reserve_prefill_kv(0, "third")

    assert list(scheduler._prefill_affinity_backend) == ["first", "third"]
    assert scheduler._prefill_affinity_backend["first"] == first_backend
    assert all("second" not in groups for groups in scheduler._prefill_affinity_groups.values())
    assert sum(len(groups) for groups in scheduler._prefill_affinity_groups.values()) == 2
    assert scheduler.reserve_prefill_kv(0, "first")["key"] == first_backend
    assert len(scheduler._prefill_affinity_backend) == 2


def test_two_manager_rpc_clients_share_group_balanced_state(monkeypatch):
    scheduler = make_scheduler([("10.0.0.1", 8000), ("10.0.0.1", 8001)], policy="group-balanced")
    monkeypatch.setattr(proxy, "shared_scheduler", scheduler)
    manager = proxy.SchedulerManager(address=("127.0.0.1", 0), authkey=b"unit-test-authkey")
    server = manager.get_server()

    def serve_until_stopped():
        try:
            server.serve_forever()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise

    thread = threading.Thread(target=serve_until_stopped, daemon=True)
    thread.start()
    clients = []
    try:
        for _ in range(2):
            client = proxy.SchedulerManager(address=server.address, authkey=b"unit-test-authkey")
            client.connect()
            clients.append(client.get_scheduler())
        first = clients[0].begin_request(1, "rpc-group")["key"]
        assert clients[1].begin_request(1, "rpc-group")["key"] == first
        assert clients[1].begin_request(1, "other-group")["key"] != first
        assert clients[0].healthcheck()["request_num"] == 3
        assert clients[1].healthcheck()["request_num"] == 3
    finally:
        server.stop_event.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
