# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.ut.model_loader.rfork.test_lease_release import make_session, run_and_join
from tests.ut.model_loader.rfork.test_lease_release import runtime as runtime


@pytest.fixture
def ops_runtime(request):
    return request.getfixturevalue("runtime")


def test_ops_config_defaults_and_no_environment_fallback(ops_runtime, monkeypatch):
    for name in (
        "RFORK_HEARTBEAT_INTERVAL_SEC",
        "RFORK_LEASE_RELEASE_MAX_ATTEMPTS",
        "RFORK_LEASE_RELEASE_RETRY_INTERVAL_SEC",
    ):
        monkeypatch.setenv(name, "99")
    config = type(ops_runtime.config).from_extra_config({})
    assert config.heartbeat_interval_sec == 30
    assert config.lease_release_max_attempts == 3
    assert config.lease_release_retry_interval_sec == 30


def test_ops_config_explicit_json_values(ops_runtime):
    config = type(ops_runtime.config).from_extra_config(
        {
            "rfork_heartbeat_interval_sec": 20,
            "rfork_lease_release_max_attempts": 5,
            "rfork_lease_release_retry_interval_sec": 2.5,
            "rfork_request_timeout_sec": 8,
            "enable_multithread_load": True,
        }
    )
    assert config.heartbeat_interval_sec == 20
    assert config.lease_release_max_attempts == 5
    assert config.lease_release_retry_interval_sec == 2.5
    assert config.request_timeout_sec == 8


@pytest.mark.parametrize("name", ["heartbeat_interval_sec", "lease_release_retry_interval_sec"])
@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"), None, "20", [], {}])
def test_ops_intervals_reject_invalid_values(ops_runtime, name, value):
    with pytest.raises(ValueError, match=f"rfork_{name}"):
        type(ops_runtime.config).from_extra_config({f"rfork_{name}": value})
    with pytest.raises(ValueError, match=f"rfork_{name}"):
        replace(ops_runtime.config, **{name: value})


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, 3.0, float("nan"), float("inf"), None, "3", []])
def test_ops_attempts_require_positive_json_integer(ops_runtime, value):
    with pytest.raises(ValueError, match="rfork_lease_release_max_attempts"):
        type(ops_runtime.config).from_extra_config({"rfork_lease_release_max_attempts": value})


@pytest.mark.parametrize(
    ("attempts", "result", "expected"), [(1, "RETRYABLE", 1), (4, "RETRYABLE", 4), (4, "REJECTED", 1)]
)
def test_async_release_uses_configured_budget_and_interval(ops_runtime, attempts, result, expected):
    ops_runtime.config = replace(
        ops_runtime.config, lease_release_max_attempts=attempts, lease_release_retry_interval_sec=7
    )
    session = make_session(ops_runtime)
    event = Mock(is_set=Mock(return_value=False), wait=Mock(return_value=False))
    session.lease_release_stop_event = event
    session.planner.release_seed_once.return_value = getattr(ops_runtime.types.LeaseReleaseResult, result)
    run_and_join(session)
    assert session.planner.release_seed_once.call_count == expected
    assert event.wait.call_count == expected - 1
    for call in event.wait.call_args_list:
        assert call.args == (7,)
    assert session.seed_lease is ops_runtime.lease
    assert session._lease_release_exhausted


def test_release_stop_interrupts_configured_long_wait(ops_runtime):
    ops_runtime.config = replace(ops_runtime.config, lease_release_retry_interval_sec=3600)
    session = make_session(ops_runtime)
    entered = threading.Event()
    original_wait = session.lease_release_stop_event.wait

    def wait(timeout):
        entered.set()
        return original_wait(timeout)

    session.lease_release_stop_event.wait = wait
    session.planner.release_seed_once.return_value = ops_runtime.types.LeaseReleaseResult.RETRYABLE
    with session._lock:
        session.release_seed()
        worker = session.lease_release_thread
    try:
        assert entered.wait(1)
    finally:
        session.lease_release_stop_event.set()
        worker.join(2)
    assert not worker.is_alive()
    session.planner.release_seed_once.assert_called_once()


def test_sync_release_uses_same_config_but_seed_removal_keeps_internal_policy(ops_runtime, monkeypatch):
    config = replace(ops_runtime.config, lease_release_max_attempts=2, lease_release_retry_interval_sec=7)
    client = ops_runtime.client.RForkPlannerClient(config, ops_runtime.identity)
    post = Mock(return_value=SimpleNamespace(status_code=503, text="busy"))
    sleep = Mock()
    monkeypatch.setattr(ops_runtime.client.requests, "post", post)
    monkeypatch.setattr(ops_runtime.client.time, "sleep", sleep)
    assert not client.release_seed(ops_runtime.lease)
    assert post.call_count == 2
    sleep.assert_called_once_with(7)
    post.reset_mock()
    sleep.reset_mock()
    advertisement = ops_runtime.types.SeedAdvertisement("127.0.0.1", 1234, 0)
    assert not client.remove_seed(advertisement)
    assert post.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.1, 0.2]


def test_session_passes_heartbeat_config_to_thread(ops_runtime, monkeypatch):
    config = replace(ops_runtime.config, heartbeat_interval_sec=17)
    session = ops_runtime.session.RForkSession(config, ops_runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.transfer_backend = Mock()
    session.state = ops_runtime.types.RForkLifecycleState.REGISTERED
    monkeypatch.setattr(session, "_seed_transfer_info", Mock())
    monkeypatch.setattr(ops_runtime.session, "start_rfork_server", Mock(return_value=SimpleNamespace(port=1234)))
    thread = Mock()
    monkeypatch.setattr(ops_runtime.session.threading, "Thread", thread)
    assert session.start_seed_service(object(), True)
    kwargs = thread.call_args.kwargs["kwargs"]
    assert kwargs["sleep_interval"] == 17
    assert kwargs["initial_delay"] is True
    assert kwargs["stop_event"] is session.heartbeat_stop_event
    thread.return_value.start.assert_called_once()


def test_heartbeat_uses_configured_interval_for_initial_and_periodic_wait(ops_runtime, monkeypatch):
    client = ops_runtime.client.RForkPlannerClient(
        replace(ops_runtime.config, heartbeat_interval_sec=17), ops_runtime.identity
    )
    events = []
    stop = Mock(is_set=Mock(side_effect=[False, True]))
    stop.wait.side_effect = lambda seconds: events.append(("wait", seconds)) or False
    monkeypatch.setattr(client, "report_seed_once", lambda *args, **kwargs: events.append(("report", 1234)) or True)
    client.run_seed_heartbeat(1234, stop_event=stop, initial_delay=True)
    assert events == [("wait", 17), ("report", 1234), ("wait", 17)]


def test_heartbeat_stop_interrupts_initial_delay(ops_runtime, monkeypatch):
    client = ops_runtime.client.RForkPlannerClient(ops_runtime.config, ops_runtime.identity)
    report = Mock()
    monkeypatch.setattr(client, "report_seed_once", report)
    stop = threading.Event()
    stop.set()
    client.run_seed_heartbeat(1234, sleep_interval=3600, stop_event=stop, initial_delay=True)
    report.assert_not_called()
