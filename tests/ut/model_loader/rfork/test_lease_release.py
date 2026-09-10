# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import logging
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import requests


@pytest.fixture
def runtime(monkeypatch):
    """Load the real lease/session code with only external runtime dependencies stubbed.

    This fixture is isolated from other UT modules and needs no vLLM/NPU installation.
    Transport, model preparation and seed HTTP serving are injected independently.
    """
    prefix = "vllm_ascend.model_loader.rfork"
    root = Path(__file__).resolve().parents[4] / "vllm_ascend/model_loader/rfork"

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name):
        full_name = f"{prefix}.{name}"
        spec = importlib.util.spec_from_file_location(full_name, root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        return module

    stub("vllm.logger", logger=logging.getLogger("rfork-release-test"))
    stub("vllm.utils.network_utils", get_ip=lambda: "127.0.0.1", join_host_port=lambda host, port: f"{host}:{port}")
    stub(f"{prefix}.identity", build_seed_key=lambda **kwargs: "model-key")
    stub(f"{prefix}.transfer_backend", RForkTransferBackend=Mock)
    stub(f"{prefix}.seed_server", RForkSeedServerHandle=Mock, start_rfork_server=Mock())
    types = load("types")
    config = load("config")
    load("seed_client")
    client = load("planner_client")
    session = load("session")
    monkeypatch.setattr(session.atexit, "register", lambda callback: None)
    cfg = config.RForkConfig(
        "model", "strategy", "http://planner", request_timeout_sec=0.1, lease_release_retry_interval_sec=0.001
    )
    identity = types.RForkIdentity(0, 0, compatibility_fingerprint="fingerprint")
    lease = types.SeedLease("127.0.0.1", 1234, "private-user-id", 0, "model-key")
    return SimpleNamespace(types=types, client=client, session=session, config=cfg, identity=identity, lease=lease)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, "RELEASED"),
        (404, "RELEASED"),
        (400, "REJECTED"),
        (401, "REJECTED"),
        (405, "REJECTED"),
        (302, "REJECTED"),
        (408, "RETRYABLE"),
        (429, "RETRYABLE"),
        (503, "RETRYABLE"),
    ],
)
def test_release_classifies_response_without_changing_wire_protocol(runtime, monkeypatch, status, expected):
    post = Mock(return_value=SimpleNamespace(status_code=status, text="rejected"))
    monkeypatch.setattr(runtime.client.requests, "post", post)
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)
    assert client.release_seed_once(runtime.lease).name == expected
    post.assert_called_once_with(
        "http://planner/put_seed",
        headers={"SEED_IP": "127.0.0.1", "SEED_PORT": "1234", "USER_ID": "private-user-id", "SEED_RANK": "0"},
        timeout=0.1,
        allow_redirects=False,
    )


def test_release_body_is_bounded_sanitized_and_redacts_lease(runtime, monkeypatch, caplog):
    monkeypatch.setattr(
        runtime.client.requests,
        "post",
        Mock(
            return_value=SimpleNamespace(
                status_code=400,
                text="userID: private-user-id not found\n\x1b" + "x" * 1024,
            )
        ),
    )
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)
    assert not client.release_seed(runtime.lease)
    assert "private-user-id" not in caplog.text
    assert "<lease-id> not found" in caplog.text
    assert "\x1b" not in caplog.text
    assert "x" * 257 not in caplog.text
    runtime.client.requests.post.assert_called_once()


def test_transient_requests_are_bounded(runtime, monkeypatch):
    post = Mock(side_effect=requests.Timeout("slow"))
    monkeypatch.setattr(runtime.client.requests, "post", post)
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)
    assert not client.release_seed(runtime.lease)
    assert post.call_count == runtime.config.lease_release_max_attempts


def make_session(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.planner.acquire_seed.return_value = runtime.lease
    session.planner.remove_seed.return_value = True
    session.transfer_backend = Mock()
    session.transfer_backend.register_memory_region.return_value = True
    session.transfer_backend.read_weights_from_seed.return_value = True
    session.transfer_backend.unregister_memory_region.return_value = True
    session.transfer_backend.finalize_transfer_engine.return_value = True
    assert session.acquire_seed()
    return session


def run_and_join(session):
    # Hold the session lock until the worker reference is captured to avoid racing its completion.
    with session._lock:
        assert session.release_seed() is False
        worker = session.lease_release_thread
    assert worker is not None
    worker.join(2)
    assert not worker.is_alive()


def test_blocked_release_does_not_block_transfer_return_seed_start_or_shutdown(runtime, monkeypatch):
    session = make_session(runtime)
    entered, resume, startup_done = threading.Event(), threading.Event(), threading.Event()

    def release(lease):
        entered.set()
        assert resume.wait(3)
        return runtime.types.LeaseReleaseResult.RELEASED

    session.planner.release_seed_once.side_effect = release
    monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", lambda *args: object())
    results = []

    def startup():
        results.append(session.transfer_from_seed(object(), True))
        results.append(session.start_seed_service(object(), True))
        startup_done.set()

    thread = threading.Thread(target=startup, daemon=True)
    thread.start()
    try:
        assert entered.wait(1)
        assert startup_done.wait(1), "startup is waiting for lease release I/O"
        assert results == [True, runtime.types.RForkSeedServiceStartResult.DEFERRED]
        for _ in range(5):
            assert session.release_seed() is False
        worker = session.lease_release_thread
        # Shutdown must return without joining the blocked release or freeing live resources.
        assert session.shutdown() is False
        session.transfer_backend.finalize_transfer_engine.assert_not_called()
        session.planner.release_seed_once.assert_called_once()
    finally:
        resume.set()
        thread.join(2)
        if session.lease_release_thread is not None:
            session.lease_release_thread.join(2)
    assert not worker.is_alive()
    assert session.seed_lease is None
    runtime.session.start_rfork_server.assert_not_called()
    assert session.shutdown() is True


@pytest.mark.parametrize(("result", "attempts"), [("REJECTED", 1), ("RETRYABLE", 3)])
def test_release_failure_budget_does_not_restart_or_discard_lease(runtime, result, attempts):
    session = make_session(runtime)
    session.planner.release_seed_once.return_value = getattr(runtime.types.LeaseReleaseResult, result)
    run_and_join(session)
    assert session.seed_lease is runtime.lease
    assert session._lease_release_exhausted
    for _ in range(3):
        assert session.release_seed() is False
        assert session.start_seed_service(object(), True) is runtime.types.RForkSeedServiceStartResult.FAILED
    assert session.planner.release_seed_once.call_count == attempts
    assert not session.acquire_seed()
    assert session.shutdown() is False
    session.transfer_backend.finalize_transfer_engine.assert_not_called()


def test_retry_eventually_acknowledged_promotes_prepared_model_once(runtime, monkeypatch):
    session = make_session(runtime)
    model = object()
    session.state = runtime.types.RForkLifecycleState.REGISTERED
    session.planner.release_seed_once.side_effect = [
        runtime.types.LeaseReleaseResult.RETRYABLE,
        runtime.types.LeaseReleaseResult.RELEASED,
    ]
    promote = Mock(return_value=True)
    monkeypatch.setattr(session, "_start_seed_service_locked", promote)
    with session._lock:
        assert session.start_seed_service(model, True) is runtime.types.RForkSeedServiceStartResult.DEFERRED
        worker = session.lease_release_thread
    worker.join(2)
    assert not worker.is_alive()
    assert session.seed_lease is None
    promote.assert_called_once_with(model, True, None)
    assert session.planner.release_seed_once.call_count == 2


def test_fallback_does_not_wait_for_release_and_preserves_unresolved_status(runtime):
    session = make_session(runtime)
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.REJECTED
    with session._lock:
        outcome = session.prepare_for_fallback()
        worker = session.lease_release_thread
    worker.join(2)
    assert outcome.memory_reset and outcome.service_stopped and not outcome.lease_released
    assert session.seed_lease is runtime.lease
    assert session._lease_release_exhausted


def test_release_worker_start_failure_preserves_model(runtime, monkeypatch):
    session = make_session(runtime)
    monkeypatch.setattr(runtime.session.threading.Thread, "start", Mock(side_effect=RuntimeError("cannot start")))
    monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", lambda *args: object())
    assert session.transfer_from_seed(object(), True)
    assert session._lease_release_exhausted and session.seed_lease is runtime.lease
    assert session.lease_release_thread is None


def test_deferred_fallback_memory_preparation_stays_on_calling_thread(runtime, monkeypatch):
    session = make_session(runtime)
    caller = threading.get_ident()
    prepared_on = []
    model = object()
    session.transfer_backend.register_memory_region.side_effect = (
        lambda *args: prepared_on.append(threading.get_ident()) or True
    )
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED
    promoted = []

    def publish(*args):
        assert session.state is runtime.types.RForkLifecycleState.REGISTERED
        promoted.append(threading.get_ident())
        return True

    monkeypatch.setattr(session, "_start_seed_service_locked", publish)
    with session._lock:
        assert session.start_seed_service(model, True) is runtime.types.RForkSeedServiceStartResult.DEFERRED
        worker = session.lease_release_thread
    worker.join(2)
    assert not worker.is_alive()
    assert prepared_on == [caller]
    assert len(promoted) == 1 and promoted[0] != caller
