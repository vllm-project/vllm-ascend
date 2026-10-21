# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class _SeedHandle:
    def __init__(self):
        self.is_alive = True

    def stop(self):
        self.is_alive = False
        return True


class _PausedRetryEvent:
    """Keep a sleeping retry worker alive after stop to reproduce a shutdown scheduling race."""

    def __init__(self):
        self.event = threading.Event()
        self.waiting = threading.Event()
        self.resume = threading.Event()

    def set(self):
        self.event.set()

    def is_set(self):
        return self.event.is_set()

    def wait(self, timeout):
        self.waiting.set()
        stopped = self.event.wait(timeout)
        assert self.resume.wait(3)
        return stopped


class _PausedResultLock:
    """Pause the release worker immediately after it unlocks a recorded retryable response."""

    def __init__(self, session):
        self.lock = session._lock
        self.session = session
        self.pause_next_exit = False
        self.recorded = threading.Event()
        self.resume = threading.Event()

    def __enter__(self):
        return self.lock.__enter__()

    def __exit__(self, *args):
        pause = self.pause_next_exit and threading.current_thread() is self.session.lease_release_thread
        if pause:
            self.pause_next_exit = False
        result = self.lock.__exit__(*args)
        if pause:
            self.recorded.set()
            assert self.resume.wait(3)
        return result


def _session(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner.bind_structural_digest("digest")
    session.transfer_backend.finalize_transfer_engine.return_value = True
    session.state = runtime.types.RForkLifecycleState.REGISTERED
    return session


@pytest.mark.parametrize("status", [404, 405, 501])
def test_missing_removal_route_stops_metadata_but_retains_published_memory(runtime, monkeypatch, status):
    session = _session(runtime)
    handle = _SeedHandle()
    session.seed_server = handle
    session.state = runtime.types.RForkLifecycleState.SERVING
    session.planner.last_advertisement = runtime.types.SeedAdvertisement("127.0.0.1", 1234, 0)
    post = Mock(return_value=SimpleNamespace(status_code=status))
    monkeypatch.setattr(runtime.client.requests, "post", post)

    assert not session.shutdown()
    assert not handle.is_alive
    assert session.seed_server is handle
    assert session.planner.last_advertisement is not None
    session.transfer_backend.finalize_transfer_engine.assert_not_called()
    session.transfer_backend.unregister_memory_region.assert_not_called()


@pytest.mark.parametrize("status", [409, 503])
def test_pending_removal_keeps_metadata_available_for_existing_receivers(runtime, monkeypatch, status):
    session = _session(runtime)
    handle = _SeedHandle()
    session.seed_server = handle
    session.state = runtime.types.RForkLifecycleState.SERVING
    session.planner.last_advertisement = runtime.types.SeedAdvertisement("127.0.0.1", 1234, 0)
    monkeypatch.setattr(runtime.client.requests, "post", Mock(return_value=SimpleNamespace(status_code=status)))
    monkeypatch.setattr(runtime.client, "SEED_REMOVAL_RETRY_BACKOFF_SEC", 0)

    assert not session.shutdown()
    assert handle.is_alive
    assert session.seed_server is handle
    assert session.planner.last_advertisement is not None
    session.transfer_backend.finalize_transfer_engine.assert_not_called()


def test_shutdown_returns_an_unreleased_legacy_lease_before_finalizing(runtime, monkeypatch):
    session = _session(runtime)
    session.seed_lease = replace(runtime.lease, lease_ttl_sec=None)
    post = Mock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr(runtime.client.requests, "post", post)

    assert session.shutdown()
    assert session.seed_lease is None
    assert session.state is runtime.types.RForkLifecycleState.FINALIZED
    assert session.shutdown()
    post.assert_called_once_with(
        "http://planner/put_seed",
        headers={"SEED_IP": "127.0.0.1", "SEED_PORT": "1234", "USER_ID": "private-user-id", "SEED_RANK": "0"},
        timeout=0.1,
        allow_redirects=False,
    )
    session.transfer_backend.finalize_transfer_engine.assert_called_once_with()


def test_unresolved_legacy_quota_does_not_pin_unpublished_receiver_memory(runtime, monkeypatch, caplog):
    session = _session(runtime)
    session.seed_lease = replace(runtime.lease, lease_ttl_sec=None)
    post = Mock(return_value=SimpleNamespace(status_code=503, text="unavailable"))
    monkeypatch.setattr(runtime.client.requests, "post", post)

    assert not session.shutdown()
    assert session.state is runtime.types.RForkLifecycleState.FINALIZED
    assert session.seed_lease is not None
    session.transfer_backend.finalize_transfer_engine.assert_called_once_with()
    assert "legacy" in caplog.text.lower()
    assert not session.shutdown()
    assert post.call_count == 2
    session.transfer_backend.finalize_transfer_engine.assert_called_once_with()


def test_legacy_quota_acknowledgement_can_finish_after_receiver_finalization(runtime, monkeypatch):
    session = _session(runtime)
    callbacks = [session._atexit_callback]
    monkeypatch.setattr(runtime.session.atexit, "unregister", callbacks.remove)
    session.seed_lease = replace(runtime.lease, lease_ttl_sec=None)
    entered, resume, finished = threading.Event(), threading.Event(), threading.Event()
    outcomes = []

    def post(*_args, **_kwargs):
        entered.set()
        assert resume.wait(3)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(runtime.client.requests, "post", post)
    with session._lock:
        session._release_seed_locked()
        release_worker = session.lease_release_thread

    def shutdown():
        outcomes.append(session.shutdown())
        finished.set()

    shutdown_worker = threading.Thread(target=shutdown, daemon=True)
    try:
        assert entered.wait(1)
        shutdown_worker.start()
        assert finished.wait(1)
        assert outcomes == [False]
        assert session.state is runtime.types.RForkLifecycleState.FINALIZED
        assert session.seed_lease is not None
        assert callbacks == [session._atexit_callback]
        session.transfer_backend.finalize_transfer_engine.assert_called_once_with()
    finally:
        resume.set()
        release_worker.join(2)
        if shutdown_worker.ident is not None:
            shutdown_worker.join(2)

    assert session.seed_lease is None
    assert callbacks == []
    assert session.shutdown()


def test_legacy_shutdown_retains_backend_when_native_finalization_fails(runtime, monkeypatch):
    session = _session(runtime)
    session.seed_lease = replace(runtime.lease, lease_ttl_sec=None)
    session.transfer_backend.finalize_transfer_engine.return_value = False
    monkeypatch.setattr(runtime.client.requests, "post", Mock(return_value=SimpleNamespace(status_code=200)))

    assert not session.shutdown()
    assert session.seed_lease is None
    assert session.state is not runtime.types.RForkLifecycleState.FINALIZED
    session.transfer_backend.finalize_transfer_engine.assert_called_once_with()


def test_legacy_shutdown_releases_quota_when_retry_worker_is_sleeping(runtime, monkeypatch):
    session = _session(runtime)
    session.seed_lease = replace(runtime.lease, lease_ttl_sec=None)
    session.config = replace(runtime.config, lease_release_retry_interval_sec=30)
    stop_event = _PausedRetryEvent()
    session.lease_release_stop_event = stop_event
    post = Mock(side_effect=[SimpleNamespace(status_code=503, text="unavailable"), SimpleNamespace(status_code=200)])
    monkeypatch.setattr(runtime.client.requests, "post", post)
    with session._lock:
        session._release_seed_locked()
        worker = session.lease_release_thread

    try:
        assert stop_event.waiting.wait(1)
        assert session.shutdown()
        assert session.seed_lease is None
        assert post.call_count == 2
        session.transfer_backend.finalize_transfer_engine.assert_called_once_with()
    finally:
        stop_event.set()
        stop_event.resume.set()
        worker.join(2)


def test_legacy_shutdown_observes_completed_release_result_atomically(runtime, monkeypatch):
    session = _session(runtime)
    session.seed_lease = replace(runtime.lease, lease_ttl_sec=None)
    paused_lock = _PausedResultLock(session)
    session._lock = paused_lock
    record_result = session._record_lease_release_locked
    post = Mock(side_effect=[SimpleNamespace(status_code=503, text="unavailable"), SimpleNamespace(status_code=200)])

    def record(lease, result):
        finished = record_result(lease, result)
        if result is runtime.types.LeaseReleaseResult.RETRYABLE:
            paused_lock.pause_next_exit = True
        return finished

    monkeypatch.setattr(runtime.client.requests, "post", post)
    monkeypatch.setattr(session, "_record_lease_release_locked", record)
    with session._lock:
        session._release_seed_locked()
        worker = session.lease_release_thread

    try:
        assert paused_lock.recorded.wait(1)
        assert session.shutdown()
        assert session.seed_lease is None
        assert post.call_count == 2
    finally:
        paused_lock.resume.set()
        worker.join(2)


@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}])
def test_seed_removal_capability_rejects_non_boolean_configuration(runtime, value):
    with pytest.raises(ValueError, match="rfork_seed_removal_supported"):
        type(runtime.config).from_extra_config({"rfork_seed_removal_supported": value})
