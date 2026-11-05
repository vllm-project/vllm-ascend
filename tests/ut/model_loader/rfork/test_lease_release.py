# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.ut.model_loader.rfork.session_test_utils import dummy_model, make_session, run_and_join


@pytest.mark.parametrize("tp_rank", [0, 2])
def test_session_passes_tp_rank_to_transfer_backend(runtime, monkeypatch, tp_rank):
    backend_factory = Mock()
    monkeypatch.setattr(runtime.session, "RForkTransferBackend", backend_factory)
    identity = replace(runtime.identity, tp_rank=tp_rank)

    session = runtime.session.RForkSession(runtime.config, identity)

    backend_factory.assert_called_once_with(tp_rank=tp_rank)
    assert session.transfer_backend is backend_factory.return_value


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, "RELEASED"),
        (404, "RELEASED"),
        (400, "REJECTED"),
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


@pytest.mark.parametrize(
    ("status", "expected"),
    [(200, "ACCEPTED"), (400, "REJECTED"), (408, "RETRYABLE"), (429, "RETRYABLE"), (503, "RETRYABLE")],
)
def test_seed_report_classifies_response_without_changing_wire_protocol(runtime, monkeypatch, status, expected):
    post = Mock(return_value=SimpleNamespace(status_code=status))
    monkeypatch.setattr(runtime.client.requests, "post", post)
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)
    client.bind_structural_digest("digest")

    assert client.report_seed_once(1234, seed_ip="127.0.0.1").status.name == expected
    post.assert_called_once_with(
        "http://planner/add_seed",
        headers={
            "SEED_KEY": client.seed_key,
            "SEED_IP": "127.0.0.1",
            "SEED_PORT": "1234",
            "SEED_RANK": "0",
            "SEED_REFCNT": "0",
        },
        timeout=0.1,
        allow_redirects=False,
    )


def test_acquire_seed_does_not_swallow_programming_errors(runtime, monkeypatch):
    monkeypatch.setattr(runtime.client.requests, "get", Mock(side_effect=AttributeError("unexpected bug")))
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)
    client.bind_structural_digest("digest")

    with pytest.raises(AttributeError, match="unexpected bug"):
        client.acquire_seed()


def test_shutdown_is_reentrant_safe_and_unregisters_atexit(runtime, monkeypatch):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    unregister = Mock()
    monkeypatch.setattr(runtime.session.atexit, "unregister", unregister)
    monkeypatch.setattr(session, "_stop_seed_service", Mock(return_value=True))
    nested_results = []

    def finalize():
        nested_results.append(session.shutdown())
        return True

    session.transfer_backend.finalize_transfer_engine.side_effect = finalize

    assert session.shutdown()
    assert session.shutdown()
    assert nested_results == [False]
    session.transfer_backend.finalize_transfer_engine.assert_called_once_with()
    unregister.assert_called_once_with(session._atexit_callback)


def test_blocked_release_does_not_block_inference_or_shutdown(runtime, monkeypatch):
    session = make_session(runtime)
    entered, resume, startup_done = threading.Event(), threading.Event(), threading.Event()

    def release(_lease):
        entered.set()
        assert resume.wait(3)
        return runtime.types.LeaseReleaseResult.RELEASED

    session.planner.release_seed_once.side_effect = release
    monkeypatch.setattr(
        runtime.session,
        "fetch_seed_transfer_info",
        lambda *args: SimpleNamespace(session_id="seed-session"),
    )
    results = []

    def startup():
        results.append(session.transfer_from_seed(object(), True))
        results.append(session.start_seed_service(dummy_model(), True))
        startup_done.set()

    thread = threading.Thread(target=startup, daemon=True)
    thread.start()
    try:
        assert entered.wait(1)
        assert startup_done.wait(1)
        assert results == [True, runtime.types.RForkSeedServiceStartResult.DEFERRED]
        assert session.shutdown() is False
        session.transfer_backend.finalize_transfer_engine.assert_not_called()
    finally:
        resume.set()
        thread.join(2)
        if session.lease_release_thread is not None:
            session.lease_release_thread.join(2)

    assert session.seed_lease is None
    assert session.shutdown() is True


def test_retry_acknowledgement_promotes_the_model_once(runtime, monkeypatch):
    session = make_session(runtime)
    model = dummy_model()
    session.state = runtime.types.RForkLifecycleState.READY
    session.planner.release_seed_once.side_effect = [
        runtime.types.LeaseReleaseResult.RETRYABLE,
        runtime.types.LeaseReleaseResult.RELEASED,
    ]
    promote = Mock(return_value=True)
    monkeypatch.setattr(session, "_start_seed_service", promote)

    assert session.start_seed_service(model, True) is runtime.types.RForkSeedServiceStartResult.DEFERRED
    session.lease_release_thread.join(2)

    assert session.seed_lease is None
    promote.assert_called_once_with(model, True, None)
    assert session.planner.release_seed_once.call_count == 2


def test_permanent_release_rejection_preserves_the_unresolved_lease(runtime):
    session = make_session(runtime)
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.REJECTED

    run_and_join(session)

    assert session.seed_lease is runtime.lease
    assert session._lease_release_exhausted
    assert session.planner.release_seed_once.call_count == 1
    assert session.start_seed_service(dummy_model(), True) is runtime.types.RForkSeedServiceStartResult.FAILED
    assert session.shutdown() is False


def test_transient_release_recovers_after_fast_attempt_budget(runtime, monkeypatch):
    session = make_session(runtime)
    session.config = replace(session.config, lease_release_max_attempts=2, lease_release_retry_interval_sec=0.01)
    monkeypatch.setattr(runtime.session, "LEASE_RELEASE_DEGRADED_RETRY_INTERVAL_SEC", 0.02)
    session.state = runtime.types.RForkLifecycleState.READY
    session.planner.release_seed_once.side_effect = [
        *[runtime.types.LeaseReleaseResult.RETRYABLE] * 4,
        runtime.types.LeaseReleaseResult.RELEASED,
    ]
    promote = Mock(return_value=True)
    monkeypatch.setattr(session, "_start_seed_service", promote)

    assert session.start_seed_service(dummy_model(), True) is runtime.types.RForkSeedServiceStartResult.DEFERRED
    session.lease_release_thread.join(2)

    assert session.seed_lease is None
    assert not session._lease_release_exhausted
    assert session.planner.release_seed_once.call_count == 5
    promote.assert_called_once()


@pytest.mark.parametrize("failure", ["metadata", "read"])
def test_transfer_failure_requires_cleanup_before_publication(runtime, monkeypatch, failure):
    session = make_session(runtime)
    monkeypatch.setattr(
        runtime.session,
        "fetch_seed_transfer_info",
        lambda *args: None if failure == "metadata" else object(),
    )
    session.transfer_backend.read_weights_from_seed.return_value = failure != "read"
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED

    assert not session.transfer_from_seed(object(), True)
    assert session.state is runtime.types.RForkLifecycleState.CLEANUP_REQUIRED
    run_and_join(session)
    assert session.start_seed_service(dummy_model(), True) is runtime.types.RForkSeedServiceStartResult.FAILED
    assert session.prepare_for_fallback().memory_reset


def test_shutdown_retains_resources_for_an_unresolved_lease(runtime, caplog):
    session = make_session(runtime)

    assert not session.shutdown()
    assert session.seed_lease is runtime.lease
    session.transfer_backend.finalize_transfer_engine.assert_not_called()
    assert "expiry/reclamation policy" in caplog.text


def test_register_destination_binds_structural_digest(runtime, monkeypatch):
    backend = Mock()
    backend.register_memory_region.return_value = True
    monkeypatch.setattr(runtime.session, "RForkTransferBackend", Mock(return_value=backend))
    session = runtime.session.RForkSession(runtime.config, runtime.identity)

    assert session.register_destination(dummy_model(), True)
    assert session.planner.structural_digest is not None
    assert session.planner.seed_key == "model-key"


def test_seed_service_refuses_advertisement_after_digest_drift(runtime, monkeypatch, caplog):
    backend = Mock()
    backend.register_memory_region.return_value = True
    backend.unregister_memory_region.return_value = True
    monkeypatch.setattr(runtime.session, "RForkTransferBackend", Mock(return_value=backend))
    digests = iter(["digest-a", "digest-b"])
    monkeypatch.setattr(runtime.session, "build_structural_digest", lambda tensors: next(digests))
    session = runtime.session.RForkSession(runtime.config, runtime.identity)

    assert session.register_destination(dummy_model(), True)
    assert session.prepare_for_fallback().can_schedule_seed
    assert session.start_seed_service(dummy_model(), True) is runtime.types.RForkSeedServiceStartResult.FAILED
    assert session.planner.structural_digest == "digest-a"
    assert "structure changed" in caplog.text
