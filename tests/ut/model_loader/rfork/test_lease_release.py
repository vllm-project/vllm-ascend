# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.ut.model_loader.rfork.session_test_utils import make_session, run_and_join


@pytest.mark.parametrize(
    ("status", "expected"),
    [(200, "RELEASED"), (404, "RELEASED"), (400, "REJECTED"), (408, "RETRYABLE"), (503, "RETRYABLE")],
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


def test_blocked_release_does_not_block_inference_or_shutdown(runtime, monkeypatch):
    session = make_session(runtime)
    entered, resume, startup_done = threading.Event(), threading.Event(), threading.Event()

    def release(_lease):
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
    model = object()
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


@pytest.mark.parametrize(("result", "attempts"), [("REJECTED", 1), ("RETRYABLE", 3)])
def test_release_failure_budget_preserves_the_unresolved_lease(runtime, result, attempts):
    session = make_session(runtime)
    session.planner.release_seed_once.return_value = getattr(runtime.types.LeaseReleaseResult, result)

    run_and_join(session)

    assert session.seed_lease is runtime.lease
    assert session._lease_release_exhausted
    assert session.planner.release_seed_once.call_count == attempts
    assert session.start_seed_service(object(), True) is runtime.types.RForkSeedServiceStartResult.FAILED
    assert session.shutdown() is False


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
    assert session.start_seed_service(object(), True) is runtime.types.RForkSeedServiceStartResult.FAILED
    assert session.prepare_for_fallback().memory_reset


def test_shutdown_retains_resources_for_an_unresolved_lease(runtime, caplog):
    session = make_session(runtime)

    assert not session.shutdown()
    assert session.seed_lease is runtime.lease
    session.transfer_backend.finalize_transfer_engine.assert_not_called()
    assert "expiry/reclamation policy" in caplog.text
