# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.model_loader.rfork.session_test_utils import (
    dummy_model,
    join_seed_worker,
    make_session,
    run_and_join,
    wait_for_lease_release,
)

from .rfork_test_support import _load_module


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
        results.append(session.read_weights_from_seed())
        session.validate_transferred_model_layout(dummy_model())
        results.append(session.schedule_seed_service())
        startup_done.set()

    thread = threading.Thread(target=startup, daemon=True)
    thread.start()
    try:
        assert entered.wait(1)
        assert startup_done.wait(1)
        assert results == [True, True]
        assert session.shutdown() is False
        session.transfer_backend.finalize_transfer_engine.assert_not_called()
    finally:
        resume.set()
        thread.join(2)
        if session._seed_worker_thread is not None:
            session._seed_worker_thread.join(2)

    assert session.seed_lease is None
    assert session.shutdown() is True


def test_retry_acknowledgement_promotes_the_model_once(runtime, monkeypatch):
    session = make_session(runtime)
    session.state = runtime.types.RForkLifecycleState.READY
    session.planner.release_seed_once.side_effect = [
        runtime.types.LeaseReleaseResult.RETRYABLE,
        runtime.types.LeaseReleaseResult.RELEASED,
    ]
    workers = []
    release_once = session.planner.release_seed_once
    results = iter(release_once.side_effect)

    def release(_lease):
        workers.append(threading.current_thread())
        return next(results)

    release_once.side_effect = release

    def publish(*args, **kwargs):
        workers.append(threading.current_thread())
        return True

    promote = Mock(side_effect=publish)
    monkeypatch.setattr(session, "_serve_seed", promote)

    assert session.schedule_seed_service() is (True)
    join_seed_worker(session, 2)

    assert session.seed_lease is None
    promote.assert_called_once_with()
    assert session.planner.release_seed_once.call_count == 2
    assert len(workers) == 3 and len(set(workers)) == 1


def test_seed_publication_runs_in_background_after_caller_registration(runtime, monkeypatch):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    entered, resume = threading.Event(), threading.Event()
    caller = threading.current_thread()
    session.transfer_backend.register_memory_region.side_effect = lambda *args, **kwargs: (
        threading.current_thread() is caller
    )
    session.transfer_backend.registered_structural_digest = "digest"

    def publish():
        assert threading.current_thread() is not caller
        entered.set()
        assert resume.wait(3)
        return True

    monkeypatch.setattr(session, "_serve_seed", publish)
    try:
        assert session.register_local_weights_for_seed(dummy_model())
        assert session.schedule_seed_service() is True
        assert entered.wait(1)
    finally:
        resume.set()
        join_seed_worker(session, 2)
        assert session.shutdown()


def test_seed_worker_creation_failure_resets_registration(runtime, monkeypatch):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.transfer_backend.registered_structural_digest = "digest"
    assert session.register_local_weights_for_seed(dummy_model())
    monkeypatch.setattr(runtime.session.threading.Thread, "start", Mock(side_effect=RuntimeError("no thread")))

    assert session.schedule_seed_service() is False
    assert not session._seed_publication_pending
    session.transfer_backend.unregister_memory_region.assert_called_once()


def test_seed_worker_does_not_wait_to_retry_an_already_released_lease(runtime, monkeypatch):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    wait = Mock()
    monkeypatch.setattr(session._shutdown_event, "wait", wait)

    session._retry_seed_lease_release()

    wait.assert_not_called()


def test_worker_releases_lease_then_waits_for_model_before_publishing(runtime, monkeypatch):
    session = make_session(runtime)
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED
    released = threading.Event()
    release = session._retry_seed_lease_release

    def release_lease():
        release()
        released.set()

    monkeypatch.setattr(session, "_retry_seed_lease_release", release_lease)
    publish = Mock(return_value=True)
    monkeypatch.setattr(session, "_serve_seed", publish)
    with session.loading():
        with session._lock:
            session._release_seed_locked()
            worker = session._seed_worker_thread
        assert released.wait(2)
        assert worker.is_alive()
        session.state = runtime.types.RForkLifecycleState.READY
        assert session.schedule_seed_service()
        assert session._seed_worker_thread is worker
    join_seed_worker(session, 2)

    publish.assert_called_once_with()
    session.planner.release_seed_once.assert_called_once()


def test_publish_after_release_worker_has_finished(runtime, monkeypatch):
    session = make_session(runtime)
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED
    publish = Mock(return_value=True)
    monkeypatch.setattr(session, "_serve_seed", publish)

    with session._lock:
        assert not session._release_seed_locked()
        worker = session._seed_worker_thread
    worker.join(2)
    assert not worker.is_alive()
    assert session.seed_lease is None
    session.state = runtime.types.RForkLifecycleState.READY
    assert session.schedule_seed_service()
    join_seed_worker(session, 2)
    publish.assert_called_once_with()


def test_shutdown_waits_for_model_loading_before_releasing_resources(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner.remove_seed = Mock(return_value=True)
    session.transfer_backend.finalize_transfer_engine.return_value = True
    results = []
    with session.loading():
        worker = threading.Thread(target=lambda: results.append(session.shutdown()))
        worker.start()
        assert session._shutdown_event.wait(1)
        assert not session.lease_renew_stop_event.is_set()
        session.transfer_backend.finalize_transfer_engine.assert_not_called()
    worker.join(2)
    assert not worker.is_alive()
    assert results == [True]


def test_session_cannot_load_a_second_model(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    with session.loading():
        pass
    with pytest.raises(runtime.session.RForkSessionUnavailableError, match="already loaded"), session.loading():
        pytest.fail("a second model load must not start")
    session.transfer_backend.unregister_memory_region.assert_not_called()
    assert session.shutdown()


def test_worker_exits_if_loading_finishes_without_publication(runtime, monkeypatch):
    session = make_session(runtime)
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED
    publish = Mock()
    monkeypatch.setattr(session, "_serve_seed", publish)
    with session.loading():
        with session._lock:
            session._release_seed_locked()
        wait_for_lease_release(session)
        assert session._seed_worker_thread.is_alive()
    join_seed_worker(session)
    publish.assert_not_called()
    assert session.shutdown()


def test_permanent_release_rejection_preserves_the_unresolved_lease(runtime):
    session = make_session(runtime)
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.REJECTED

    run_and_join(session)

    assert session.seed_lease is runtime.lease
    assert session._lease_release_stopped
    assert session.planner.release_seed_once.call_count == 1
    assert session.schedule_seed_service() is False
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
    monkeypatch.setattr(session, "_serve_seed", promote)

    assert session.schedule_seed_service() is True
    join_seed_worker(session, 2)

    assert session.seed_lease is None
    assert not session._lease_release_stopped
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

    assert not session.read_weights_from_seed()
    assert session.state is runtime.types.RForkLifecycleState.CLEANUP_REQUIRED
    assert session.schedule_seed_service() is False
    assert session.prepare_for_fallback()
    wait_for_lease_release(session)
    assert session.shutdown()
    join_seed_worker(session)


def test_shutdown_retains_resources_for_an_unresolved_lease(runtime, caplog):
    session = make_session(runtime)

    assert not session.shutdown()
    assert session.seed_lease is runtime.lease
    session.transfer_backend.finalize_transfer_engine.assert_not_called()
    assert "expiry/reclamation policy" in caplog.text


def test_register_receiver_weights_binds_structural_digest(runtime, monkeypatch):
    backend = Mock()
    backend.register_memory_region.return_value = True
    backend.registered_structural_digest = "registered-digest"
    monkeypatch.setattr(runtime.session, "RForkTransferBackend", Mock(return_value=backend))
    session = runtime.session.RForkSession(runtime.config, runtime.identity)

    assert session.register_receiver_weights(dummy_model())
    assert session.planner.structural_digest == "registered-digest"
    assert session.planner.seed_key == "model-key"


def test_renewal_start_failure_releases_lease_and_requires_cleanup(runtime, monkeypatch):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.planner.acquire_seed.return_value = runtime.lease
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED
    session.transfer_backend.register_memory_region.return_value = True
    session.transfer_backend.registered_structural_digest = "digest"
    session.transfer_backend.unregister_memory_region.return_value = True
    assert session.register_receiver_weights(dummy_model())
    with monkeypatch.context() as context:
        context.setattr(runtime.session.threading.Thread, "start", Mock(side_effect=RuntimeError("no thread")))
        with pytest.raises(RuntimeError, match="lease renewal"):
            session.acquire_seed_lease()
    assert session.seed_lease is runtime.lease
    session.planner.release_seed_once.assert_not_called()
    assert session.state is runtime.types.RForkLifecycleState.CLEANUP_REQUIRED
    assert session.prepare_for_fallback()
    wait_for_lease_release(session)
    session.planner.release_seed_once.assert_called_once_with(runtime.lease)
    assert session.shutdown()
    join_seed_worker(session)
    assert session.seed_lease is None


def test_renewal_failure_does_not_change_transfer_result(runtime, monkeypatch):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.planner.acquire_seed.return_value = replace(runtime.lease, lease_ttl_sec=1.0)
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED
    session.planner.remove_seed.return_value = True
    session.transfer_backend.register_memory_region.return_value = True
    session.transfer_backend.registered_structural_digest = "digest"
    session.transfer_backend.unregister_memory_region.return_value = True
    renewed = threading.Event()

    def fail_renewal(_lease):
        renewed.set()
        return False

    def read(seed_info):
        assert renewed.wait(2)
        return True

    session.planner.renew_seed_once.side_effect = fail_renewal
    session.transfer_backend.read_weights_from_seed.side_effect = read
    monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", lambda *args: SimpleNamespace(session_id="seed"))

    assert session.register_receiver_weights(dummy_model())
    assert session.acquire_seed_lease()
    assert session.read_weights_from_seed()
    assert session.prepare_for_fallback()
    wait_for_lease_release(session)
    assert session.shutdown()


@pytest.mark.parametrize("failure", [None, "registration", "seed", "transfer"])
def test_receiver_registers_acquires_and_reads_weights(runtime, monkeypatch, failure):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock(seed_key="model-key")
    session.planner.acquire_seed.return_value = None if failure == "seed" else runtime.lease
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED
    session.planner.remove_seed.return_value = True
    session.transfer_backend.register_memory_region.return_value = failure != "registration"
    session.transfer_backend.registered_structural_digest = "digest"
    session.transfer_backend.read_weights_from_seed.return_value = failure != "transfer"
    session.transfer_backend.unregister_memory_region.return_value = True
    monkeypatch.setattr(
        runtime.session, "fetch_seed_transfer_info", lambda *args: SimpleNamespace(session_id="seed-session")
    )

    try:
        registered = session.register_receiver_weights(dummy_model())
        assert registered is (failure != "registration")
        if registered:
            acquired = session.acquire_seed_lease()
            assert acquired is (failure != "seed")
            if acquired:
                assert session.read_weights_from_seed() is (failure is None)
    finally:
        session.prepare_for_fallback()
        if session.seed_lease is not None:
            wait_for_lease_release(session)
        assert session.shutdown()
        join_seed_worker(session, 2)


def test_seed_service_refuses_advertisement_after_digest_drift(runtime, monkeypatch, caplog):
    backend = Mock()
    digests = iter(["digest-a", "digest-b"])

    def register(*args, **kwargs):
        backend.registered_structural_digest = next(digests)
        return True

    backend.register_memory_region.side_effect = register
    backend.unregister_memory_region.return_value = True
    monkeypatch.setattr(runtime.session, "RForkTransferBackend", Mock(return_value=backend))
    session = runtime.session.RForkSession(runtime.config, runtime.identity)

    assert session.register_receiver_weights(dummy_model())
    assert session.prepare_for_fallback()
    assert session.register_local_weights_for_seed(dummy_model()) is False
    assert session.planner.structural_digest == "digest-a"
    assert "structure changed" in caplog.text


class _SeedHandle:
    def __init__(self):
        self.is_alive = True

    def stop(self):
        self.is_alive = False
        return True


def _session(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner.bind_structural_digest("digest")
    session.transfer_backend.finalize_transfer_engine.return_value = True
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


def test_shutdown_refuses_a_transfer_after_the_legacy_release_started(runtime, monkeypatch):
    session = _session(runtime)
    session.seed_lease = replace(runtime.lease, lease_ttl_sec=None)
    entered, resume = threading.Event(), threading.Event()
    outcomes = []

    def release(_lease):
        entered.set()
        assert resume.wait(3)
        return runtime.types.LeaseReleaseResult.RELEASED

    monkeypatch.setattr(session.planner, "release_seed_once", release)
    monkeypatch.setattr(
        runtime.session, "fetch_seed_transfer_info", lambda *args: SimpleNamespace(session_id="seed-session")
    )
    worker = threading.Thread(target=lambda: outcomes.append(session.shutdown()), daemon=True)
    worker.start()
    try:
        # put_seed is in flight outside _lock, so the lease may already be gone on the planner.
        assert entered.wait(1)
        assert not session.read_weights_from_seed()
        session.transfer_backend.read_weights_from_seed.assert_not_called()
    finally:
        resume.set()
        worker.join(2)

    assert outcomes == [True]
    assert session.seed_lease is None


def _configure_real_session(runtime, monkeypatch, model):
    prefix = "vllm_ascend.model_loader.rfork"
    layout = sys.modules[f"{prefix}.tensor_layout"]
    transfer = _load_module(monkeypatch, f"{prefix}.transfer_backend", "transfer_backend.py")
    monkeypatch.setattr(runtime.session, "RForkTransferBackend", transfer.RForkTransferBackend)
    monkeypatch.setattr(
        runtime.client,
        "build_seed_key",
        lambda **kwargs: f"model-key:{kwargs['structural_digest']}",
    )
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda tensor: True)
    monkeypatch.setattr(layout, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(transfer, "read_npu_format", lambda tensor: 0)

    def memory_snapshot():
        blocks = [
            {
                "address": tensor.data_ptr(),
                "size": tensor.numel() * tensor.element_size(),
                "state": "active_allocated",
            }
            for _, tensor in layout.collect_transferable_tensors(model)
        ]
        return [{"blocks": blocks}]

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=memory_snapshot)),
        raising=False,
    )

    success = SimpleNamespace(is_error=lambda: False, to_string=lambda: "success")
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    backend = session.transfer_backend
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda registrations: success,
        batch_unregister_memory=lambda addresses: success,
        batch_transfer_sync_read=lambda *args: success,
        finalize=lambda: success,
    )
    backend._memory_registration_cls = lambda *fields: fields
    backend.transfer_session_id = "receiver-session"

    # Stub external scheduling hooks while exercising real registration and lifecycle logic.
    monkeypatch.setattr(session, "_run_seed_heartbeat", lambda *args: None)
    monkeypatch.setattr(session.planner, "report_seed_once", lambda *args, **kwargs: True)
    monkeypatch.setattr(session.planner, "remove_seed", lambda *args, **kwargs: True)
    server = Mock(is_alive=True, port=1234, stop=Mock(return_value=True))
    start_server = Mock(return_value=server)
    monkeypatch.setattr(runtime.session, "create_rfork_server", start_server)
    return session, backend, layout, start_server


def _transfer_model(runtime, monkeypatch, session, backend, model):
    seed_info = runtime.types.SeedTransferInfo("seed-session", dict(backend.weight_manifest))
    monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", lambda *args: seed_info)
    session.seed_lease = runtime.lease
    assert session.read_weights_from_seed()


def test_fallback_without_bound_identity_binds_successful_registration(runtime, monkeypatch):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    session, backend, layout, start_server = _configure_real_session(runtime, monkeypatch, model)

    try:
        assert session.planner.structural_digest is None
        result = session.register_local_weights_for_seed(model) and session.schedule_seed_service()

        assert result is True
        join_seed_worker(session, 2)
        assert session.planner.structural_digest == backend.registered_structural_digest
        assert session.planner.seed_key == f"model-key:{backend.registered_structural_digest}"
        assert start_server.call_count == 1
        assert start_server.call_args.args[0] == session.planner.seed_key
    finally:
        session.seed_lease = None
        assert session.shutdown()


def test_fallback_with_changed_structure_does_not_publish_under_old_identity(runtime, monkeypatch):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2, 3))
    session, _, _, start_server = _configure_real_session(runtime, monkeypatch, model)
    try:
        assert session.register_receiver_weights(model)
        original_key = session.planner.seed_key
        assert session.prepare_for_fallback()
        model.weight = torch.nn.Parameter(torch.ones(3, 2))

        assert session.register_local_weights_for_seed(model) is False
        assert session.planner.seed_key == original_key
        start_server.assert_not_called()
    finally:
        assert session.shutdown()


def test_final_shape_drift_rejects_seed_publication(runtime, monkeypatch):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    session, _, _, start_server = _configure_real_session(runtime, monkeypatch, model)
    release = threading.Event()
    try:
        assert session.register_receiver_weights(model)

        def wait_for_release():
            assert release.wait(3)
            session.seed_lease = None

        monkeypatch.setattr(session, "_retry_seed_lease_release", wait_for_release)
        _transfer_model(runtime, monkeypatch, session, session.transfer_backend, model)
        model.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(3, 2))
        with pytest.raises(RuntimeError, match="final tensor layout validation failed: .*structure differs"):
            session.validate_transferred_model_layout(model)
        assert not session.schedule_seed_service()
        start_server.assert_not_called()
    finally:
        release.set()
        assert session.prepare_for_fallback()
        wait_for_lease_release(session)
        assert session.shutdown()
        join_seed_worker(session, 2)


def test_fallback_publishes_with_worker_waiting_after_lease_release(runtime, monkeypatch):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2, 3))
    session, backend, _, start_server = _configure_real_session(runtime, monkeypatch, model)
    monkeypatch.setattr(session.planner, "release_seed_once", lambda lease: runtime.types.LeaseReleaseResult.RELEASED)
    try:
        with session.loading():
            assert session.register_receiver_weights(model)
            _transfer_model(runtime, monkeypatch, session, backend, model)
            assert session._seed_worker_thread is None
            assert session.prepare_for_fallback()
            worker = session._seed_worker_thread
            wait_for_lease_release(session)
            assert worker.is_alive()
            assert session.register_local_weights_for_seed(model)
            assert session.schedule_seed_service()
            assert session._seed_worker_thread is worker
        join_seed_worker(session)
        start_server.assert_called_once()
    finally:
        assert session.shutdown()


def _published_session(runtime):
    """A SERVING session whose seed server and planner are cheap test doubles."""
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock()
    session.transfer_backend = Mock()
    session.transfer_backend.finalize_transfer_engine.return_value = True
    session.seed_server = SimpleNamespace(stop=lambda: True)
    session.heartbeat_thread = None
    session.state = runtime.types.RForkLifecycleState.SERVING
    return session


def test_final_shutdown_exempts_unsupported_removal(runtime):
    session = _published_session(runtime)
    session.planner.remove_seed.return_value = runtime.types.SeedRemovalResult.UNSUPPORTED

    assert session.shutdown(final=True) is True

    session.transfer_backend.finalize_transfer_engine.assert_called_once()
    assert session.state is runtime.types.RForkLifecycleState.FINALIZED
    assert session.seed_server is None


@pytest.mark.parametrize("outcome", ["UNCONFIRMED", "DRAINING", "RETRYABLE"])
def test_final_shutdown_does_not_exempt_other_removal_outcomes(runtime, outcome):
    session = _published_session(runtime)
    session.planner.remove_seed.return_value = getattr(runtime.types.SeedRemovalResult, outcome)

    assert session.shutdown(final=True) is False

    session.transfer_backend.finalize_transfer_engine.assert_not_called()


def test_explicit_shutdown_keeps_unsupported_as_failure(runtime):
    session = _published_session(runtime)
    session.planner.remove_seed.return_value = runtime.types.SeedRemovalResult.UNSUPPORTED

    assert session.shutdown() is False

    session.transfer_backend.finalize_transfer_engine.assert_not_called()


def test_final_shutdown_still_requires_server_stop_success(runtime):
    session = _published_session(runtime)
    session.planner.remove_seed.return_value = runtime.types.SeedRemovalResult.UNSUPPORTED
    session.seed_server = SimpleNamespace(stop=lambda: False)

    assert session.shutdown(final=True) is False

    session.transfer_backend.finalize_transfer_engine.assert_not_called()


def test_atexit_callback_requests_final_shutdown(runtime, monkeypatch):
    session = _published_session(runtime)
    session.planner.remove_seed.return_value = runtime.types.SeedRemovalResult.UNSUPPORTED
    calls = []

    def record_shutdown(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(session, "shutdown", record_shutdown)

    assert session._atexit_callback() is True

    assert calls == [{"final": True}]


def test_shutdown_releases_legacy_lease_without_ttl(runtime):
    session = make_session(runtime)
    legacy_lease = replace(runtime.lease, lease_ttl_sec=None)
    session.seed_lease = legacy_lease
    session.planner.release_seed_once.return_value = runtime.types.LeaseReleaseResult.RELEASED

    assert session.shutdown()

    session.planner.release_seed_once.assert_called_once_with(legacy_lease)
    assert session.seed_lease is None
    assert session.state is runtime.types.RForkLifecycleState.FINALIZED
    session.transfer_backend.finalize_transfer_engine.assert_called_once_with()


def test_failed_deferred_promotion_cleans_up_seed_start(runtime, monkeypatch):
    session = make_session(runtime)
    session.state = runtime.types.RForkLifecycleState.READY
    session.planner.release_seed_once.side_effect = [
        runtime.types.LeaseReleaseResult.RETRYABLE,
        runtime.types.LeaseReleaseResult.RELEASED,
    ]
    serve = Mock(return_value=False)
    monkeypatch.setattr(session, "_serve_seed", serve)
    cleanup = Mock()
    monkeypatch.setattr(session, "_cleanup_failed_seed_start", cleanup)

    assert session.schedule_seed_service() is True
    join_seed_worker(session, 2)

    assert session.seed_lease is None
    assert session.planner.release_seed_once.call_count == 2
    serve.assert_called_once_with()
    cleanup.assert_called_once_with()


def test_seed_service_uses_dynamic_port_when_resolved_port_overflows(runtime, monkeypatch):
    session = make_session(runtime)
    session.seed_lease = None
    session.state = runtime.types.RForkLifecycleState.READY
    session.config = replace(session.config, seed_port_base=20000)
    session.identity = replace(runtime.identity, global_rank=30000)
    session.transfer_backend.transfer_session_id = "native-session"
    session.transfer_backend.weight_manifest = {"weight": 1}
    create_server = Mock(return_value=Mock(is_alive=True, port=1234))
    monkeypatch.setattr(runtime.session, "create_rfork_server", create_server)
    monkeypatch.setattr(session, "_run_seed_heartbeat", lambda *args: None)
    session.planner.report_seed_once.return_value = True

    assert session.schedule_seed_service() is True
    join_seed_worker(session, 2)

    assert create_server.call_args.kwargs["port"] == 0
    create_server.return_value.start.assert_called_once_with(session.config.seed_timeout_sec)
    assert session.state is runtime.types.RForkLifecycleState.SERVING
    session.lease_renew_stop_event.set()
