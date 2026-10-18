"""Resource ownership enforces drain-before-release independently of Worker policy."""

from unittest.mock import Mock, create_autospec

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import make_worker
from tests.ut.distributed.ascend_store.v1.worker.resource_fixtures import RecordingBackend, make_caches, make_resources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.execution.executor import Executor


def test_executors_must_stop_before_registration_and_backend_release():
    backend = RecordingBackend()
    topology, resources = make_resources(backend)
    executor = create_autospec(Executor, instance=True, spec_set=True)
    executor.stopped = False
    resources.bind_executors((executor,))
    caches = make_caches(topology)
    resources.prepare_kv_caches(caches)
    resources.register_kv_buffers()
    resources.start_executors()
    executor.start.assert_called_once_with()

    with pytest.raises(RuntimeError, match="Executors must stop"):
        resources.close()
    assert resources.kv_caches is caches and not backend.closed
    assert not any(call[0] == "unregister_region" for call in backend.calls)
    executor.close.side_effect = lambda: setattr(executor, "stopped", True)
    resources.stop_executors()
    assert resources.executors_stopped
    resources.close()
    resources.close()
    assert resources.kv_caches is None and backend.closed
    assert sum(call[0] == "backend_close" for call in backend.calls) == 1


def test_failed_drain_attempts_all_executors_and_keeps_resources_for_retry():
    backend = RecordingBackend()
    _, resources = make_resources(backend)
    first_error = RuntimeError("first drain failed")
    first = create_autospec(Executor, instance=True, spec_set=True)
    second = create_autospec(Executor, instance=True, spec_set=True)
    first.stopped = second.stopped = False
    first.close.side_effect = first_error
    second.close.side_effect = RuntimeError("second drain failed")
    resources.bind_executors((first, second))
    with pytest.raises(RuntimeError, match="first drain failed") as caught:
        resources.stop_executors()
    assert caught.value is first_error
    first.close.assert_called_once_with()
    second.close.assert_called_once_with()
    assert not resources.executors_stopped and not backend.closed

    first.close.side_effect = lambda: setattr(first, "stopped", True)
    second.close.side_effect = lambda: setattr(second, "stopped", True)
    resources.stop_executors()
    resources.close()
    assert backend.closed


def test_start_failure_keeps_all_executors_owned_for_cleanup():
    backend = RecordingBackend()
    _, resources = make_resources(backend)
    first = create_autospec(Executor, instance=True, spec_set=True)
    second = create_autospec(Executor, instance=True, spec_set=True)
    first.stopped = second.stopped = False
    second.start.side_effect = RuntimeError("start failed")
    resources.bind_executors((first, second))
    with pytest.raises(RuntimeError, match="start failed"):
        resources.start_executors()
    first.close.side_effect = lambda: setattr(first, "stopped", True)
    second.close.side_effect = lambda: setattr(second, "stopped", True)
    resources.stop_executors()
    resources.close()
    first.close.assert_called_once_with()
    second.close.assert_called_once_with()
    assert backend.closed


def test_executor_binding_cannot_replace_existing_ownership():
    backend = RecordingBackend()
    _, resources = make_resources(backend)
    resources.bind_executors(())
    with pytest.raises(RuntimeError, match="bound once"):
        resources.bind_executors(())
    resources.close()
    with pytest.raises(RuntimeError, match="closed"):
        resources.start_executors()


def test_session_binding_cannot_discard_an_owned_session():
    backend = RecordingBackend()
    _, resources = make_resources(backend)
    owned, replacement = Mock(), Mock()
    resources.bind_backend_sessions(owned)
    with pytest.raises(RuntimeError, match="bound once"):
        resources.bind_backend_sessions(replacement)
    resources.close()
    owned.close.assert_called_once_with()
    replacement.close.assert_not_called()


@pytest.mark.parametrize("layerwise", (False, True), ids=("bulk", "layerwise"))
def test_worker_binds_real_executors_and_preserves_start_and_cleanup_order(layerwise, monkeypatch):
    worker, resources, _ = make_worker(layerwise=layerwise, async_load=True, bind=False)
    events = []

    def record(action, event):
        def call(*args, **kwargs):
            result = action(*args, **kwargs)
            events.append(event)
            return result

        return call

    assert resources._executors == (worker._store_executor, worker._load_executor)
    for name, executor in zip(("store", "load"), resources._executors, strict=True):
        assert isinstance(executor, Executor)
        monkeypatch.setattr(executor, "start", record(executor.start, f"start_{name}"))
        monkeypatch.setattr(executor, "close", record(executor.close, f"stop_{name}"))
    monkeypatch.setattr(worker, "fence_previous_store", record(worker.fence_previous_store, "store_fence"))
    monkeypatch.setattr(worker, "_check_execution_results", record(worker._check_execution_results, "results"))
    monkeypatch.setattr(resources, "register_kv_buffers", record(resources.register_kv_buffers, "registration"))
    monkeypatch.setattr(resources, "close", record(resources.close, "release"))
    try:
        worker.bind_kv_caches({})
        worker.close()
        assert events == [
            "registration",
            "start_store",
            "start_load",
            "store_fence",
            "stop_store",
            "stop_load",
            "results",
            "release",
        ]
        assert resources.executors_stopped and resources.closed
    finally:
        worker.close()
