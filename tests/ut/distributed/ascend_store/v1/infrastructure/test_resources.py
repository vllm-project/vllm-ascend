"""Registration rollback and Worker teardown preserve native memory lifetime."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, create_autospec

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    make_backend_spec,
    make_topology,
    make_worker,
)
from tests.ut.distributed.ascend_store.v1.infrastructure.resources_fixtures import (
    RecordingBackend,
    make_caches,
    make_resources,
    register_caches,
)
from tests.ut.distributed.ascend_store.v1.worker.layerwise.fixtures import (
    make_gva_spec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.sessions import MemcacheSessions
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.execution.executor import Executor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import (
    KVPoolResources,
)


def test_resources_own_exact_shared_storage_region_and_close_in_order() -> None:
    backend = RecordingBackend()
    topology, resources = make_resources(backend)
    caches = make_caches(topology, shared_storage=True)
    storage = next(iter(caches.values())).untyped_storage()

    register_caches(resources, caches)

    assert backend.calls == [("register_region", storage.data_ptr(), storage.nbytes())]
    resources.close()
    resources.close()

    assert backend.calls == [
        ("register_region", storage.data_ptr(), storage.nbytes()),
        ("unregister_region", storage.data_ptr(), storage.nbytes()),
        ("backend_close",),
    ]
    assert resources.kv_caches is None
    assert backend.closed
    with pytest.raises(RuntimeError, match="closed"):
        register_caches(resources, caches)


def test_gva_registration_alignment_does_not_change_mapping_payload() -> None:
    alignment = 2 * 1024 * 1024
    topology = make_topology(physical_layers=(0,))
    backend = RecordingBackend()
    resources = KVPoolResources(backend, make_gva_spec(), topology.transfer_groups, align_shared_storage=True)
    storage = torch.empty(2 * alignment, dtype=torch.uint8)
    aligned_offset = -storage.data_ptr() % alignment
    view_offset = aligned_offset + 64
    cache = storage[view_offset : view_offset + 32].view(8, 4)
    resources.prepare_kv_caches({topology.groups[0].layer_names[0]: cache})
    assert backend.calls == []
    assert resources.kv_caches[topology.groups[0].layer_names[0]] is cache
    resources.register_kv_buffers()
    assert backend.calls == [("register_region", storage.data_ptr() + aligned_offset, 96)]
    resources.close()


@pytest.mark.parametrize("reverse", (False, True))
def test_registration_covers_strided_aliases_without_registering_unrelated_storage(reverse):
    backend = RecordingBackend()
    topology, resources = make_resources(backend)
    storage = torch.empty(200, dtype=torch.int16)
    first = storage.as_strided((3, 2, 2), (25, 8, 2), storage_offset=7)
    second = storage.as_strided((2, 3), (30, 3), storage_offset=80)
    names = topology.groups[0].layer_names
    tensors = (second, first) if reverse else (first, second)
    caches = dict(zip(names, tensors, strict=True))
    register_caches(resources, caches)
    # First view starts at element 7; the second view's last element is 116.
    assert backend.calls == [("register_region", storage.data_ptr() + 14, 220)]
    assert resources.kv_caches is caches
    resources.close()
    assert resources.kv_caches is None


@pytest.mark.parametrize("empty_first", (False, True))
@pytest.mark.parametrize("use_gva", (False, True))
def test_resources_skip_empty_rope_views_in_registration(empty_first, use_gva) -> None:
    backend = RecordingBackend()
    topology = make_topology(physical_layers=(0,))
    resources = KVPoolResources(
        backend,
        make_backend_spec(layerwise_access=LayerwiseAccessKind.GVA if use_gva else None),
        topology.transfer_groups,
    )
    nope_cache = torch.empty((8, 4), dtype=torch.float32)
    rope_cache = torch.empty((8, 4, 0), dtype=torch.float32)
    caches = (rope_cache, nope_cache) if empty_first else (nope_cache, rope_cache)

    register_caches(resources, {topology.groups[0].layer_names[0]: caches})

    address = nope_cache.data_ptr()
    region_size = nope_cache.numel() * nope_cache.element_size()
    assert backend.calls == [("register_region", address, region_size)]
    assert resources.kv_caches[topology.groups[0].layer_names[0]] is caches
    resources.close()
    assert backend.calls[-2:] == [("unregister_region", address, region_size), ("backend_close",)]


def test_registration_failure_rolls_back_successful_prefix_before_backend_close() -> None:
    backend = RecordingBackend(fail_registration_index=1)
    topology, resources = make_resources(backend)
    caches = make_caches(topology)

    with pytest.raises(RuntimeError, match="registration failed at region 1"):
        register_caches(resources, caches)

    assert resources.kv_caches is caches
    assert [call[0] for call in backend.calls] == ["register_region", "register_region", "unregister_region"]
    resources.close()
    assert [call[0] for call in backend.calls][-1] == "backend_close"
    assert resources.kv_caches is None


def test_incomplete_registration_rollback_retains_owner_and_retries_release() -> None:
    backend = RecordingBackend(fail_registration_index=1, fail_unregister_count=1)
    topology, resources = make_resources(backend)
    caches = make_caches(topology)

    with pytest.raises(RuntimeError, match="rollback is incomplete"):
        register_caches(resources, caches)

    assert resources.kv_caches is caches
    assert "backend_close" not in [call[0] for call in backend.calls]
    resources.close()
    assert [call[0] for call in backend.calls][-2:] == ["unregister_region", "backend_close"]
    assert resources.kv_caches is None


def test_backend_close_failure_retains_cache_owner_until_retry() -> None:
    backend = RecordingBackend(fail_close_count=1)
    topology, resources = make_resources(backend, physical_layers=(0,))
    caches = make_caches(topology)
    register_caches(resources, caches)

    with pytest.raises(RuntimeError, match="backend close failed"):
        resources.close()

    assert resources.kv_caches is caches
    assert [call[0] for call in backend.calls].count("unregister_region") == 1
    resources.close()
    assert [call[0] for call in backend.calls].count("unregister_region") == 1
    assert resources.kv_caches is None


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_remote_cleanup_failure_keeps_registration_and_client_owned_until_retry(use_gva, monkeypatch) -> None:
    backend = RecordingBackend()
    topology, resources = make_resources(backend, physical_layers=(0,))
    caches = make_caches(topology)
    if use_gva:
        monkeypatch.setattr(backend, "allocate_gva", lambda keys, sizes: (1000,), raising=False)
        revoke = MagicMock(side_effect=[(-1,), (0,)])
        monkeypatch.setattr(backend, "revoke_gva", revoke, raising=False)
        sessions = MemcacheSessions(backend)
    else:
        revoke = MagicMock(side_effect=[(-8,), (0,)])
        monkeypatch.setattr(backend, "revoke_key_range_store", revoke)
        sessions = backend.create_sessions()
    resources.bind_backend_sessions(sessions)
    register_caches(resources, caches)
    sessions.start_store(["pending"], [64])

    with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
        resources.close()

    assert sessions.pending_store_keys == ("pending",)
    assert resources.kv_caches is caches
    assert not backend.closed
    assert not any(item[0] in ("unregister_region", "backend_close") for item in backend.calls)
    resources.close()
    resources.close()
    assert not sessions.pending_store_keys
    assert [item[0] for item in backend.calls][-2:] == ["unregister_region", "backend_close"]
    assert resources.kv_caches is None
    assert backend.closed
    assert revoke.call_count == 2


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


def test_execution_cleanup_waits_for_drain_and_sessions_and_retains_failure_for_retry():
    backend = RecordingBackend()
    topology, resources = make_resources(backend, physical_layers=(0,))
    executor = create_autospec(Executor, instance=True, spec_set=True)
    executor.stopped = False
    resources.bind_executors((executor,))
    sessions = Mock()
    sessions.close.side_effect = [RuntimeError("session cleanup failed"), None, None]
    resources.bind_backend_sessions(sessions)
    cleanup = Mock(side_effect=[RuntimeError("route cleanup failed"), None])
    resources.bind_execution_cleanup(cleanup)
    caches = make_caches(topology)
    register_caches(resources, caches)

    with pytest.raises(RuntimeError, match="Executors must stop"):
        resources.close()
    sessions.close.assert_not_called()
    cleanup.assert_not_called()
    executor.stopped = True
    with pytest.raises(RuntimeError, match="session cleanup failed"):
        resources.close()
    cleanup.assert_not_called()
    with pytest.raises(RuntimeError, match="route cleanup failed"):
        resources.close()
    assert resources.kv_caches is caches and not backend.closed
    assert [call[0] for call in backend.calls] == ["register_region"]
    resources.close()
    resources.close()
    assert cleanup.call_count == 2
    assert resources.kv_caches is None and backend.closed
    assert [call[0] for call in backend.calls] == ["register_region", "unregister_region", "backend_close"]


@pytest.mark.parametrize("state", ("already_bound", "caches_bound", "closed"))
def test_execution_cleanup_binding_cannot_replace_or_acquire_late(state):
    backend = RecordingBackend()
    topology, resources = make_resources(backend)
    owned, replacement = Mock(), Mock()
    if state == "already_bound":
        resources.bind_execution_cleanup(owned)
    elif state == "caches_bound":
        register_caches(resources, make_caches(topology))
    else:
        resources.close()
    with pytest.raises(RuntimeError, match="bound once before KV caches"):
        resources.bind_execution_cleanup(replacement)
    resources.close()
    replacement.assert_not_called()
    assert owned.call_count == (1 if state == "already_bound" else 0)


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
