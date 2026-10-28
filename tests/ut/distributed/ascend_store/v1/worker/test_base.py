"""Tests for worker.base."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import BrokenExecutor, wait
from dataclasses import replace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeLoadStartGate,
    begin_step,
    make_backend_spec,
    make_coordinator,
    make_topology,
    make_worker,
    store_one,
)
from tests.ut.distributed.ascend_store.v1.infrastructure.resources_fixtures import (
    RecordingBackend,
    make_caches,
    make_resources,
)
from tests.ut.distributed.ascend_store.v1.worker.layerwise.fixtures import (
    make_gva_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    LookupRequest,
    RangeStoreCommand,
    StateCheckpointSource,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import (
    KVPoolResources,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import base as worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import (
    BulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import worker as bulk_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    KeyRangeLayerwiseWorker,
    key_range,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    TransferSource,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.completion import (
    TransferEvidence,
)


def test_worker_mapping_failure_precedes_registration_and_closes_backend(monkeypatch) -> None:
    backend = RecordingBackend()
    topology, resources = make_resources(backend, physical_layers=(0,))
    caches = make_caches(topology)
    monkeypatch.setattr(bulk_worker, "build_mappings", MagicMock(side_effect=RuntimeError("mapping binding failed")))
    worker = BulkWorker(topology, resources, coordinator=make_coordinator(topology))
    with pytest.raises(RuntimeError, match="mapping binding failed"):
        worker.bind_kv_caches(caches)

    lifecycle = [call[0] for call in backend.calls]
    assert lifecycle == ["backend_close"]
    assert resources.kv_caches is None


def test_bulk_binding_uses_retained_tensors_without_consuming_registration_tables(monkeypatch):
    backend = RecordingBackend()
    topology, resources = make_resources(backend, physical_layers=(0,))
    caches = make_caches(topology)
    prepare = resources.prepare_kv_caches

    def prepare_without_tables(kv_caches):
        prepare(kv_caches)
        return {}

    monkeypatch.setattr(resources, "prepare_kv_caches", prepare_without_tables)
    worker = BulkWorker(topology, resources, coordinator=make_coordinator(topology), store_enabled=False)
    try:
        worker.bind_kv_caches(caches)
        assert resources.kv_caches is caches
        begin_step(worker, load=(LoadCommand("request", 0, 4, ((3,),), (b"a",)),))
        worker.start_load()
        assert not worker.collect_load_result().failed_block_ids
        worker.end_step()
        get_call = next(call for call in backend.calls if call[0] == "get")
        assert get_call[2:] == (((next(iter(caches.values()))[3].data_ptr(),),), ((16,),))
    finally:
        worker.close()
    assert resources.kv_caches is None
    assert [call[0] for call in backend.calls if "region" in call[0]] == ["register_region", "unregister_region"]


def test_invalid_tp_tensor_geometry_fails_before_native_registration() -> None:
    topology = make_topology(physical_layers=(0,), tp_mismatch=True)
    backend = RecordingBackend()
    resources = KVPoolResources(backend, make_backend_spec(), topology.transfer_groups)
    worker = BulkWorker(topology, resources, coordinator=make_coordinator(topology))
    with pytest.raises(RuntimeError, match="unflatten.*sizes.*multiply"):
        worker.bind_kv_caches(make_caches(topology))
    assert backend.calls == [("backend_close",)]
    assert resources.kv_caches is None


@pytest.mark.parametrize("phase", ("naming", "object_layout"))
def test_layerwise_capability_binding_failure_precedes_registration_and_closes_backend(phase, monkeypatch):
    backend = RecordingBackend()
    topology = make_topology(physical_layers=(0,))
    resources = KVPoolResources(backend, make_backend_spec(), topology.groups)
    fail = Mock(side_effect=RuntimeError("capability binding failed"))
    worker = KeyRangeLayerwiseWorker(
        topology,
        resources,
        coordinator=make_coordinator(topology),
        key_builder=fail if phase == "naming" else None,
    )
    if phase == "object_layout":
        monkeypatch.setattr(key_range, "build_key_range_remote", fail)
    with pytest.raises(RuntimeError, match="capability binding failed"):
        worker.bind_kv_caches(make_caches(topology))
    fail.assert_called_once()
    assert backend.calls == [("validate_layerwise_support",), ("backend_close",)]
    assert resources.kv_caches is None
    assert not worker._execution_started


def test_unknown_store_source_keeps_real_registration_and_backend_open() -> None:
    topology = make_topology(physical_layers=(0,))
    backend = RecordingBackend()
    backend.put_result = RuntimeError("put failed after address handoff")
    backend_spec = make_backend_spec(layerwise_access=None)
    resources = KVPoolResources(backend, backend_spec, topology.groups)
    cleanup = Mock()
    resources.bind_execution_cleanup(cleanup)
    worker = BulkWorker(
        topology,
        resources,
        coordinator=make_coordinator(topology, 64),
        source_ready_event_factory=FakeEvent,
    )
    caches = make_caches(topology)
    worker.bind_kv_caches(caches)
    worker.begin_step(
        KVTransferStep(store=StoreCommandBatch((RangeStoreCommand("request", 0, 4, ((7,),), (b"a",), 4, 17),)))
    )
    worker.finish_step()

    with pytest.raises(RuntimeError, match="Store source release is unknown"):
        worker.close()

    assert resources.kv_caches is caches
    assert "unregister_region" not in [call[0] for call in backend.calls]
    assert "backend_close" not in [call[0] for call in backend.calls]
    cleanup.assert_not_called()


class _BlockingStoreBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.copy_started = (threading.Event(), threading.Event())
        self.allow_copy = (threading.Event(), threading.Event())
        self.copied_keys: list[tuple[str, ...]] = []

    def put(self, keys, addresses, sizes):
        index = len(self.copied_keys)
        self.copied_keys.append(tuple(keys))
        self.copy_started[index].set()
        assert self.allow_copy[index].wait(timeout=5)
        return super().put(keys, addresses, sizes)


@pytest.mark.parametrize("route", ("bulk", "async_bulk", "key_range", "gva"))
def test_empty_load_does_not_build_batches_or_submit_background_work(route, monkeypatch) -> None:
    if route == "gva":
        worker, resources, _ = make_gva_worker()
    else:
        worker, resources, _ = make_worker(layerwise=route == "key_range", async_load=route == "async_bulk")

    def unexpected_work(*args, **kwargs):
        pytest.fail("A step without Load commands must not prepare or submit Load work")

    monkeypatch.setattr(worker, "_build_load_batch", unexpected_work)
    monkeypatch.setattr(worker, "_submit_load", unexpected_work)
    try:
        for _ in range(2):
            begin_step(worker)
            worker.start_load()
            assert not worker.collect_load_result().completed_request_ids
            worker.finish_step()
            worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("layerwise,async_load", ((False, False), (False, True), (True, False)))
def test_completed_store_failure_logs_releases_sources_and_allows_next_step(
    layerwise: bool, async_load: bool, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(worker_module, "logger", logging.getLogger(__name__))
    backend = FakeBackend()
    backend.presence = [0]
    if layerwise:
        backend.store_session_copy_result = [-9]
    else:
        backend.put_result = [-9]
    worker, resources, _ = make_worker(backend, layerwise=layerwise, async_load=async_load)
    command = RangeStoreCommand("request", 0, 4, ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    if layerwise:
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
    worker.finish_step()
    worker.fence_previous_store()

    assert "KV cache Store failed for request request (job 17)" in caplog.text
    assert worker.take_released_store_job_ids() == {17}
    assert worker._pending_store_batch is None
    assert worker.lookup(LookupRequest(0, 4, (0,), (b"a",))).available_end_token == 0
    if layerwise:
        assert "batch_commit" not in [call[0] for call in backend.calls]
        assert "batch_revoke" in [call[0] for call in backend.calls]
    worker.end_step()

    backend.put_result = None
    backend.store_session_copy_result = None
    next_command = RangeStoreCommand("next", 0, 4, ((9,),), (b"b",), 4, 18)
    begin_step(worker, store=(next_command,))
    if layerwise:
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
    worker.finish_step()
    worker.fence_previous_store()
    assert worker.take_released_store_job_ids() == {18}
    worker.close()
    assert resources.closed


@pytest.mark.parametrize("released_request_ids", (None, frozenset()), ids=("drain", "nonblocking_poll"))
def test_store_executor_failure_still_blocks_worker_execution(monkeypatch, released_request_ids) -> None:
    worker, resources, _ = make_worker()

    def fail_executor(*_args):
        raise RuntimeError("unexpected Store executor failure")

    monkeypatch.setattr(worker, "_execute_bulk_store", fail_executor)
    command = RangeStoreCommand("request", 0, 4, ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    worker.finish_step()
    pending = worker._pending_store_batch
    assert pending is not None
    _, unfinished = wait((pending.future,), timeout=2)
    assert not unfinished
    with pytest.raises(RuntimeError, match="unexpected Store executor failure"):
        worker.fence_previous_store(released_request_ids)
    assert worker._pending_store_batch is pending
    assert worker.take_released_store_job_ids() == set()
    worker.end_step()
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        worker.begin_step(KVTransferStep())
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        worker.close()
    assert worker._store_executor.stopped
    assert not resources.closed


@pytest.mark.parametrize("leased", (False, True))
def test_bulk_decode_and_other_requests_overlap_store_but_shutdown_retains_sources(leased) -> None:
    backend = _BlockingStoreBackend()
    worker, resources, _ = make_worker(backend)
    first = RangeStoreCommand("first", 0, 4, ((7,),), (b"a",), 4, 17 if leased else None)
    second = RangeStoreCommand("second", 0, 4, ((9,),), (b"b",), 4, 18 if leased else None)
    fenced = threading.Event()
    closed = threading.Event()
    completions = []
    failures = []
    callers = []

    def fence_first():
        try:
            completions.extend(worker.fence_previous_store(frozenset(("first",))))
        except BaseException as error:
            failures.append(error)
        finally:
            fenced.set()

    def close_worker():
        try:
            worker.close()
        except BaseException as error:
            failures.append(error)
        finally:
            closed.set()

    try:
        begin_step(worker, store=(first,))
        worker.finish_step()
        worker.end_step()
        assert backend.copy_started[0].wait(timeout=2)

        # A decode that writes no source blocks must not wait for the unrelated ongoing copy.
        assert worker.fence_previous_store(frozenset()) == ()
        begin_step(worker)
        worker.finish_step()
        worker.end_step()
        assert worker.take_released_store_job_ids() == set()
        assert not backend.allow_copy[0].is_set()

        begin_step(worker, store=(second,))
        worker.finish_step()
        worker.end_step()
        caller = threading.Thread(target=fence_first)
        callers.append(caller)
        caller.start()
        assert not fenced.wait(timeout=0.05)
        backend.allow_copy[0].set()
        assert backend.copy_started[1].wait(timeout=2)
        assert fenced.wait(timeout=2), "Fencing first must not wait for the later second Store"
        assert not failures
        assert [completion.request_id for completion in completions] == ["first"]
        assert worker.take_released_store_job_ids() == ({17} if leased else set())
        assert worker.fence_previous_store(frozenset()) == ()
        assert worker.take_released_store_job_ids() == set()

        caller = threading.Thread(target=close_worker)
        callers.append(caller)
        caller.start()
        assert not closed.wait(timeout=0.05)
        assert not resources.closed
        backend.allow_copy[1].set()
        assert closed.wait(timeout=2)
        assert not failures
        assert worker.take_released_store_job_ids() == ({18} if leased else set())
        assert worker.take_released_store_job_ids() == set()
        assert len(backend.copied_keys) == 2 and backend.copied_keys[0] != backend.copied_keys[1]
        assert worker._pending_store_batch is None
        assert resources.closed
    finally:
        for allowed in backend.allow_copy:
            allowed.set()
        for caller in callers:
            caller.join(timeout=5)
            assert not caller.is_alive()
        worker.close()


def test_released_request_waits_for_its_latest_store_and_recovers_each_earlier_batch() -> None:
    backend = _BlockingStoreBackend()
    worker, resources, _ = make_worker(backend)
    command = RangeStoreCommand("request", 0, 4, ((7, 8),), (b"a", b"b"), 8, 17)
    latest = replace(command, start_token=4, end_token=8, store_job_id=18)
    fenced = threading.Event()
    completions = []
    failures = []

    def fence_request():
        try:
            completions.extend(worker.fence_previous_store(frozenset(("request",))))
        except BaseException as error:
            failures.append(error)
        finally:
            fenced.set()

    caller = threading.Thread(target=fence_request)
    try:
        begin_step(worker, store=(command,))
        worker.finish_step()
        worker.end_step()
        assert backend.copy_started[0].wait(timeout=2)
        begin_step(worker, store=(latest,))
        worker.finish_step()
        worker.end_step()
        caller.start()
        backend.allow_copy[0].set()
        assert backend.copy_started[1].wait(timeout=2)
        assert not fenced.wait(timeout=0.05), "The request still owns its latest, unfinished Store"
        assert not resources.closed
        backend.allow_copy[1].set()
        assert fenced.wait(timeout=2)
        assert not failures
        assert [completion.store_job_id for completion in completions] == [17, 18]
        assert worker.take_released_store_job_ids() == {17, 18}
        assert worker.fence_previous_store(frozenset()) == ()
        assert worker.take_released_store_job_ids() == set()
        assert len(backend.copied_keys) == 2 and backend.copied_keys[0] != backend.copied_keys[1]
    finally:
        for allowed in backend.allow_copy:
            allowed.set()
        if caller.ident is not None:
            caller.join(timeout=5)
            assert not caller.is_alive()
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("failed_initializer", (1, 2), ids=("store", "load"))
def test_cache_binding_closes_partially_started_executors_and_resources(monkeypatch, failed_initializer) -> None:
    backend = FakeBackend()
    initializer_calls = 0

    def fail_initializer() -> None:
        nonlocal initializer_calls
        initializer_calls += 1
        backend.calls.append(("set_device",))
        if initializer_calls == failed_initializer:
            raise RuntimeError("thread initialization failed")

    monkeypatch.setattr(backend, "set_device", fail_initializer)
    worker, resources, _ = make_worker(backend, async_load=True, bind=False)

    with pytest.raises(BrokenExecutor, match="initializer"):
        worker.bind_kv_caches({"cache": object()})

    assert initializer_calls == failed_initializer
    assert worker._store_executor is not None
    assert worker._store_executor.stopped
    assert worker._load_executor is not None
    assert worker._load_executor.stopped
    assert resources.closed


@pytest.mark.parametrize("channel", ("store", "load"))
def test_worker_close_keeps_resources_until_interrupted_shutdown_is_retried(channel, monkeypatch) -> None:
    worker, resources, _ = make_worker(async_load=channel == "load")
    command = RangeStoreCommand("request", 0, 4, ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    worker.finish_step()
    worker.end_step()
    executor = worker._store_executor if channel == "store" else worker._load_executor
    assert executor is not None
    pool = executor._pool
    shutdown = pool.shutdown

    def interrupt_shutdown(*args, **kwargs) -> None:
        raise KeyboardInterrupt("shutdown interrupted")

    try:
        monkeypatch.setattr(pool, "shutdown", interrupt_shutdown)
        with pytest.raises(KeyboardInterrupt, match="shutdown interrupted"):
            worker.close()

        assert worker._pending_store_batch is None
        assert worker.take_released_store_job_ids() == {17}
        assert not executor.stopped
        assert not resources.closed
    finally:
        monkeypatch.setattr(pool, "shutdown", shutdown)
        worker.close()

    assert executor.stopped
    assert resources.closed


def test_worker_owns_active_step_and_async_completion_lifecycle() -> None:
    worker, _, _ = make_worker(async_load=True, store=False)
    command = LoadCommand("request", 0, 4, ((1,),), (b"a",))

    with pytest.raises(RuntimeError, match="has not begun"):
        worker.start_load()
    begin_step(worker, load=(command,))
    with pytest.raises(RuntimeError, match="has not ended"):
        worker.begin_step(KVTransferStep())
    worker.start_load()
    worker.end_step()
    begin_step(worker)
    _, unfinished = wait(worker._pending_loads, timeout=2)
    assert not unfinished
    result = worker.collect_load_result()
    assert result.completed_request_ids == {"request"}
    assert worker.collect_load_result().completed_request_ids == set()
    worker.end_step()
    worker.close()


def test_async_load_failure_drains_pending_work_and_rejects_overlap(monkeypatch) -> None:
    backend = FakeBackend()

    def fail_get(*_args):
        raise RuntimeError("backend get failed")

    monkeypatch.setattr(backend, "get", fail_get)
    worker, _, _ = make_worker(backend, async_load=True, store=False)
    command = LoadCommand("request", 0, 4, ((1,),), (b"a",))
    begin_step(worker, load=(command,))
    worker.start_load()
    worker.end_step()

    begin_step(worker, load=(replace(command, request_id="other"), command))
    with pytest.raises(RuntimeError, match="already has a pending asynchronous Load"):
        worker.start_load()
    assert worker._pending_load_request_ids == {"request"}
    _, unfinished = wait(worker._pending_loads, timeout=2)
    assert not unfinished
    result = worker.collect_load_result()
    assert result.completed_request_ids == {"request"}
    assert result.failed_block_ids == {1}
    assert [(location.group_id, location.block_id) for location in result.failed_locations] == [(0, 1)]
    worker.close()


@pytest.mark.parametrize("route", ("bulk", "bulk_async", "key_range", "gva"))
def test_successful_load_never_constructs_object_evidence_and_preserves_notifications(route, monkeypatch):
    topology = make_topology(physical_layers=(2, 5, 9))
    if route == "gva":
        worker, resources, backend = make_gva_worker(topology=topology)
        worker._start_gate_factory = lambda: FakeLoadStartGate(opened=True)
    else:
        worker, resources, backend = make_worker(
            topology=topology,
            layerwise=route == "key_range",
            async_load=route == "bulk_async",
            store=False,
            start_gate_factory=lambda: FakeLoadStartGate(opened=True),
        )
    commands = (
        LoadCommand("first", 0, 8, ((1, 2),), (b"a", b"b")),
        LoadCommand("second", 0, 4, ((7,),), (b"c",)),
        LoadCommand("empty", 0, 0, ((),), ()),
    )
    if route == "gva":
        batch = worker._build_load_batch(commands)
        backend.objects.update(
            {key: (10000 + index * 1000, 96, True) for index, key in enumerate(batch.selected_keys())}
        )
    source = Mock(side_effect=AssertionError("Successful Load must not construct TransferSource"))
    evidence = Mock(side_effect=AssertionError("Successful Load must not construct TransferEvidence"))
    monkeypatch.setattr(TransferSource, "__init__", source)
    monkeypatch.setattr(TransferEvidence, "__init__", evidence)
    try:
        begin_step(worker, load=commands)
        worker.start_load()
        if route == "bulk_async":
            worker._pending_loads[0].result(timeout=5)
        elif route in ("key_range", "gva"):
            for layer in (2, 2, 5, 5, 9):
                worker.wait_for_layer_load(f"layers.{layer}.group.0")
        result = worker.collect_load_result()
        assert result.completed_request_ids == ({"first", "second", "empty"} if route == "bulk_async" else set())
        assert not result.failed_locations and not result.failed_block_ids and not result.failed_request_ids
        assert not worker.collect_load_result().completed_request_ids
        if route in ("key_range", "gva"):
            copy_name = "copy" if route == "gva" else "batch_copy_get"
            assert sum(call[0] == copy_name for call in backend.calls) == 3
            assert not worker._backend_sessions.has_load_sessions
        source.assert_not_called()
        evidence.assert_not_called()
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("layerwise", (False, True), ids=("bulk", "layerwise"))
@pytest.mark.parametrize("scenario", ("full_hit", "partial_hit", "duplicate", "lookup_error"))
def test_store_admission_copies_only_missing_unique_keys_and_releases_all_jobs(
    layerwise, scenario, monkeypatch, caplog
) -> None:
    backend = FakeBackend()
    presence = {"full_hit": [1, 1], "partial_hit": [1, 0], "duplicate": [0], "lookup_error": [0, 0]}
    backend.presence = presence[scenario]
    if scenario == "lookup_error":

        def fail_lookup(keys):
            backend.calls.append(("exists", tuple(keys)))
            raise RuntimeError("exists failed")

        monkeypatch.setattr(backend, "exists", fail_lookup)
    events = []

    def make_event():
        event = FakeEvent()
        events.append(event)
        return event

    worker, resources, _ = make_worker(
        backend, layerwise=layerwise, requires_exists_before_put=True, source_ready_event_factory=make_event
    )
    first = RangeStoreCommand("first", 0, 4, ((1,),), (b"a",), 4, 17)
    second_hash = b"a" if scenario == "duplicate" else b"b"
    second = RangeStoreCommand("second", 0, 4, ((3,),), (second_hash,), 4, 18)
    try:
        begin_step(worker, store=(first, second))
        if layerwise:
            worker.save_layer("layers.0.group.0")
            worker.save_layer("layers.1.group.0")
        worker.finish_step()
        completions = worker.fence_previous_store()

        if not layerwise:
            assert len(completions) == 2
            assert all(item.evidence.succeeded for item in completions) is (scenario != "lookup_error")
            assert all(item.evidence.source_release_confirmed for item in completions)
        if scenario == "lookup_error":
            assert "KV cache Store failed for request first" in caplog.text
            assert "KV cache Store failed for request second" in caplog.text
        assert worker.take_released_store_job_ids() == {17, 18}
        assert [call[0] for call in backend.calls].count("exists") == 1
        copies = [call for call in backend.calls if call[0] == ("batch_copy_put" if layerwise else "put")]
        if scenario in ("full_hit", "lookup_error"):
            assert copies == []
            assert not any(call[0] in ("batch_put_start", "batch_commit") for call in backend.calls)
            assert all(event.recorded and not event.synchronized for event in events)
        else:
            source_block = 3 if scenario == "partial_hit" else 1
            assert len(copies) == (2 if layerwise else 1)
            assert all(len(call[1]) == 1 for call in copies)
            row = tuple(cache[source_block].data_ptr() for cache in resources.kv_caches.values())
            expected_addresses = [((address,),) for address in row] if layerwise else [(row,)]
            assert [call[2] for call in copies] == expected_addresses
            assert events and all(event.recorded and event.synchronized for event in events)
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("layerwise", (False, True), ids=("bulk", "layerwise"))
def test_store_admission_preserves_missing_rows_in_each_group(layerwise) -> None:
    backend = FakeBackend()
    backend.presence = [1, 0, 0, 1]
    worker, resources, _ = make_worker(
        backend, topology=make_topology(group_ids=(0, 1)), layerwise=layerwise, requires_exists_before_put=True
    )
    command = RangeStoreCommand("request", 0, 8, ((1, 2), (3, 4)), (b"a", b"b"), 8, 17)
    try:
        begin_step(worker, store=(command,))
        if layerwise:
            worker.save_layer("layers.0.group.0")
            worker.save_layer("layers.1.group.0")
        worker.finish_step()
        worker.fence_previous_store()

        candidates = next(call[1] for call in backend.calls if call[0] == "exists")
        selected_keys = (candidates[1], candidates[2])
        copies = [call for call in backend.calls if call[0] == ("batch_copy_put" if layerwise else "put")]
        assert [call[1] for call in copies] == [selected_keys] * (2 if layerwise else 1)
        rows = tuple(
            tuple(resources.kv_caches[f"layers.{layer}.group.{group}"][block].data_ptr() for layer in (0, 1))
            for group, block in ((0, 2), (1, 3))
        )
        expected_addresses = [tuple((row[layer],) for row in rows) for layer in (0, 1)] if layerwise else [rows]
        assert [call[2] for call in copies] == expected_addresses
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize(
    "topology,presence,partition_keys",
    (
        (make_topology(tp_mismatch=True), [1, 0], "@head_or_tp_rank:1@"),
        (make_topology(target_layer_counts=(1, 1)), [0, 1], "@pp_rank:0@"),
    ),
    ids=("tp_head", "pipeline_stage"),
)
def test_store_admission_selects_only_the_missing_effective_key_axis(topology, presence, partition_keys) -> None:
    backend = FakeBackend()
    backend.presence = presence
    worker, resources, _ = make_worker(backend, topology=topology, requires_exists_before_put=True)
    try:
        completion = store_one(worker, RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17))
        candidates = next(call[1] for call in backend.calls if call[0] == "exists")
        put_call = next(call for call in backend.calls if call[0] == "put")
        assert put_call[1] == (candidates[presence.index(0)],)
        assert partition_keys in put_call[1][0]
        (evidence,) = completion.evidence.transfer_evidence
        assert evidence.source.key == put_call[1][0]
        assert (evidence.source.group_id, evidence.source.local_block_id) == (0, 1)
        assert evidence.result_code == 0
        assert worker.take_released_store_job_ids() == {17}
    finally:
        worker.close()
    assert resources.closed


def test_checkpoint_store_uses_exact_state_source_after_admission() -> None:
    base = make_topology(group_ids=(0, 1))
    groups = (
        KVPoolGroupTopology(
            0,
            FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32),
            (KVPoolLayerTopology(0, ("layers.0.attention",)),),
            base.groups[0].key_metadata,
        ),
        KVPoolGroupTopology(
            1,
            MambaSpec(block_size=8, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align"),
            (KVPoolLayerTopology(1, ("layers.1.state",)),),
            base.groups[1].key_metadata,
            uses_align_state=True,
        ),
    )
    topology = replace(
        base,
        config=replace(
            base.config,
            cache_transfer_granularity=8,
            hash_block_size=4,
            align_state_group_ids=frozenset((1,)),
            group_specs=tuple(group.kv_cache_spec for group in groups),
        ),
        groups=groups,
    )
    backend = FakeBackend()
    backend.presence = [1, 0]
    worker, resources, _ = make_worker(backend, topology=topology, requires_exists_before_put=True)
    command = CheckpointStoreCommand("checkpoint", ((3,), (99,)), (b"a",), 0, (StateCheckpointSource(1, 7, 4),), 23)
    try:
        completion = store_one(worker, command)
        put_call = next(call for call in backend.calls if call[0] == "put")
        assert len(put_call[1]) == 1 and "@group:1@" in put_call[1][0]
        assert put_call[2:] == (((resources.kv_caches["layers.1.state"][7].data_ptr(),),), ((32,),))
        (evidence,) = completion.evidence.transfer_evidence
        assert (evidence.source.group_id, evidence.source.local_block_id) == (1, 7)
        assert evidence.result_code == 0
        assert worker.take_released_store_job_ids() == {23}
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("target_pp", (None, (1, 0, 3)), ids=("execution_pp", "consumer_pp"))
def test_lookup_requires_every_remote_partition_even_when_local_keys_are_present(target_pp):
    topology = make_topology(
        group_ids=(1, 3),
        physical_layers=(2, 3),
        pp_size=2,
        pp_rank=1,
        tp_size=2,
        tp_rank=1,
        tp_mismatch=True,
        key_rank_count=4,
        key_slices_per_rank=2,
        dcp_size=2,
        dcp_rank=1,
        target_layer_counts=target_pp,
    )
    worker, _, backend = make_worker(topology=topology, store=False)
    request = LookupRequest(0, 8, (1, 3), (b"a", b"b"))
    pp_size = len(target_pp) if target_pp else topology.config.pp_size
    expected = [
        tuple(
            PoolKey(
                replace(group.key_metadata, pp_rank=pp, dcp_rank=dcp, head_or_tp_rank=head), value.hex()
            ).to_string()
            for pp in range(pp_size)
            for dcp in range(2)
            for head in range(4)
            for value in (b"a", b"b")
        )
        for group in topology.transfer_groups
    ]
    try:
        assert worker.lookup(request).available_end_token == 8
        assert [call[1] for call in backend.calls if call[0] == "exists"] == expected
        # This remote rank is absent from every local Key/Local table.
        assert all((0, 0, 0) not in keys for keys in worker._keys.values())
        for missing_hash_index, available_end in ((1, 4), (0, 0)):
            presence = [1] * len(expected[0])
            presence[missing_hash_index] = 0
            backend.presence = presence
            assert worker.lookup(request).available_end_token == available_end
    finally:
        worker.close()
