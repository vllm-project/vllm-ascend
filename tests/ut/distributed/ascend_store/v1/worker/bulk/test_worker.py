"""Bulk execution modes preserve notification, thread ownership and independent I/O lanes."""

from __future__ import annotations

import threading
from concurrent.futures import BrokenExecutor, CancelledError, Future, wait
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeResources,
    begin_step,
    make_backend_spec,
    make_coordinator,
    make_topology,
    make_worker,
    store_one,
)
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import (
    FakeBlocks,
    make_config,
    make_output,
    make_request,
    new_request_data,
)
from tests.ut.distributed.ascend_store.v1.worker.bulk.fixtures import (
    TensorBytesBackend,
    make_align_state_topology,
    make_multi_spec_caches,
    make_sparse_group_topology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import factory as factory_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration import (
    resolve as configuration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.resolve import (
    resolve_group_layers,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.execution.executor import Executor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import (
    CheckpointStoreCommand,
    LoadCommand,
    LookupRequest,
    RangeStoreCommand,
    StateCheckpointSource,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import (
    KVPoolResources,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import (
    AsynchronousBulkScheduler,
    SynchronousBulkScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import (
    BulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.completion import (
    LoadCompletion,
    StoreCompletion,
    StoreEvidence,
)


@pytest.mark.parametrize("layout", ("NHD", "HND", "LBHNC"))
def test_worker_preserves_configured_cache_layout(layout) -> None:
    topology = make_topology()
    backend = FakeBackend()
    resources = FakeResources(backend, SimpleNamespace(), topology)
    worker = BulkWorker(topology, resources, coordinator=make_coordinator(topology), kv_cache_layout=layout)
    assert worker._kv_cache_layout == layout
    worker.close()


def test_bulk_store_registers_target_and_mtp_entries_and_deduplicates_confirmed_keys() -> None:
    base = make_topology(physical_layers=(0,))
    names = ("model.layers.0.attention", "mtp.layers.0.attention", "mtp.layers.1.attention")
    group = replace(base.groups[0], layers=resolve_group_layers(list(names), base_layer_count=2))
    topology = replace(base, groups=(group,))
    backend = TensorBytesBackend()
    resources = KVPoolResources(backend, make_backend_spec(requires_exists_before_put=True), topology.transfer_groups)
    worker = BulkWorker(
        topology,
        resources,
        coordinator=make_coordinator(topology, 64),
        source_ready_event_factory=FakeEvent,
    )
    caches = {name: torch.zeros((8, 4, 1, 2), dtype=torch.float32) for name in names}
    try:
        worker.bind_kv_caches(caches)
        assert backend.registered_regions == [
            (cache.data_ptr(), cache.numel() * cache.element_size()) for cache in caches.values()
        ]
        command = RangeStoreCommand("target-and-draft", 0, 4, ((1,),), (b"a",), 4, 17)
        begin_step(worker, store=(command,))
        # Bulk publication occurs at finalization, after both target and draft cache writes.
        for index, name in enumerate(names):
            caches[name][1].fill_(index + 1)
        assert not any(call[0] == "put" for call in backend.calls)
        worker.finish_step()
        (completion,) = worker.fence_previous_store()
        worker.end_step()
        assert completion.evidence.succeeded and completion.evidence.source_release_confirmed
        (key,) = backend.objects
        expected_payload = b"".join(caches[name][1].view(torch.uint8).numpy().tobytes() for name in names)
        assert backend.objects[key] == expected_payload
        put_call = next(call for call in backend.calls if call[0] == "put")
        assert put_call[2] == (tuple(caches[name][1].data_ptr() for name in names),)
        assert put_call[3] == ((32, 32, 32),)
        assert worker.take_released_store_job_ids() == {17}

        for name in names:
            caches[name][2].copy_(caches[name][1])
        duplicate = store_one(
            worker, replace(command, request_id="duplicate", block_ids_by_group=((2,),), store_job_id=18)
        )
        assert duplicate.evidence.succeeded and duplicate.evidence.source_release_confirmed
        assert len([call for call in backend.calls if call[0] == "put"]) == 1
        assert backend.objects[key] == expected_payload
        assert worker.take_released_store_job_ids() == {18}
    finally:
        worker.close()
    assert backend.closed and resources.kv_caches is None


@pytest.mark.parametrize("load_async", (False, True), ids=("sync", "async"))
def test_memcache_multi_spec_store_lookup_load_preserves_payloads_and_private_state(load_async, monkeypatch) -> None:
    cache_config, caches = make_multi_spec_caches()
    backend = TensorBytesBackend()
    backend_spec = BackendSpec("memcache", lambda *_args, **_kwargs: backend, LayerwiseAccessKind.GVA, True)
    monkeypatch.setattr(factory_module, "resolve_backend_spec", lambda _name: backend_spec)
    monkeypatch.setattr(configuration, "_kvpp_size", lambda _config: 1)
    monkeypatch.setattr(configuration, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, ranks=[0]))
    monkeypatch.setattr(configuration, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(configuration, "resolve_dcp_kv_cache_spec", lambda spec, _size: spec)
    monkeypatch.setattr(configuration.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_args: (32, 32))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={"backend": "memcache", "load_async": load_async},
        ),
        model_config=SimpleNamespace(
            max_model_len=64,
            model="deepseek_v41",
            use_mla=True,
            hf_text_config=SimpleNamespace(num_hidden_layers=4, model_type="deepseek_v41"),
            get_total_num_hidden_layers=lambda: 4,
            get_total_num_kv_heads=lambda: 1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
    )
    source_blocks = {0: 1, 2: 3}
    destination_blocks = {0: 5, 2: 6}
    expected = {}
    for group_id, block_id in source_blocks.items():
        for index, name in enumerate(sorted(cache_config.kv_cache_groups[group_id].layer_names)):
            block = caches[name][block_id]
            values = (
                torch.arange(block.numel(), dtype=block.dtype).reshape(block.shape)
                + (group_id + 1) * 1000
                + index * 100
            )
            block.copy_(values)
            expected[name] = values
    private_name = "model.layers.2.compressor.state_cache"
    private_block = 4
    caches[private_name][private_block].fill_(99)
    private_state = caches[private_name][private_block].clone()

    worker = factory_module.create_kv_pool_worker(config, cache_config)
    try:
        worker.bind_kv_caches(caches)
        assert len(backend.registered_regions) == 2
        command = RangeStoreCommand("producer", 0, 32, ((1,), (private_block,), (3,)), (b"a",), 32, 17)
        completion = store_one(worker, command)
        assert completion.evidence.succeeded
        assert [
            (item.source.group_id, item.source.local_block_id) for item in completion.evidence.transfer_evidence
        ] == [
            (0, 1),
            (2, 3),
        ]
        assert worker.take_released_store_job_ids() == {17}
        for group_id in source_blocks:
            (key,) = [key for key in backend.objects if f"@group:{group_id}@" in key]
            names = sorted(cache_config.kv_cache_groups[group_id].layer_names)
            payload = b"".join(expected[name].contiguous().view(torch.uint8).numpy().tobytes() for name in names)
            assert backend.objects[key] == payload
        assert len(backend.objects) == 2
        assert worker.lookup(LookupRequest(0, 32, (0, 2), (b"a",))).available_end_token == 32
        for group_id, block_id in destination_blocks.items():
            for name in cache_config.kv_cache_groups[group_id].layer_names:
                caches[name][block_id].zero_()
        begin_step(worker, load=(LoadCommand("consumer", 0, 32, ((5,), (private_block,), (6,)), (b"a",)),))
        worker.start_load()
        worker.close()
        result = worker.collect_load_result()
        if load_async:
            assert result.completed_request_ids == {"consumer"}
        assert not result.failed_locations
        for group_id, block_id in destination_blocks.items():
            for name in cache_config.kv_cache_groups[group_id].layer_names:
                torch.testing.assert_close(caches[name][block_id], expected[name])
        torch.testing.assert_close(caches[private_name][private_block], private_state)
        worker.end_step()
    finally:
        worker.close()
    assert backend.closed


@pytest.mark.parametrize("load_async", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize("store_enabled", (False, True), ids=("no_store", "store"))
def test_bulk_constructor_selects_only_the_required_executors(load_async, store_enabled):
    worker, resources, _ = make_worker(async_load=load_async, store=store_enabled, bind=False)
    try:
        assert type(worker) is BulkWorker
        assert isinstance(worker._load_executor, Executor) is load_async
        assert isinstance(worker._store_executor, Executor) is store_enabled
        if load_async and store_enabled:
            assert worker._load_executor is not worker._store_executor
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("load_async", (False, True), ids=("sync", "async"))
def test_bulk_load_uses_the_selected_thread_and_notification_mode(load_async, monkeypatch):
    backend = FakeBackend()
    caller_id = threading.get_ident()
    initialized_on: list[int] = []
    executed_on: list[int] = []
    monkeypatch.setattr(backend, "set_device", lambda: initialized_on.append(threading.get_ident()))

    def get(keys, addresses, sizes):
        executed_on.append(threading.get_ident())
        return [0] * len(keys)

    monkeypatch.setattr(backend, "get", get)
    worker, resources, _ = make_worker(backend, async_load=load_async, store=False)
    command = LoadCommand("request", 0, 4, ((7,),), (b"a",))
    try:
        begin_step(worker, load=(command,))
        worker.start_load()
        if load_async:
            assert len(worker._pending_loads) == 1
            worker._pending_loads[0].result(timeout=2)
            assert executed_on == initialized_on
            assert executed_on != [caller_id]
            assert worker._pending_load_request_ids == {"request"}
        else:
            assert executed_on == [caller_id]
            assert not initialized_on
            assert not worker._pending_loads
            assert not worker._pending_load_request_ids
        result = worker.collect_load_result()
        assert result.completed_request_ids == ({"request"} if load_async else set())
        assert not result.failed_locations
        assert not worker.collect_load_result().completed_request_ids
        assert not worker._pending_load_request_ids
    finally:
        worker.close()
    assert resources.closed


def test_bulk_load_can_complete_while_store_is_blocked(monkeypatch):
    backend = FakeBackend()
    store_started = threading.Event()
    release_store = threading.Event()
    store_threads: list[int] = []
    load_threads: list[int] = []

    def put(keys, addresses, sizes):
        store_threads.append(threading.get_ident())
        store_started.set()
        assert release_store.wait(timeout=5)
        return [0] * len(keys)

    def get(keys, addresses, sizes):
        load_threads.append(threading.get_ident())
        return [0] * len(keys)

    monkeypatch.setattr(backend, "put", put)
    monkeypatch.setattr(backend, "get", get)
    worker, resources, _ = make_worker(backend, async_load=True)
    store = RangeStoreCommand("store", 0, 4, ((7,),), (b"a",), 4, 17)
    load = LoadCommand("load", 0, 4, ((9,),), (b"b",))
    try:
        begin_step(worker, store=(store,))
        worker.finish_step()
        worker.end_step()
        assert store_started.wait(timeout=2)
        begin_step(worker, load=(load,))
        worker.start_load()
        worker._pending_loads[0].result(timeout=2)
        assert worker.collect_load_result().completed_request_ids == {"load"}
        assert not release_store.is_set()
        assert worker.fence_previous_store(frozenset()) == ()
        assert not worker.take_released_store_job_ids()
        assert store_threads[0] != load_threads[0]
    finally:
        release_store.set()
        worker.close()
    assert resources.closed
    assert worker.take_released_store_job_ids() == {17}


def test_bulk_async_load_submit_failure_rolls_back_request_ownership():
    worker, resources, _ = make_worker(async_load=True, store=False)
    command = LoadCommand("request", 0, 4, ((7,),), (b"a",))
    try:
        worker._load_executor.close()
        begin_step(worker, load=(command,))
        with pytest.raises(RuntimeError, match="shutdown"):
            worker.start_load()
        assert not worker._pending_load_request_ids
        assert not worker._pending_loads
        assert not worker._load_batches
    finally:
        worker.close()
    assert resources.closed


def test_bulk_store_keeps_batch_identity_with_the_original_future_result(monkeypatch):
    source_ready = FakeEvent()
    worker, resources, _ = make_worker(source_ready_event_factory=lambda: source_ready)
    first = RangeStoreCommand("request", 0, 4, ((7,),), (b"a",), 4, 17)
    unleased = RangeStoreCommand("unleased", 0, 4, ((9,),), (b"b",), 4)
    completions = (StoreCompletion("request", StoreEvidence((), True, True), 17),)
    observed: list[tuple[tuple[RangeStoreCommand, ...], FakeEvent]] = []

    def store(commands, source_ready_event):
        observed.append((commands, source_ready_event))
        return completions

    monkeypatch.setattr(worker, "_execute_bulk_store", store)
    try:
        begin_step(worker, store=(first, first, unleased))
        worker.finish_step()
        pending = worker._pending_store_batch
        assert pending is not None
        assert isinstance(pending.future, Future)
        assert pending.future.result(timeout=2) is completions
        assert pending.request_ids == {"request", "unleased"}
        assert pending.store_job_ids == {17}
        assert observed[0][0] == (first, first, unleased)
        assert observed[0][1] is source_ready
        assert source_ready.recorded
        assert worker.fence_previous_store() == completions
        assert worker.take_released_store_job_ids() == {17}
        assert worker.fence_previous_store() == ()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("observe_before_close", (False, True), ids=("close", "collect_then_close"))
def test_bulk_async_load_business_failure_remains_deliverable_across_shutdown(monkeypatch, observe_before_close):
    worker, resources, _ = make_worker(async_load=True, store=False)
    error = RuntimeError("unexpected Load operation failure")

    def fail(batch):
        raise error

    monkeypatch.setattr(worker, "_execute_bulk_load", fail)
    command = LoadCommand("request", 0, 4, ((7,),), (b"a",))
    begin_step(worker, load=(command,))
    worker.start_load()
    _, unfinished = wait(worker._pending_loads, timeout=2)
    assert not unfinished
    if observe_before_close:
        result = worker.collect_load_result()
        assert result.completed_request_ids == {"request"}
        assert result.failed_block_ids == {7}
        assert not worker._pending_load_request_ids
    worker.close()
    assert worker._load_executor.stopped
    assert resources.closed
    result = worker.collect_load_result()
    assert result.completed_request_ids == (set() if observe_before_close else {"request"})
    assert result.failed_block_ids == (set() if observe_before_close else {7})
    assert not worker._pending_load_request_ids
    assert not worker._pending_loads
    assert not worker._load_batches
    assert not worker.collect_load_result().completed_request_ids


@pytest.mark.parametrize("failed_batch_index", (0, 1, 2), ids=("first", "middle", "last"))
def test_bulk_async_load_mixed_results_are_consumed_independently(monkeypatch, caplog, failed_batch_index):
    worker, resources, _ = make_worker(async_load=True, store=False)
    error = RuntimeError("unexpected Load operation failure")
    executed: list[str] = []
    accepted: list[str] = []
    execute_load = worker._execute_bulk_load
    accept_completion = worker._accept_load_completion
    request_ids = ("first", "middle", "last")

    def load(batch):
        request_id = batch.request_ids[0]
        executed.append(request_id)
        if request_id == request_ids[failed_batch_index]:
            raise error
        return execute_load(batch)

    def accept(completion):
        accepted.append(completion.request_id)
        accept_completion(completion)

    monkeypatch.setattr(worker, "_execute_bulk_load", load)
    monkeypatch.setattr(worker, "_accept_load_completion", accept)
    try:
        for index, request_id in enumerate(request_ids):
            command = LoadCommand(request_id, 0, 4, ((7 + index,),), (request_id.encode(),))
            begin_step(worker, load=(command,))
            worker.start_load()
            worker.end_step()
        pending = tuple(worker._pending_loads)
        _, unfinished = wait(pending, timeout=2)
        assert not unfinished
        assert executed == list(request_ids)
        for index, future in enumerate(pending):
            if index == failed_batch_index:
                assert future.exception() is error
            else:
                assert tuple(completion.request_id for completion in future.result()) == (request_ids[index],)

        begin_step(worker)
        result = worker.collect_load_result()
        assert result.completed_request_ids == set(request_ids)
        assert result.failed_block_ids == {7 + failed_batch_index}
        assert [(location.group_id, location.block_id) for location in result.failed_locations] == [
            (0, 7 + failed_batch_index)
        ]
        assert accepted == list(request_ids)
        assert not worker._pending_loads
        assert not worker._pending_load_request_ids
        assert not worker._load_batches
        errors = [record for record in caplog.records if record.exc_info and record.exc_info[1] is error]
        assert len(errors) == 1
        assert str((request_ids[failed_batch_index],)) in errors[0].getMessage()
        for _ in range(2):
            result = worker.collect_load_result()
            assert not result.completed_request_ids
            assert not result.failed_block_ids
            assert not result.failed_locations

        # The failed request no longer owns a pending slot and can submit another batch.
        worker.end_step()
        monkeypatch.setattr(worker, "_execute_bulk_load", execute_load)
        retry_id = request_ids[failed_batch_index]
        begin_step(worker, load=(LoadCommand(retry_id, 0, 4, ((7,),), (b"retry",)),))
        worker.start_load()
        _, unfinished = wait(worker._pending_loads, timeout=2)
        assert not unfinished
        result = worker.collect_load_result()
        assert result.completed_request_ids == {retry_id}
        assert not result.failed_block_ids
    finally:
        worker.close()
    assert worker._load_executor.stopped
    assert resources.closed
    assert not worker._pending_loads
    assert not worker._pending_load_request_ids
    assert not worker._load_batches


def test_bulk_async_load_business_failure_covers_every_request_in_its_batch(monkeypatch):
    worker, resources, _ = make_worker(async_load=True, store=False)

    def fail(batch):
        raise RuntimeError("business operation failed")

    monkeypatch.setattr(worker, "_execute_bulk_load", fail)
    commands = (LoadCommand("first", 0, 8, ((1, 2),), (b"a", b"b")), LoadCommand("second", 0, 4, ((3,),), (b"c",)))
    try:
        begin_step(worker, load=commands)
        worker.start_load()
        _, unfinished = wait(worker._pending_loads, timeout=2)
        assert not unfinished
        result = worker.collect_load_result()
        assert result.completed_request_ids == {"first", "second"}
        assert result.failed_block_ids == {1, 2, 3}
        assert {(location.group_id, location.block_id) for location in result.failed_locations} == {
            (0, 1),
            (0, 2),
            (0, 3),
        }
        assert not worker._pending_load_request_ids
        assert not worker._pending_loads
        assert not worker._load_batches
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("cancelled", (False, True), ids=("broken_executor", "cancelled_future"))
def test_bulk_async_load_execution_error_leaves_later_batches_queued(monkeypatch, cancelled):
    worker, resources, _ = make_worker(async_load=True, store=False)
    first: Future[tuple[LoadCompletion, ...]] = Future()
    failed: Future[tuple[LoadCompletion, ...]] = Future()
    last: Future[tuple[LoadCompletion, ...]] = Future()
    error = BrokenExecutor("executor unavailable")
    first.set_result((LoadCompletion("first", ()),))
    if cancelled:
        failed.cancel()
    else:
        failed.set_exception(error)
    last.set_result((LoadCompletion("last", ()),))
    submitted = iter((first, failed, last))
    monkeypatch.setattr(worker._load_executor, "submit", lambda *_args: next(submitted))
    try:
        for index, request_id in enumerate(("first", "middle", "last")):
            command = LoadCommand(request_id, 0, 4, ((index + 1,),), (request_id.encode(),))
            begin_step(worker, load=(command,))
            worker.start_load()
            worker.end_step()
        begin_step(worker)
        with pytest.raises(CancelledError if cancelled else BrokenExecutor) as raised:
            worker.collect_load_result()
        if not cancelled:
            assert raised.value is error
        assert tuple(worker._pending_loads) == (last,)
        assert tuple(worker._load_batches) == (last,)
        assert worker._load_batches[last].request_ids == ("last",)
        assert "last" in worker._pending_load_request_ids
    finally:
        worker.close()
    assert resources.closed


def test_bulk_async_load_consumes_finished_batches_once_and_preserves_late_notifications(monkeypatch):
    backend = FakeBackend()
    second_started = threading.Event()
    release_second = threading.Event()
    get_calls = 0

    def get(keys, addresses, sizes):
        nonlocal get_calls
        get_calls += 1
        if get_calls == 2:
            second_started.set()
            assert release_second.wait(timeout=5)
        return [0] * len(keys)

    monkeypatch.setattr(backend, "get", get)
    worker, resources, _ = make_worker(backend, async_load=True, store=False)
    first = LoadCommand("first", 0, 4, ((7,),), (b"a",))
    companion = LoadCommand("companion", 0, 4, ((8,),), (b"b",))
    second = LoadCommand("second", 0, 4, ((9,),), (b"c",))
    try:
        begin_step(worker, load=(first, companion))
        worker.start_load()
        worker._pending_loads[0].result(timeout=2)
        worker.end_step()
        begin_step(worker, load=(second,))
        worker.start_load()
        assert second_started.wait(timeout=2)

        completed = worker.collect_load_result()
        assert completed.completed_request_ids == {"first", "companion"}
        assert not completed.failed_locations
        assert worker._pending_load_request_ids == {"second"}
        assert len(worker._pending_loads) == 1
        assert not worker.collect_load_result().completed_request_ids
        worker.end_step()
        begin_step(worker)
        release_second.set()
        worker.close()

        assert resources.closed
        completed = worker.collect_load_result()
        assert completed.completed_request_ids == {"second"}
        assert not completed.failed_locations
        assert not worker.collect_load_result().completed_request_ids
        assert not worker._pending_load_request_ids
        assert not worker._pending_loads
    finally:
        release_second.set()
        worker.close()


@pytest.mark.parametrize(
    ("topology", "expected_ranks", "expected_offsets", "expected_sizes"),
    (
        (
            make_topology(
                physical_layers=(0,),
                tp_mismatch=True,
                tp_rank=1,
                tp_size=2,
                key_rank_count=4,
                key_slices_per_rank=2,
            ),
            (2, 3),
            ((0, 8, 16, 24), (4, 12, 20, 28)),
            ((4, 4, 4, 4), (4, 4, 4, 4)),
        ),
        (
            make_topology(
                physical_layers=(0,),
                tp_mismatch=True,
                tp_rank=2,
                tp_size=4,
                key_rank_count=4,
                key_slices_per_rank=1,
            ),
            (2,),
            ((0,),),
            ((32,),),
        ),
    ),
)
def test_tp_mismatch_bulk_binds_effective_keys_to_exact_local_slices(
    topology,
    expected_ranks,
    expected_offsets,
    expected_sizes,
) -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=topology)

    completion = store_one(worker, RangeStoreCommand("request", 0, 3, ((2,),), (b"a",), 3, 17))

    put_call = next(call for call in backend.calls if call[0] == "put")
    assert tuple(f"@head_or_tp_rank:{rank}@" in key for key, rank in zip(put_call[1], expected_ranks, strict=True)) == (
        True,
    ) * len(expected_ranks)
    block_base = resources.kv_caches["layers.0.group.0"][2].data_ptr()
    assert put_call[2] == tuple(tuple(block_base + offset for offset in row) for row in expected_offsets)
    assert put_call[3] == expected_sizes
    assert [item.source.local_block_id for item in completion.evidence.transfer_evidence] == [2] * len(expected_ranks)

    worker.close()
    assert resources.closed


def test_tp_mismatch_bulk_preserves_axis_result_provenance() -> None:
    topology = make_topology(
        physical_layers=(0,),
        tp_mismatch=True,
        tp_rank=1,
        tp_size=2,
        key_rank_count=4,
        key_slices_per_rank=2,
    )
    backend = FakeBackend()
    backend.get_result = [0, -1]
    worker, resources, _ = make_worker(backend, topology=topology, store=False)
    command = LoadCommand("request", 0, 3, ((2,),), (b"a",))

    batch = worker._build_load_batch((command,))
    (completion,) = worker._execute_bulk_load(batch)

    assert [item.result_code for item in completion.failures] == [-1]
    assert [item.source.local_block_id for item in completion.failures] == [2]
    assert "@head_or_tp_rank:3@" in completion.failures[0].source.key

    worker.close()
    assert resources.closed


def _make_tensor_tp_worker(backend, layout, tp_size, tp_rank, *, total_heads, populate_cache, padded_kernel_blocks):
    cache_block_size = 4
    num_cache_blocks = 8
    local_heads = total_heads // tp_size
    topology = make_topology(
        physical_layers=(0, 1),
        tp_mismatch=True,
        tp_rank=tp_rank,
        tp_size=tp_size,
        key_rank_count=2,
        key_slices_per_rank=2 // tp_size,
    )
    spec = replace(topology.groups[0].kv_cache_spec, num_kv_heads=local_heads, head_size=2, head_size_v=3)
    topology = replace(topology, groups=(replace(topology.groups[0], kv_cache_spec=spec),))
    head_major = layout in ("HND", "LBHNC")
    kernel_block_size = 2 if padded_kernel_blocks else cache_block_size
    block_scale = cache_block_size // kernel_block_size
    caches = {}
    expected = {}
    for layer_index, name in enumerate(topology.groups[0].layer_names):
        entries = []
        expected_entries = []
        for entry_index, head_dim in enumerate((2, 3)):
            shape = (
                (num_cache_blocks * block_scale, local_heads, kernel_block_size, head_dim)
                if head_major
                else (num_cache_blocks * block_scale, kernel_block_size, local_heads, head_dim)
            )
            cache = torch.zeros(shape, dtype=torch.float32)
            if padded_kernel_blocks:
                kernel_elements = cache[0].numel()
                storage = torch.zeros(shape[0] * (kernel_elements + 1), dtype=cache.dtype)
                cache = torch.as_strided(storage, shape, (kernel_elements + 1, *cache.stride()[1:]))
            values = (
                layer_index * 10_000
                + entry_index * 1_000
                + torch.arange(cache_block_size, dtype=torch.float32)[:, None, None] * 100
                + (tp_rank * local_heads + torch.arange(local_heads, dtype=torch.float32))[None, :, None] * 10
                + torch.arange(head_dim, dtype=torch.float32)[None, None, :]
            )
            if populate_cache:
                for kernel_index in range(block_scale):
                    kernel_values = values[kernel_index * kernel_block_size : (kernel_index + 1) * kernel_block_size]
                    cache[7 * block_scale + kernel_index].copy_(
                        kernel_values.permute(1, 0, 2) if head_major else kernel_values
                    )
            entries.append(cache)
            expected_entries.append(values)
        caches[name] = tuple(entries)
        expected[name] = tuple(expected_entries)
    resources = KVPoolResources(backend, make_backend_spec(layerwise_access=None), topology.transfer_groups)
    worker = BulkWorker(
        topology,
        resources,
        coordinator=make_coordinator(topology, 64),
        kv_cache_layout=layout,
        source_ready_event_factory=FakeEvent,
    )
    worker.bind_kv_caches(caches)
    return worker, caches, expected


@pytest.mark.parametrize("layout", ("NHD", "HND", "LBHNC"))
@pytest.mark.parametrize("producer_tp_size", (1, 2))
@pytest.mark.parametrize("padded_kernel_blocks", (False, True))
@pytest.mark.parametrize("total_heads", (2, 4))
def test_tp_mismatch_roundtrip_preserves_tensor_values_in_each_layout(
    layout, producer_tp_size, padded_kernel_blocks, total_heads
) -> None:
    backend = TensorBytesBackend()
    workers = []
    expected_objects = {}
    try:
        for rank in range(producer_tp_size):
            worker, _caches, expected = _make_tensor_tp_worker(
                backend,
                layout,
                producer_tp_size,
                rank,
                total_heads=total_heads,
                populate_cache=True,
                padded_kernel_blocks=padded_kernel_blocks,
            )
            workers.append(worker)
            completion = store_one(worker, RangeStoreCommand("producer", 0, 4, ((7,),), (b"a",), 4, 17))
            assert completion.evidence.succeeded
            slice_count = 2 // producer_tp_size
            kernel_tokens = 2 if padded_kernel_blocks else 4
            heads_per_key = total_heads // 2
            for slice_index in range(slice_count):
                payload_parts = []
                for entries in expected.values():
                    for values in entries:
                        # Each key must match the corresponding higher-TP rank's packed Bulk object.
                        head_slice = values[:, slice_index * heads_per_key : (slice_index + 1) * heads_per_key]
                        for start in range(0, len(head_slice), kernel_tokens):
                            kernel = head_slice[start : start + kernel_tokens]
                            if layout in ("HND", "LBHNC"):
                                kernel = kernel.permute(1, 0, 2)
                            payload_parts.append(kernel.contiguous().numpy().tobytes())
                expected_objects[rank * slice_count + slice_index] = b"".join(payload_parts)
        assert len(backend.objects) == 2
        assert [backend.objects[key] for key in sorted(backend.objects)] == [
            expected_objects[rank] for rank in range(2)
        ]

        consumer_tp_size = 3 - producer_tp_size
        for rank in range(consumer_tp_size):
            worker, caches, expected = _make_tensor_tp_worker(
                backend,
                layout,
                consumer_tp_size,
                rank,
                total_heads=total_heads,
                populate_cache=False,
                padded_kernel_blocks=padded_kernel_blocks,
            )
            workers.append(worker)
            begin_step(worker, load=(LoadCommand("consumer", 0, 4, ((7,),), (b"a",)),))
            worker.start_load()
            assert not worker.collect_load_result().failed_block_ids
            worker.end_step()
            block_scale = 2 if padded_kernel_blocks else 1
            for name, entries in caches.items():
                for cache, values in zip(entries, expected[name], strict=True):
                    block = cache[7 * block_scale : 8 * block_scale]
                    if layout in ("HND", "LBHNC"):
                        block = block.permute(0, 2, 1, 3)
                    torch.testing.assert_close(block.reshape(values.shape), values)
    finally:
        for worker in workers:
            worker.close()


@pytest.mark.parametrize(
    "retention_interval,stored_swa_boundaries,middle_end,fallback_end",
    ((None, (16, 32, 48, 64, 80, 96), 80, 64), (0, (80, 96), 0, 0), (64, (48, 64, 80, 96), 80, 64)),
)
def test_bulk_lookup_recovers_sparse_retention_objects_and_loads_tail(
    retention_interval, stored_swa_boundaries, middle_end, fallback_end
) -> None:
    topology = make_sparse_group_topology(block_size=16, sliding_window=33)
    backend = TensorBytesBackend()
    resources = KVPoolResources(backend, make_backend_spec(requires_exists_before_put=True), topology.transfer_groups)
    worker = BulkWorker(
        topology,
        resources,
        coordinator=make_coordinator(topology, 256, retention_interval=retention_interval),
        source_ready_event_factory=FakeEvent,
    )
    block_hashes = tuple(bytes([index]) for index in range(1, 9))
    cache_values = torch.arange(8 * 2 * 16, dtype=torch.float32).reshape(8, 16, 1, 2)
    caches = {group.layer_names[0]: cache_values + group.group_id * 1000 for group in topology.transfer_groups}
    source_blocks = ((), (1, 2, 3, 4, 5, 6), (), (1, 2, 3, 4, 5, 6))
    try:
        worker.bind_kv_caches(caches)
        command = RangeStoreCommand("producer", 0, 96, source_blocks, block_hashes[:6], 97, 17)
        assert store_one(worker, command).evidence.succeeded
        assert worker.take_released_store_job_ids() == {17}
        swa_keys = tuple(key for key in backend.objects if "@group:3@" in key)
        assert sorted(int(key.rsplit("@", 1)[1], 16) * 16 for key in swa_keys) == list(stored_swa_boundaries)

        query_start = len(backend.calls)
        # This consumer extends beyond the producer's prompt; later hashes are absent remotely.
        for start, end, expected_end in ((0, 128, 96), (0, 80, middle_end), (64, 128, 96)):
            result = worker.lookup(LookupRequest(start, end, (1, 3), block_hashes))
            assert result.available_end_token == expected_end
        queried = {key for call in backend.calls[query_start:] if call[0] == "exists" for key in call[1]}
        assert all(key in queried for key in backend.objects)

        expected_tail = {name: cache[6].clone() for name, cache in caches.items()}
        for cache in caches.values():
            cache[7].zero_()
        destination_blocks = ((), (1, 2, 3, 4, 5, 7), (), (1, 2, 3, 4, 5, 7))
        begin_step(worker, load=(LoadCommand("consumer", 0, 96, destination_blocks, block_hashes[:6]),))
        worker.start_load()
        assert not worker.collect_load_result().failed_locations
        worker.end_step()
        for name, cache in caches.items():
            torch.testing.assert_close(cache[7], expected_tail[name])

        window_key = next(key for key in swa_keys if key.endswith("@05"))
        del backend.objects[window_key]
        after_eviction = worker.lookup(LookupRequest(0, 128, (1, 3), block_hashes))
        assert after_eviction.available_end_token == fallback_end
    finally:
        worker.close()
    assert backend.closed and resources.kv_caches is None


@pytest.mark.parametrize("attention_block_size,hit_end,tail_end", ((16, 24, 24), (64, 32, 40)))
@pytest.mark.parametrize(
    "load_async,scheduler_type",
    ((False, SynchronousBulkScheduler), (True, AsynchronousBulkScheduler)),
    ids=("sync", "async"),
)
def test_hybrid_checkpoint_round_trip_receives_full_objects_with_lookup_tail_keys(
    attention_block_size, hit_end, tail_end, load_async, scheduler_type
) -> None:
    topology = make_align_state_topology(
        attention_block_size=attention_block_size, state_block_size=32, hash_block_size=8, tp_size=1
    )
    attention, state = topology.groups
    attention = replace(attention, kv_cache_spec=replace(attention.kv_cache_spec, head_size=2, head_size_v=3))
    topology = replace(topology, groups=(attention, state))
    backend = TensorBytesBackend()
    resources = KVPoolResources(
        backend, make_backend_spec(layerwise_access=None, requires_exists_before_put=True), topology.groups
    )
    worker = BulkWorker(
        topology,
        resources,
        coordinator=make_coordinator(topology, 256),
        load_async=load_async,
        source_ready_event_factory=FakeEvent,
    )
    # Unequal K/V entries have padding between pages, not inside their payloads.
    backing_pages = [torch.full((8, attention_block_size * width + 4), -999.0) for width in (2, 3)]
    attention_caches = tuple(
        backing[:, :-4].view(8, attention_block_size, 1, width)
        for backing, width in zip(backing_pages, (2, 3), strict=True)
    )
    for index, cache in enumerate(attention_caches):
        cache.copy_(torch.arange(cache.numel()).reshape(cache.shape) + index * 1000)
    state_cache = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    hashes = tuple(bytes([index]) for index in range(1, topology.cache_transfer_granularity // 8 + 1))
    source_blocks = (1, 2) if attention_block_size == 16 else (1,)
    destination_blocks = (5, 6) if attention_block_size == 16 else (5,)
    config = replace(
        make_config(granularity=topology.cache_transfer_granularity, groups=(0, 1), align_groups=frozenset((1,))),
        hash_block_size=8,
    )
    lookup = SimpleNamespace(lookup=worker.lookup, close=lambda: None)
    scheduler = scheduler_type(config, lookup)
    try:
        worker.bind_kv_caches({attention.layer_names[0]: attention_caches, state.layer_names[0]: state_cache})
        boundaries = (hit_end,) if tail_end == hit_end else (hit_end, tail_end)
        for index, boundary in enumerate(boundaries):
            command = CheckpointStoreCommand(
                "producer", (source_blocks, ()), hashes, 0, (StateCheckpointSource(1, 3 + index, boundary),), 17 + index
            )
            assert store_one(worker, command).evidence.succeeded
        if tail_end > hit_end:
            # A full Attention companion can outlive the state at the same boundary.
            state_tail_key = next(
                key for key in backend.objects if "@group:1@" in key and key.endswith(f"@{tail_end // 8:02x}")
            )
            del backend.objects[state_tail_key]

        request = make_request(
            "consumer",
            prompt_tokens=topology.cache_transfer_granularity,
            tokens=topology.cache_transfer_granularity + 1,
        )
        request.block_hashes = list(hashes)
        raw = worker.lookup(LookupRequest(0, request.num_prompt_tokens, (0, 1), hashes))
        assert raw.available_end_token == hit_end
        assert raw.tail_key_boundaries == (TailKeyBoundary(0, tail_end), TailKeyBoundary(1, hit_end))
        assert scheduler.get_num_new_matched_tokens(request, 0) == (hit_end, scheduler.load_is_deferred)
        scheduler.confirm_allocation(request, FakeBlocks(list(destination_blocks), [7]), hit_end)
        output = make_output(
            new=(new_request_data("consumer", (destination_blocks, (7,)), hit_end),), scheduled_tokens={"consumer": 0}
        )
        step = scheduler.build_step(output)
        (load,) = step.load_commands
        assert (load.start_token, load.end_token) == (0, hit_end)
        assert load.tail_key_boundaries == raw.tail_key_boundaries

        expected_pages = [backing.clone() for backing in backing_pages]
        for expected, cache in zip(expected_pages, attention_caches, strict=True):
            for source, destination in zip(source_blocks, destination_blocks, strict=True):
                expected[destination, :-4] = cache[source].reshape(-1)
        expected_state = state_cache.clone()
        expected_state[7] = state_cache[3]
        worker.begin_step(step)
        worker.start_load()
        # Public close drains the async copy before resource release and result collection.
        worker.close()
        result = worker.collect_load_result()
        assert not result.failed_request_ids and not result.failed_locations
        if scheduler.load_is_deferred:
            assert result.completed_request_ids == {"consumer"}
        worker.end_step()
        get_call = next(call for call in reversed(backend.calls) if call[0] == "get")
        full_entry_sizes = tuple(cache[0].numel() * cache.element_size() for cache in attention_caches)
        assert get_call[3] == (full_entry_sizes,) * len(destination_blocks) + ((state_cache[0].nbytes,),)
        assert all(key in backend.objects for key in get_call[1])
        for backing, expected in zip(backing_pages, expected_pages, strict=True):
            torch.testing.assert_close(backing, expected)
        torch.testing.assert_close(state_cache, expected_state)
    finally:
        scheduler.close()
        worker.close()
    assert backend.closed and resources.kv_caches is None


def test_ordinary_bulk_worker_owns_lookup_load_store_rows_and_ranges() -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend)

    backend.presence = [1, 0]
    lookup = worker.lookup(LookupRequest(0, 8, (0,), (b"a", b"b")))
    assert lookup.available_end_token == 4
    exists_call = next(call for call in backend.calls if call[0] == "exists")
    assert [key.endswith(suffix) for key, suffix in zip(exists_call[1], ("@61", "@62"), strict=True)] == [True, True]

    backend.get_result = [0, 0]
    load = LoadCommand("load", 0, 6, ((2, 3),), (b"a", b"b"))
    begin_step(worker, load=(load,))
    worker.start_load()
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    get_call = next(call for call in backend.calls if call[0] == "get")
    caches = tuple(resources.kv_caches.values())
    assert get_call[2] == tuple(tuple(cache[block].data_ptr() for cache in caches) for block in (2, 3))
    assert get_call[3] == ((32, 32), (32, 32))

    completion = store_one(worker, RangeStoreCommand("store", 4, 6, ((2, 3),), (b"a", b"b"), 6, 17))
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert put_call[1][0].endswith("@62")
    assert put_call[2:] == ((tuple(cache[3].data_ptr() for cache in caches),), ((32, 32),))
    (evidence,) = completion.evidence.transfer_evidence
    assert evidence.source.local_block_id == 3
    assert evidence.source.physical_layer_ids == (0, 1)

    worker.close()
    assert resources.closed


@pytest.mark.parametrize("put_results", ([0, 0, 0, 0], [0, 0, 0, -1]), ids=("success", "mixed_failure"))
def test_sparse_hybrid_groups_keep_original_identity_through_public_worker(put_results) -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=make_sparse_group_topology())

    lookup = worker.lookup(LookupRequest(0, 8, (1, 3), (b"a", b"b")))
    assert lookup.available_end_token == 8
    exists_calls = [call for call in backend.calls if call[0] == "exists"]
    assert [len(call[1]) for call in exists_calls] == [2, 2]
    assert all("@group:1@" in key for key in exists_calls[0][1])
    assert all("@group:3@" in key for key in exists_calls[1][1])

    backend.calls.clear()
    command = LoadCommand("load", 0, 8, ((), (11, 12), (), (31, 32)), (b"a", b"b"))
    begin_step(worker, load=(command,))
    worker.start_load()
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    get_call = next(call for call in backend.calls if call[0] == "get")
    assert ["@group:1@" in key for key in get_call[1]] == [True, True, False, False]
    assert ["@group:3@" in key for key in get_call[1]] == [False, False, True, True]
    caches = tuple(resources.kv_caches.values())
    assert get_call[2] == tuple(
        (cache[block].data_ptr(),) for cache, ids in zip(caches, ((11, 12), (31, 32)), strict=True) for block in ids
    )
    assert get_call[3] == ((32,), (32,), (32,), (32,))

    backend.calls.clear()
    backend.put_result = put_results
    completion = store_one(worker, RangeStoreCommand("store", 0, 8, ((), (11, 12), (), (31, 32)), (b"a", b"b"), 8, 17))
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert put_call[1:] == (get_call[1], get_call[2], get_call[3])
    assert [item.source.group_id for item in completion.evidence.transfer_evidence] == [1, 1, 3, 3]
    assert [item.source.local_block_id for item in completion.evidence.transfer_evidence] == [11, 12, 31, 32]
    assert [item.source.physical_layer_ids for item in completion.evidence.transfer_evidence] == [
        (0,),
        (0,),
        (1,),
        (1,),
    ]

    assert [item.result_code for item in completion.evidence.transfer_evidence] == put_results
    assert completion.evidence.succeeded is all(code == 0 for code in put_results)
    assert completion.evidence.source_release_confirmed
    assert worker.take_released_store_job_ids() == {17}

    worker.close()
    assert resources.closed


def test_hybrid_bulk_worker_owns_fine_lookup_load_range_and_checkpoint_store(monkeypatch) -> None:
    backend = FakeBackend()
    backend.presence_results = [[1, 1], [1, 1, 1, 1]]
    worker, resources, _ = make_worker(backend, topology=make_align_state_topology())

    lookup = worker.lookup(LookupRequest(0, 8, (0, 1), (b"a", b"b")))
    assert lookup.available_end_token == 8
    exists_calls = [call for call in backend.calls if call[0] == "exists"]
    assert [len(call[1]) for call in exists_calls] == [2, 4]
    assert all("@group:0@" in key for key in exists_calls[0][1])
    assert ["@head_or_tp_rank:0@" in key for key in exists_calls[1][1][::2]] == [True, False]
    assert ["@head_or_tp_rank:1@" in key for key in exists_calls[1][1][::2]] == [False, True]

    backend.calls.clear()
    monkeypatch.setattr(worker._coordinator, "select_for_load", lambda *_args: (None, None))
    load = LoadCommand("load", 0, 4, ((3,), (7,)), (b"a",), (TailKeyBoundary(0, 4), TailKeyBoundary(1, 4)))
    begin_step(worker, load=(load,))
    worker.start_load()
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    get_call = next(call for call in backend.calls if call[0] == "get")
    assert ["@group:0@" in key for key in get_call[1]] == [True, False]
    assert ["@group:1@" in key for key in get_call[1]] == [False, True]
    attention, state = resources.kv_caches.values()
    expected_addresses = ((attention[3].data_ptr(),), (state[7].data_ptr(),))
    assert get_call[2] == expected_addresses
    assert get_call[3] == ((32,), (32,))

    backend.calls.clear()
    range_completion = store_one(worker, RangeStoreCommand("range", 0, 8, ((3,), (7,)), (b"a", b"b"), 8, 17))
    range_put = next(call for call in backend.calls if call[0] == "put")
    assert len(range_put[1]) == 1 and "@group:0@" in range_put[1][0]
    assert range_put[2:] == (expected_addresses[:1], ((32,),))
    assert [item.source.group_id for item in range_completion.evidence.transfer_evidence] == [0]

    backend.calls.clear()
    checkpoint_completion = store_one(
        worker,
        CheckpointStoreCommand("checkpoint", ((3,), (99,)), (b"a",), 0, (StateCheckpointSource(1, 7, 4),), 18),
    )
    checkpoint_put = next(call for call in backend.calls if call[0] == "put")
    assert ["@group:0@" in key for key in checkpoint_put[1]] == [True, False]
    assert ["@group:1@" in key for key in checkpoint_put[1]] == [False, True]
    assert checkpoint_put[2:] == (expected_addresses, ((32,), (32,)))
    assert [item.source.group_id for item in checkpoint_completion.evidence.transfer_evidence] == [0, 1]
    assert [item.source.physical_layer_ids for item in checkpoint_completion.evidence.transfer_evidence] == [(0,), (1,)]

    worker.close()
    assert resources.closed


def test_hybrid_checkpoint_requires_unique_exact_align_state_sources() -> None:
    worker, resources, _ = make_worker(topology=make_align_state_topology())

    duplicate = CheckpointStoreCommand(
        "duplicate",
        ((3,), (99,)),
        (b"a",),
        0,
        (StateCheckpointSource(1, 7, 4), StateCheckpointSource(1, 8, 4)),
        17,
    )
    with pytest.raises(ValueError, match="duplicate cache-group sources"):
        worker._build_store_candidates((duplicate,))

    invalid = CheckpointStoreCommand("invalid", ((3,), (99,)), (b"a",), 0, (StateCheckpointSource(0, 3, 4),), 18)
    with pytest.raises(ValueError, match="invalid source groups"):
        worker._build_store_candidates((invalid,))

    unaligned = CheckpointStoreCommand("unaligned", ((3,), (99,)), (b"a",), 0, (StateCheckpointSource(1, 7, 3),), 19)
    with pytest.raises(ValueError, match="not aligned to the hash block size"):
        worker._build_store_candidates((unaligned,))

    null_source = CheckpointStoreCommand("null", ((3,), (99,)), (b"a",), 0, (StateCheckpointSource(1, 0, 4),), 20)
    null_candidates = worker._build_store_candidates((null_source,))
    assert [group.transfer_record_count for group in null_candidates.groups] == [0, 0]

    aligned = CheckpointStoreCommand("aligned", ((3,), (99,)), (b"a", b"b"), 0, (StateCheckpointSource(1, 7, 8),), 21)
    aligned_candidates = worker._build_store_candidates((aligned,))
    assert [group.transfer_record_count for group in aligned_candidates.groups] == [0, 1]

    worker.close()
    assert resources.closed


def test_consumer_pipeline_store_keeps_key_memory_and_layer_provenance_atomic() -> None:
    topology = make_topology(physical_layers=(0, 1, 2, 3), consumer_pipeline_partitions=(2, 2))
    backend = FakeBackend()
    backend.presence = [1, 0]
    worker, resources, _ = make_worker(backend, topology=topology, requires_exists_before_put=True)

    completion = store_one(worker, RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17))

    exists_call = next(call for call in backend.calls if call[0] == "exists")
    assert ["@pp_rank:0@" in exists_call[1][0], "@pp_rank:1@" in exists_call[1][1]] == [True, True]
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 1 and "@pp_rank:1@" in put_call[1][0]
    assert put_call[2] == (tuple(resources.kv_caches[f"layers.{layer}.group.0"][1].data_ptr() for layer in (2, 3)),)
    assert put_call[3] == ((32, 32),)
    (evidence,) = completion.evidence.transfer_evidence
    assert evidence.source.physical_layer_ids == (2, 3)
    assert "@pp_rank:1@" in evidence.source.key

    worker.close()
    assert resources.closed


def test_ordinary_bulk_partitions_writers_after_candidate_filtering() -> None:
    topology = make_topology(physical_layers=(0,), tp_rank=1, tp_size=4, pcp_rank=1, pcp_size=2, put_step=2)
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=topology)

    completion = store_one(
        worker,
        RangeStoreCommand("request", 0, 16, ((1, 2, 3, 4),), (b"a", b"b", b"c", b"d"), 16, 17),
    )

    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 1 and put_call[1][0].endswith("@64")
    assert put_call[2] == ((resources.kv_caches["layers.0.group.0"][4].data_ptr(),),)
    assert completion.evidence.transfer_evidence[0].source.local_block_id == 4

    worker.close()
    assert resources.closed


def test_dcp_bulk_keeps_every_row_of_its_distinct_key_shard() -> None:
    topology = make_topology(physical_layers=(0,), tp_rank=1, tp_size=4, dcp_rank=1, dcp_size=2, put_step=2)
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=topology)

    completion = store_one(worker, RangeStoreCommand("request", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17))

    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 2
    assert all("@dcp:1@" in key for key in put_call[1])
    cache = resources.kv_caches["layers.0.group.0"]
    assert put_call[2] == ((cache[1].data_ptr(),), (cache[2].data_ptr(),))
    assert [item.source.local_block_id for item in completion.evidence.transfer_evidence] == [1, 2]

    worker.close()
    assert resources.closed


def test_lookup_exposes_backend_observation_and_preserves_best_effort_miss(caplog, monkeypatch) -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, store=False)
    request = LookupRequest(0, 8, (0,), (b"a", b"b"))

    assert worker.lookup(request).available_end_token == 8
    exists_call = next(call for call in backend.calls if call[0] == "exists")
    assert len(exists_call[1]) == 2
    assert all("@group:0@" in key for key in exists_call[1])

    def fail_lookup(keys):
        backend.calls.append(("failed_exists", tuple(keys)))
        raise RuntimeError("lookup unavailable")

    monkeypatch.setattr(backend, "exists", fail_lookup)
    assert worker.lookup(request).available_end_token == 0
    assert "Remote Lookup failed" in caplog.text

    worker.close()
    assert resources.closed


@pytest.mark.parametrize("native_result", ([-1], RuntimeError("get failed")), ids=("failed", "unknown"))
def test_synchronous_bulk_load_reports_failed_block_for_invalid_backend_results(native_result, monkeypatch) -> None:
    backend = FakeBackend()

    def get(keys, addresses, sizes):
        backend.calls.append(("get", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        if isinstance(native_result, BaseException):
            raise native_result
        return native_result

    monkeypatch.setattr(backend, "get", get)
    worker, resources, _ = make_worker(backend, store=False)
    command = LoadCommand("request", 0, 4, ((7,),), (b"a",))
    try:
        begin_step(worker, load=(command,))
        worker.start_load()
        result = worker.collect_load_result()

        assert [call[0] for call in backend.calls].count("get") == 1
        assert result.failed_block_ids == {7}
        assert [(location.group_id, location.block_id) for location in result.failed_locations] == [(0, 7)]
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


def test_multi_group_load_failure_reports_original_group_and_block() -> None:
    topology = make_topology(group_ids=(1, 3), physical_layers=(0,))
    backend = FakeBackend()
    backend.get_result = [0, -1]
    worker, resources, _ = make_worker(backend, topology=topology, store=False)
    command = LoadCommand("request", 0, 4, ((), (11,), (), (33,)), (b"a",))
    begin_step(worker, load=(command,))

    with pytest.raises(
        RuntimeError,
        match=r"Hybrid KV Load failed for requests \['request'\].*cache group/block failures \[\(3, 33\)\]",
    ):
        worker.start_load()

    worker.end_step()
    worker.close()
    assert resources.closed


@pytest.mark.parametrize(
    "failure_stage,expected_trace",
    (
        (None, ["record", "prepare", "synchronize", "ready", "put"]),
        ("prepare", ["record", "prepare"]),
        ("synchronize", ["record", "prepare", "synchronize"]),
    ),
)
def test_bulk_store_prepares_before_source_wait_and_fences_before_close(
    failure_stage, expected_trace, monkeypatch
) -> None:
    trace: list[str] = []

    class TraceEvent(FakeEvent):
        def record(self) -> None:
            trace.append("record")
            super().record()

        def synchronize(self) -> None:
            trace.append("synchronize")
            if failure_stage == "synchronize":
                raise RuntimeError("source readiness failed")
            super().synchronize()
            trace.append("ready")

    class TraceBackend(FakeBackend):
        def put(self, keys, addresses, sizes):
            assert source_ready_event.synchronized
            trace.append("put")
            return super().put(keys, addresses, sizes)

    source_ready_event = TraceEvent()
    backend = TraceBackend()
    worker, resources, _ = make_worker(backend, source_ready_event_factory=lambda: source_ready_event)
    materialize_arguments = worker._materialize_bulk_arguments

    def prepare_arguments(batch):
        arguments = materialize_arguments(batch)
        trace.append("prepare")
        assert not source_ready_event.synchronized
        if failure_stage == "prepare":
            raise RuntimeError("argument preparation failed")
        return arguments

    monkeypatch.setattr(worker, "_materialize_bulk_arguments", prepare_arguments)
    command = RangeStoreCommand("request", 0, 4, ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))

    worker.finish_step()
    worker.close()

    assert trace == expected_trace
    assert worker.take_released_store_job_ids() == {17}
    assert resources.closed


@pytest.mark.parametrize("native_result", ([], RuntimeError("put failed after address handoff")))
def test_bulk_store_unknown_evidence_keeps_registered_resources(native_result) -> None:
    backend = FakeBackend()
    backend.put_result = native_result
    worker, resources, _ = make_worker(backend)
    command = RangeStoreCommand("request", 0, 4, ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    worker.finish_step()

    with pytest.raises(RuntimeError, match="Store source release is unknown"):
        worker.close()

    assert worker.take_released_store_job_ids() == set()
    assert not resources.closed
