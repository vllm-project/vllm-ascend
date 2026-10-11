"""Bulk Lookup, Load, and Store preserve keys, ranges, and failure identity."""

from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    begin_step,
    make_backend_spec,
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
from tests.ut.distributed.ascend_store.v1.worker.bulk_fixtures import (
    TensorBytesBackend,
    make_align_state_topology,
    make_sparse_group_topology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import (
    TokenRange,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import compile_bulk_projection_binder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupRequest,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    CheckpointStoreCommand,
    LoadCommand,
    RangeStoreCommand,
    StateCheckpointSource,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import (
    AsynchronousBulkScheduler,
    SynchronousBulkScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import base as worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import (
    AsynchronousBulkWorker,
    SynchronousBulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.io import BackendIO
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.io.arguments import BulkBackendArguments
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.resources import KVPoolResources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import KVTransferBatch


@pytest.mark.parametrize(
    "retention_interval,stored_swa_boundaries,middle_end,fallback_end",
    ((None, (16, 32, 48, 64, 80, 96), 80, 64), (0, (80, 96), 0, 0), (64, (48, 64, 80, 96), 80, 64)),
)
def test_bulk_lookup_recovers_sparse_retention_objects_and_loads_tail(
    retention_interval, stored_swa_boundaries, middle_end, fallback_end
) -> None:
    topology = make_sparse_group_topology(block_size=16, sliding_window=33)
    backend = TensorBytesBackend()
    resources = KVPoolResources(
        backend, make_backend_spec(requires_exists_before_put=True), 8, topology.transfer_groups
    )
    binder = compile_bulk_projection_binder(topology, 256, retention_interval=retention_interval)
    worker = SynchronousBulkWorker(topology, binder, resources, source_ready_event_factory=FakeEvent)
    block_hashes = tuple(bytes([index]) for index in range(1, 9))
    cache_values = torch.arange(8 * 2 * 16, dtype=torch.float32).reshape(8, 2, 16, 1, 1)
    caches = {group.layer_names[0]: cache_values + group.group_id * 1000 for group in topology.transfer_groups}
    source_blocks = ((), (1, 2, 3, 4, 5, 6), (), (1, 2, 3, 4, 5, 6))
    try:
        worker.bind_kv_caches(caches)
        command = RangeStoreCommand("producer", TokenRange(0, 96), source_blocks, block_hashes[:6], 97, 17)
        assert store_one(worker, command).evidence.succeeded
        assert worker.take_released_store_job_ids() == {17}
        swa_keys = tuple(key for key in backend.objects if "@group:3@" in key)
        assert sorted(int(key.rsplit("@", 1)[1], 16) * 16 for key in swa_keys) == list(stored_swa_boundaries)

        query_start = len(backend.calls)
        # This consumer extends beyond the producer's prompt; later hashes are absent remotely.
        for start, end, expected_end in ((0, 128, 96), (0, 80, middle_end), (64, 128, 96)):
            result = worker.lookup(LookupRequest(TokenRange(start, end), (1, 3), block_hashes))
            assert result.available_end_token == expected_end
        queried = {key for call in backend.calls[query_start:] if call[0] == "exists" for key in call[1]}
        assert all(key in queried for key in backend.objects)

        expected_tail = {name: cache[6].clone() for name, cache in caches.items()}
        for cache in caches.values():
            cache[7].zero_()
        destination_blocks = ((), (1, 2, 3, 4, 5, 7), (), (1, 2, 3, 4, 5, 7))
        begin_step(worker, load=(LoadCommand("consumer", TokenRange(0, 96), destination_blocks, block_hashes[:6]),))
        worker.start_load()
        assert not worker.collect_load_result().failed_locations
        worker.end_step()
        for name, cache in caches.items():
            torch.testing.assert_close(cache[7], expected_tail[name])

        window_key = next(key for key in swa_keys if key.endswith("@05"))
        del backend.objects[window_key]
        after_eviction = worker.lookup(LookupRequest(TokenRange(0, 128), (1, 3), block_hashes))
        assert after_eviction.available_end_token == fallback_end
    finally:
        worker.close()
    assert backend.closed and resources.kv_caches is None


@pytest.mark.parametrize("attention_block_size,hit_end,tail_end", ((16, 24, 24), (64, 32, 40)))
@pytest.mark.parametrize(
    "worker_type,scheduler_type",
    ((SynchronousBulkWorker, SynchronousBulkScheduler), (AsynchronousBulkWorker, AsynchronousBulkScheduler)),
    ids=("sync", "async"),
)
def test_hybrid_checkpoint_round_trip_receives_full_objects_with_lookup_tail_keys(
    attention_block_size, hit_end, tail_end, worker_type, scheduler_type
) -> None:
    topology = make_align_state_topology(
        attention_block_size=attention_block_size, state_block_size=32, hash_block_size=8, tp_size=1
    )
    attention, state = topology.groups
    attention = replace(attention, kv_cache_spec=replace(attention.kv_cache_spec, head_size=2, head_size_v=3))
    topology = replace(topology, groups=(attention, state))
    backend = TensorBytesBackend()
    resources = KVPoolResources(
        backend, make_backend_spec(layerwise_access=None, requires_exists_before_put=True), 8, topology.groups
    )
    worker = worker_type(
        topology, compile_bulk_projection_binder(topology, 256), resources, source_ready_event_factory=FakeEvent
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
    lookup = SimpleNamespace(
        query=lambda query_range, group_ids, block_hashes: worker.lookup(
            LookupRequest(query_range, group_ids, block_hashes)
        ),
        close=lambda: None,
    )
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
        raw = worker.lookup(LookupRequest(TokenRange(0, request.num_prompt_tokens), (0, 1), hashes))
        assert raw.available_end_token == hit_end
        assert raw.tail_key_boundaries == (TailKeyBoundary(0, tail_end), TailKeyBoundary(1, hit_end))
        assert scheduler.get_num_new_matched_tokens(request, 0) == (hit_end, scheduler.load_is_deferred)
        scheduler.confirm_allocation(request, FakeBlocks(list(destination_blocks), [7]), hit_end)
        output = make_output(
            new=(new_request_data("consumer", (destination_blocks, (7,)), hit_end),), scheduled_tokens={"consumer": 0}
        )
        step = scheduler.build_step(output)
        (load,) = step.load.commands
        assert load.load_range == TokenRange(0, hit_end)
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
    lookup = worker.lookup(LookupRequest(TokenRange(0, 8), (0,), (b"a", b"b")))
    assert lookup.available_end_token == 4
    exists_call = next(call for call in backend.calls if call[0] == "exists")
    assert [key.endswith(suffix) for key, suffix in zip(exists_call[1], ("@61", "@62"), strict=True)] == [
        True,
        True,
    ]

    backend.get_result = [0, 0]
    load = LoadCommand("load", TokenRange(0, 6), ((2, 3),), (b"a", b"b"))
    begin_step(worker, load=(load,))
    worker.start_load()
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    get_call = next(call for call in backend.calls if call[0] == "get")
    assert get_call[2] == ((1128, 2128), (1192, 2192))
    assert get_call[3] == ((32, 32), (16, 16))

    completion = store_one(
        worker,
        RangeStoreCommand("store", TokenRange(4, 6), ((2, 3),), (b"a", b"b"), 6, 17),
    )
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert put_call[1][0].endswith("@62")
    assert put_call[2:] == (((1192, 2192),), ((16, 16),))
    (evidence,) = completion.evidence.transfer_evidence
    assert evidence.source.block_id == 3
    assert evidence.source.physical_layer_ids == (0, 1)

    worker.close()
    assert resources.closed


@pytest.mark.parametrize("put_results", ([0, 0, 0, 0], [0, 0, 0, -1]), ids=("success", "mixed_failure"))
def test_sparse_hybrid_groups_keep_original_identity_through_public_worker(put_results) -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=make_sparse_group_topology())

    lookup = worker.lookup(LookupRequest(TokenRange(0, 8), (1, 3), (b"a", b"b")))
    assert lookup.available_end_token == 8
    exists_calls = [call for call in backend.calls if call[0] == "exists"]
    assert [len(call[1]) for call in exists_calls] == [2, 2]
    assert all("@group:1@" in key for key in exists_calls[0][1])
    assert all("@group:3@" in key for key in exists_calls[1][1])

    backend.calls.clear()
    command = LoadCommand(
        "load",
        TokenRange(0, 8),
        ((), (11, 12), (), (31, 32)),
        (b"a", b"b"),
    )
    begin_step(worker, load=(command,))
    worker.start_load()
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    get_call = next(call for call in backend.calls if call[0] == "get")
    assert ["@group:1@" in key for key in get_call[1]] == [True, True, False, False]
    assert ["@group:3@" in key for key in get_call[1]] == [False, False, True, True]
    assert get_call[2] == ((11704,), (11768,), (33984,), (34048,))
    assert get_call[3] == ((32,), (32,), (32,), (32,))

    backend.calls.clear()
    backend.put_result = put_results
    completion = store_one(
        worker,
        RangeStoreCommand(
            "store",
            TokenRange(0, 8),
            ((), (11, 12), (), (31, 32)),
            (b"a", b"b"),
            8,
            17,
        ),
    )
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert put_call[1:] == (get_call[1], get_call[2], get_call[3])
    assert [item.source.group_id for item in completion.evidence.transfer_evidence] == [1, 1, 3, 3]
    assert [item.source.block_id for item in completion.evidence.transfer_evidence] == [11, 12, 31, 32]
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

    lookup = worker.lookup(LookupRequest(TokenRange(0, 8), (0, 1), (b"a", b"b")))
    assert lookup.available_end_token == 8
    exists_calls = [call for call in backend.calls if call[0] == "exists"]
    assert [len(call[1]) for call in exists_calls] == [2, 4]
    assert all("@group:0@" in key for key in exists_calls[0][1])
    assert ["@head_or_tp_rank:0@" in key for key in exists_calls[1][1][::2]] == [True, False]
    assert ["@head_or_tp_rank:1@" in key for key in exists_calls[1][1][::2]] == [False, True]

    backend.calls.clear()
    monkeypatch.setattr(worker._bulk_projection.reachability, "select_for_load", lambda *_args: (None, None))
    load = LoadCommand(
        "load",
        TokenRange(0, 4),
        ((3,), (7,)),
        (b"a",),
        (TailKeyBoundary(0, 4), TailKeyBoundary(1, 4)),
    )
    begin_step(worker, load=(load,))
    worker.start_load()
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    get_call = next(call for call in backend.calls if call[0] == "get")
    assert ["@group:0@" in key for key in get_call[1]] == [True, False]
    assert ["@group:1@" in key for key in get_call[1]] == [False, True]
    assert get_call[2] == ((1192,), (12448,))
    assert get_call[3] == ((32,), (32,))

    backend.calls.clear()
    range_completion = store_one(
        worker,
        RangeStoreCommand(
            "range",
            TokenRange(0, 8),
            ((3,), (7,)),
            (b"a", b"b"),
            8,
            17,
        ),
    )
    range_put = next(call for call in backend.calls if call[0] == "put")
    assert len(range_put[1]) == 1 and "@group:0@" in range_put[1][0]
    assert range_put[2:] == (((1192,),), ((32,),))
    assert [item.source.group_id for item in range_completion.evidence.transfer_evidence] == [0]

    backend.calls.clear()
    checkpoint_completion = store_one(
        worker,
        CheckpointStoreCommand(
            "checkpoint",
            ((3,), (99,)),
            (b"a",),
            0,
            (StateCheckpointSource(1, 7, 4),),
            18,
        ),
    )
    checkpoint_put = next(call for call in backend.calls if call[0] == "put")
    assert ["@group:0@" in key for key in checkpoint_put[1]] == [True, False]
    assert ["@group:1@" in key for key in checkpoint_put[1]] == [False, True]
    assert checkpoint_put[2:] == (((1192,), (12448,)), ((32,), (32,)))
    assert [item.source.group_id for item in checkpoint_completion.evidence.transfer_evidence] == [0, 1]
    assert [item.source.physical_layer_ids for item in checkpoint_completion.evidence.transfer_evidence] == [
        (0,),
        (1,),
    ]

    worker.close()
    assert resources.closed


def test_hybrid_checkpoint_requires_unique_exact_align_state_sources() -> None:
    worker, resources, _ = make_worker(topology=make_align_state_topology())

    duplicate = CheckpointStoreCommand(
        "duplicate",
        ((3,), (99,)),
        (b"a",),
        0,
        (
            StateCheckpointSource(1, 7, 4),
            StateCheckpointSource(1, 8, 4),
        ),
        17,
    )
    with pytest.raises(ValueError, match="duplicate cache-group sources"):
        worker._build_store_candidates((duplicate,))

    invalid = CheckpointStoreCommand(
        "invalid",
        ((3,), (99,)),
        (b"a",),
        0,
        (StateCheckpointSource(0, 3, 4),),
        18,
    )
    with pytest.raises(ValueError, match="invalid source groups"):
        worker._build_store_candidates((invalid,))

    unaligned = CheckpointStoreCommand(
        "unaligned",
        ((3,), (99,)),
        (b"a",),
        0,
        (StateCheckpointSource(1, 7, 3),),
        19,
    )
    with pytest.raises(ValueError, match="not aligned to the hash block size"):
        worker._build_store_candidates((unaligned,))

    null_source = CheckpointStoreCommand(
        "null",
        ((3,), (99,)),
        (b"a",),
        0,
        (StateCheckpointSource(1, 0, 4),),
        20,
    )
    null_candidates = worker._build_store_candidates((null_source,))
    assert [group.row_count for group in null_candidates.groups] == [0, 0]

    aligned = CheckpointStoreCommand(
        "aligned",
        ((3,), (99,)),
        (b"a", b"b"),
        0,
        (StateCheckpointSource(1, 7, 8),),
        21,
    )
    aligned_candidates = worker._build_store_candidates((aligned,))
    assert [group.row_count for group in aligned_candidates.groups] == [0, 1]

    worker.close()
    assert resources.closed


def test_hybrid_bulk_load_failure_is_request_scoped() -> None:
    backend = FakeBackend()
    backend.get_result = [0, -1]
    worker, resources, _ = make_worker(backend, topology=make_topology(group_ids=(1, 3), physical_layers=(0,)))
    command = LoadCommand(
        "request",
        TokenRange(0, 4),
        ((), (1,), (), (2,)),
        (b"a",),
    )

    begin_step(worker, load=(command,))
    with pytest.raises(RuntimeError, match="Hybrid KV Load failed.*request"):
        worker.start_load()
    worker.end_step()

    worker.close()
    assert resources.closed


def test_consumer_pipeline_store_keeps_key_memory_and_layer_provenance_atomic() -> None:
    topology = make_topology(
        physical_layers=(0, 1, 2, 3),
        consumer_pipeline_partitions=(2, 2),
    )
    backend = FakeBackend()
    backend.presence = [1, 0]
    worker, resources, _ = make_worker(
        backend,
        topology=topology,
        requires_exists_before_put=True,
    )

    completion = store_one(
        worker,
        RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17),
    )

    exists_call = next(call for call in backend.calls if call[0] == "exists")
    assert ["@pp_rank:0@" in exists_call[1][0], "@pp_rank:1@" in exists_call[1][1]] == [True, True]
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 1 and "@pp_rank:1@" in put_call[1][0]
    assert put_call[2] == ((3064, 4064),)
    assert put_call[3] == ((32, 32),)
    (evidence,) = completion.evidence.transfer_evidence
    assert evidence.source.physical_layer_ids == (2, 3)
    assert "@pp_rank:1@" in evidence.source.key

    worker.close()
    assert resources.closed


def test_ordinary_bulk_partitions_writers_after_candidate_filtering() -> None:
    topology = make_topology(
        physical_layers=(0,),
        tp_rank=1,
        tp_size=4,
        pcp_rank=1,
        pcp_size=2,
        put_step=2,
    )
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=topology)

    completion = store_one(
        worker,
        RangeStoreCommand(
            "request",
            TokenRange(0, 16),
            ((1, 2, 3, 4),),
            (b"a", b"b", b"c", b"d"),
            16,
            17,
        ),
    )

    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 1 and put_call[1][0].endswith("@64")
    assert put_call[2] == ((1256,),)
    assert completion.evidence.transfer_evidence[0].source.block_id == 4

    worker.close()
    assert resources.closed


def test_dcp_bulk_keeps_every_row_of_its_distinct_key_shard() -> None:
    topology = make_topology(
        physical_layers=(0,),
        tp_rank=1,
        tp_size=4,
        dcp_rank=1,
        dcp_size=2,
        put_step=2,
    )
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=topology)

    completion = store_one(
        worker,
        RangeStoreCommand(
            "request",
            TokenRange(0, 8),
            ((1, 2),),
            (b"a", b"b"),
            8,
            17,
        ),
    )

    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 2
    assert all("@dcp:1@" in key for key in put_call[1])
    assert put_call[2] == ((1064,), (1128,))
    assert [item.source.block_id for item in completion.evidence.transfer_evidence] == [1, 2]

    worker.close()
    assert resources.closed


def test_lookup_exposes_backend_observation_and_preserves_best_effort_miss(caplog, monkeypatch) -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, store=False)
    request = LookupRequest(TokenRange(0, 8), (0,), (b"a", b"b"))

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


@pytest.mark.parametrize(
    "native_result",
    ([-1], None, [0, 0], [True], RuntimeError("get failed")),
    ids=("failed", "missing", "wrong_count", "invalid_type", "exception"),
)
def test_synchronous_bulk_load_reports_failed_block_for_invalid_backend_results(native_result, monkeypatch) -> None:
    backend = FakeBackend()

    def get(keys, addresses, sizes):
        backend.calls.append(("get", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        if isinstance(native_result, BaseException):
            raise native_result
        return native_result

    monkeypatch.setattr(backend, "get", get)
    worker, resources, _ = make_worker(backend, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
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
    command = LoadCommand(
        "request",
        TokenRange(0, 4),
        ((), (11,), (), (33,)),
        (b"a",),
    )
    begin_step(worker, load=(command,))

    with pytest.raises(RuntimeError, match=r"cache group/block failures \[\(3, 33\)\]"):
        worker.start_load()

    worker.end_step()
    worker.close()
    assert resources.closed


def test_bulk_store_orders_source_ready_copy_and_close_fence() -> None:
    trace: list[str] = []

    class TraceEvent(FakeEvent):
        def record(self) -> None:
            trace.append("record")
            super().record()

        def synchronize(self) -> None:
            trace.append("synchronize")
            super().synchronize()

    class TraceBackend(FakeBackend):
        def put(self, keys, addresses, sizes):
            trace.append("put")
            return super().put(keys, addresses, sizes)

    backend = TraceBackend()
    worker, resources, _ = make_worker(
        backend,
        source_ready_event_factory=TraceEvent,
    )
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))

    worker.finish_step()
    worker.close()

    assert trace == ["record", "synchronize", "put"]
    assert worker.take_released_store_job_ids() == {17}
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
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    if layerwise:
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
    worker.finish_step()
    worker.fence_previous_store()

    assert "KV cache Store failed for request request (job 17)" in caplog.text
    assert worker.take_released_store_job_ids() == {17}
    assert worker._pending_store_batch is None
    assert worker.lookup(LookupRequest(TokenRange(0, 4), (0,), (b"a",))).available_end_token == 0
    if layerwise:
        assert "batch_commit" not in [call[0] for call in backend.calls]
        assert "batch_revoke" in [call[0] for call in backend.calls]
    worker.end_step()

    backend.put_result = None
    backend.store_session_copy_result = None
    next_command = RangeStoreCommand("next", TokenRange(0, 4), ((9,),), (b"b",), 4, 18)
    begin_step(worker, store=(next_command,))
    if layerwise:
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
    worker.finish_step()
    worker.fence_previous_store()
    assert worker.take_released_store_job_ids() == {18}
    worker.close()
    assert resources.closed


@pytest.mark.parametrize("native_result", ([], [True], RuntimeError("put failed after address handoff")))
def test_bulk_store_unknown_evidence_keeps_registered_resources(native_result) -> None:
    backend = FakeBackend()
    backend.put_result = native_result
    worker, resources, _ = make_worker(backend)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    worker.finish_step()

    with pytest.raises(RuntimeError, match="Store source release is unknown"):
        worker.close()

    assert worker.take_released_store_job_ids() == set()
    assert not resources.closed


def test_bulk_store_argument_alignment_error_is_not_suppressed() -> None:
    backend = FakeBackend()
    backend_io = BackendIO(backend, make_backend_spec(layerwise_access=None))
    arguments = BulkBackendArguments(keys=["key"], addresses=[[1000]], sizes=[[32]], sources=())
    with pytest.raises(RuntimeError, match="Bulk Store arguments do not align"):
        backend_io.store_materialized(KVTransferBatch((), ()), arguments)
    assert "put" not in [call[0] for call in backend.calls]
