"""Business and lifecycle contracts that remain above the bound execution plan."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection.reachability import (
    HybridReachability,
    ReachablePrefix,
    UnitaryReachability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupCodec,
    LookupRequest,
    LookupResult,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    LoadCommandBatch,
    RangeStoreCommand,
    StateCheckpointSource,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.timeline.asynchronous_load import (
    AsynchronousLoadTimeline,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.timeline.asynchronous_store import (
    AsynchronousStoreTimeline,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.timeline.layerwise_load import (
    LayerwiseLoadTimeline,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.timeline.layerwise_store import (
    LayerwiseStoreTimeline,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.timeline.synchronous_load import (
    SynchronousLoadTimeline,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
    resolve_group_layers,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.io import (
    arguments as arguments_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
    KVTransferBatch,
)

from .v1.helpers import (
    FakeBackend,
    FakeEvent,
    build_admitted_store_batch,
    make_topology,
    make_worker,
    worker_backend_io,
)


class FakeLoadStartGate:
    def __init__(self, opened: bool = False) -> None:
        self._opened = threading.Event()
        if opened:
            self._opened.set()

    def open(self) -> None:
        self._opened.set()

    def cancel(self) -> None:
        self._opened.set()

    def wait(self, timeout) -> bool:
        return self._opened.wait(timeout)


def begin_step(worker, *, load=None, store=None) -> None:
    worker.begin_step(KVTransferStep(load or LoadCommandBatch(), store or StoreCommandBatch()))


def test_v1_rejects_yuanrong_during_configuration(monkeypatch) -> None:
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"backend": "yuanrong"}),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    kv_cache_config = SimpleNamespace(prefix_cache_retention_interval=None)
    monkeypatch.setattr(vllm_adapter, "_resolve_kv_pool_topology", lambda *_: object())

    with pytest.raises(ValueError, match="temporarily does not support the Yuanrong Backend"):
        vllm_adapter.resolve_kv_pool_route_spec(config, kv_cache_config)


def test_small_value_and_codec_contracts_share_one_domain_smoke_test() -> None:
    invalid_ranges = ((-1, 0), (4, 3))
    for start, end in invalid_ranges:
        with pytest.raises(ValueError):
            TokenRange(start, end)

    layers = resolve_group_layers(["model.layers.1.v", "mtp.layers.0.attn", "model.layers.1.k"], base_layer_count=4)
    assert layers == (
        KVPoolLayerTopology(1, ("model.layers.1.k", "model.layers.1.v")),
        KVPoolLayerTopology(4, ("mtp.layers.0.attn",)),
    )

    codec = LookupCodec()
    request = LookupRequest(TokenRange(4, 12), (1, 3), (b"a", b"b"))
    result = LookupResult(8, (TailKeyBoundary(1, 12), TailKeyBoundary(3, 8)))
    assert codec.decode_request(codec.encode_request(request)) == request
    assert codec.decode_result(codec.encode_result(result)) == result


def test_reachability_matrix_preserves_contiguous_and_partial_tail_semantics() -> None:
    unitary = UnitaryReachability(0, max_model_len=64, cache_transfer_granularity=4)
    block_hashes = (b"a", b"b", b"c")
    query_range = TokenRange(0, 12)
    observations = (
        (
            (np.asarray((0, 4, 8)), np.asarray((4, 4, 4)), block_hashes),
            (True, False, True),
        ),
    )
    assert unitary.select_for_lookup(block_hashes, query_range) == (None,)
    assert unitary.resolve_available_end(query_range, block_hashes, observations) == ReachablePrefix(4)

    groups = (
        KVPoolGroupTopology(
            0,
            FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32),
            (KVPoolLayerTopology(0, ("layers.0",)),),
            make_topology().groups[0].key_metadata,
        ),
        KVPoolGroupTopology(
            1,
            MambaSpec(
                block_size=16,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
            (KVPoolLayerTopology(1, ("layers.1",)),),
            replace(make_topology().groups[0].key_metadata, kv_cache_group_id=1),
        ),
    )
    hybrid = HybridReachability(groups, 16, 4, 64)
    hybrid_observations = tuple(
        (
            (np.asarray((0, 0, 0)), np.asarray((4, 8, 12)), block_hashes),
            (False, False, True),
        )
        for _group_id in (0, 1)
    )
    assert hybrid.resolve_available_end(query_range, block_hashes, hybrid_observations) == ReachablePrefix(
        12,
        (TailKeyBoundary(0, 12), TailKeyBoundary(1, 12)),
    )


def test_bulk_worker_store_success_and_failure_preserve_source_safety(monkeypatch) -> None:
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 7)

    success, resources, backend = make_worker(source_ready_event_factory=FakeEvent)
    begin_step(success, store=StoreCommandBatch((command,)))
    success.finish_step()
    completions = success.fence_previous_store()
    assert completions[0].evidence.succeeded
    assert completions[0].evidence.source_release_confirmed
    assert success.take_released_store_job_ids() == {7}
    success.close()
    assert resources.closed and backend.closed is False

    full_hit_events = []

    def make_full_hit_event():
        event = FakeEvent()
        full_hit_events.append(event)
        return event

    full_hit, full_hit_resources, full_hit_backend = make_worker(
        requires_exists_before_put=True, source_ready_event_factory=make_full_hit_event
    )

    def reject_full_batch(*_args, **_kwargs):
        raise AssertionError("Full-hit Store must not materialize a transfer batch")

    monkeypatch.setattr(full_hit, "_materialize_store_candidates", reject_full_batch)
    begin_step(full_hit, store=StoreCommandBatch((command,)))
    full_hit.finish_step()
    full_hit_completions = full_hit.fence_previous_store()
    assert full_hit_completions[0].evidence.succeeded
    assert full_hit.take_released_store_job_ids() == {7}
    assert [call[0] for call in full_hit_backend.calls].count("exists") == 1
    assert "put" not in [call[0] for call in full_hit_backend.calls]
    assert full_hit_events[0].recorded and not full_hit_events[0].synchronized
    full_hit.close()
    assert full_hit_resources.closed

    duplicate_backend = FakeBackend()
    duplicate_worker, duplicate_resources, _ = make_worker(duplicate_backend, source_ready_event_factory=FakeEvent)
    duplicate_command = replace(command, request_id="duplicate", block_ids_by_group=((3,),), store_job_id=8)
    begin_step(duplicate_worker, store=StoreCommandBatch((command, duplicate_command)))
    duplicate_worker.finish_step()
    duplicate_worker.fence_previous_store()
    put_call = next(call for call in duplicate_backend.calls if call[0] == "put")
    assert len(put_call[1]) == 1
    assert duplicate_worker.take_released_store_job_ids() == {7, 8}
    duplicate_worker.close()
    assert duplicate_resources.closed

    admission_backend = FakeBackend()

    def fail_admission(_keys):
        raise RuntimeError("exists failed")

    monkeypatch.setattr(admission_backend, "exists", fail_admission)
    admission_failure, admission_resources, _ = make_worker(
        admission_backend, requires_exists_before_put=True, source_ready_event_factory=FakeEvent
    )
    begin_step(admission_failure, store=StoreCommandBatch((command,)))
    admission_failure.finish_step()
    completions = admission_failure.fence_previous_store()
    assert not completions[0].evidence.succeeded
    assert completions[0].evidence.source_release_confirmed
    assert admission_failure.take_released_store_job_ids() == {7}
    admission_failure.close()
    assert admission_resources.closed

    failed_backend = FakeBackend()
    failed_backend.put_result = [-1]
    failed, failed_resources, _ = make_worker(failed_backend, source_ready_event_factory=FakeEvent)
    begin_step(failed, store=StoreCommandBatch((command,)))
    failed.finish_step()
    completions = failed.fence_previous_store()
    assert not completions[0].evidence.succeeded
    assert completions[0].evidence.source_release_confirmed
    assert failed._pending_store_batch is None
    assert failed.take_released_store_job_ids() == {7}
    failed.close()
    assert failed_resources.closed


def test_bulk_store_argument_alignment_error_is_not_suppressed() -> None:
    worker, resources, backend = make_worker()
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 7)
    batch = build_admitted_store_batch(worker, (command,))
    assert batch is not None
    arguments = worker._materialize_concrete_bulk_arguments(batch, store=True)

    try:
        with pytest.raises(RuntimeError, match="Bulk Store arguments do not align"):
            worker_backend_io(worker).store_materialized(batch, replace(arguments, keys=[]))
        assert "put" not in [call[0] for call in backend.calls]
    finally:
        worker.close()
    assert resources.closed


def test_bulk_store_candidates_compact_partial_admission_before_materialization() -> None:
    backend = FakeBackend()
    backend.presence = [1, 0]
    worker, resources, _ = make_worker(backend, requires_exists_before_put=True, source_ready_event_factory=FakeEvent)
    present = RangeStoreCommand("present", TokenRange(0, 4), ((1,),), (b"a",), 4, 7)
    missing = RangeStoreCommand("missing", TokenRange(0, 4), ((3,),), (b"b",), 4, 8)

    begin_step(worker, store=StoreCommandBatch((present, missing)))
    worker.finish_step()
    completions = worker.fence_previous_store()

    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 1
    assert put_call[2] == ((1192, 2192),)
    assert [len(completion.evidence.transfer_evidence) for completion in completions] == [0, 1]
    assert worker.take_released_store_job_ids() == {7, 8}
    worker.close()
    assert resources.closed


@pytest.mark.parametrize("layerwise", [False, True])
def test_unified_store_candidates_compact_each_group_before_timeline_selection(layerwise: bool) -> None:
    topology = make_topology(group_ids=(0, 1))
    backend = FakeBackend()
    backend.presence = [1, 0, 0, 1]
    worker, resources, _ = make_worker(
        backend,
        topology=topology,
        layerwise=layerwise,
        requires_exists_before_put=True,
    )
    command = RangeStoreCommand(
        "request",
        TokenRange(0, 8),
        ((1, 2), (3, 4)),
        (b"a", b"b"),
        8,
        17,
    )

    batch = build_admitted_store_batch(worker, (command,), layerwise=layerwise)

    assert batch is not None
    assert [group.block_ids.tolist() for group in batch.groups] == [[2], [3]]
    assert [group.token_counts.tolist() for group in batch.groups] == [[4], [4]]
    assert [group.request_splits.tolist() for group in batch.groups] == [[0, 1], [0, 1]]
    assert all(group.selected_objects is None for group in batch.groups)
    assert len(batch.selected_keys()) == 2
    assert batch.selected_keys() == tuple(group.selected_keys()[0] for group in batch.groups)
    if layerwise:
        worker_backend_io(worker).store_batch(batch, layer_id=0)
        copy_call = next(call for call in backend.calls if call[0] == "batch_copy_put")
        assert copy_call[1] == batch.selected_keys()
        assert copy_call[2] == ((1128,), (11192,))
    else:
        arguments = worker._materialize_concrete_bulk_arguments(batch, store=True)
        worker_backend_io(worker).store_materialized(batch, arguments)
        put_call = next(call for call in backend.calls if call[0] == "put")
        assert put_call[1] == batch.selected_keys()
        assert put_call[2] == ((1128, 2128), (11192, 12192))
    worker.close()
    assert resources.closed


@pytest.mark.parametrize(
    ("topology", "presence", "expected_selection"),
    [
        (make_topology(tp_mismatch=True), [1, 0], [False, True]),
        (make_topology(consumer_pipeline_partitions=(1, 1)), [0, 1], [True, False]),
    ],
)
def test_unified_store_candidates_preserve_object_axis_admission(topology, presence, expected_selection) -> None:
    backend = FakeBackend()
    backend.presence = presence
    worker, resources, _ = make_worker(
        backend,
        topology=topology,
        requires_exists_before_put=True,
    )
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)

    batch = build_admitted_store_batch(worker, (command,))

    assert batch is not None
    group = batch.groups[0]
    assert group.block_ids.tolist() == [1]
    assert group.selected_objects is not None
    assert group.selected_objects.tolist() == expected_selection
    assert batch.selected_keys() == group.selected_keys()
    assert len(batch.selected_keys()) == 1
    worker.close()
    assert resources.closed


def test_unified_store_candidates_defer_hybrid_checkpoint_materialization() -> None:
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
            MambaSpec(
                block_size=8,
                shapes=((1,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
            (KVPoolLayerTopology(1, ("layers.1.state",)),),
            base.groups[1].key_metadata,
        ),
    )
    topology = replace(base, cache_transfer_granularity=8, hash_block_size=4, groups=groups)
    backend = FakeBackend()
    backend.presence = [1, 0]
    worker, resources, _ = make_worker(
        backend,
        topology=topology,
        requires_exists_before_put=True,
    )
    command = CheckpointStoreCommand(
        "checkpoint",
        ((3,), (99,)),
        (b"a",),
        0,
        (StateCheckpointSource(1, 7, 4),),
        23,
    )

    batch = build_admitted_store_batch(worker, (command,))

    assert batch is not None
    assert [group.block_ids.tolist() for group in batch.groups] == [[], [7]]
    assert [group.request_splits.tolist() for group in batch.groups] == [[0, 0], [0, 1]]
    assert len(batch.selected_keys()) == 1
    assert "@group:1@" in batch.selected_keys()[0]
    assert next(call for call in backend.calls if call[0] == "exists")[1] != batch.selected_keys()
    arguments = worker._materialize_concrete_bulk_arguments(batch, store=True)
    worker_backend_io(worker).store_materialized(batch, arguments)
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert put_call[1] == batch.selected_keys()
    assert put_call[2] == ((12448,),)
    assert put_call[3] == ((32,),)
    worker.close()
    assert resources.closed


def test_layerwise_load_happy_path_reuses_rows_and_closes_one_session() -> None:
    worker, resources, backend = make_worker(
        layerwise=True, store=False, start_gate_factory=lambda: FakeLoadStartGate(opened=True)
    )
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=LoadCommandBatch((command,)))

    worker.start_load()
    worker.wait_for_layer_load("layers.0.group.0")
    worker.wait_for_layer_load("layers.1.group.0")
    worker.close()

    session_calls = [call for call in backend.calls if call[0] in ("batch_get_start", "batch_get_end")]
    copy_calls = [call for call in backend.calls if call[0] == "batch_copy_get"]
    assert [call[0] for call in session_calls] == ["batch_get_start", "batch_get_end"]
    assert [call[2] for call in copy_calls] == [((1064,),), ((2064,),)]
    assert [call[4] for call in copy_calls] == [((0,),), ((32,),)]
    assert resources.closed


def test_key_range_layerwise_load_reuses_lowered_execution_state_across_layers(monkeypatch) -> None:
    worker, resources, backend = make_worker(
        layerwise=True,
        store=False,
        physical_layers=(0, 1, 2),
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=LoadCommandBatch((command,)))
    worker.start_load()

    backend_io = worker_backend_io(worker)
    prepared_plan = backend_io._load_plan
    assert prepared_plan is not None
    prepared_groups = [prepared_plan.groups_by_layer[layer_id] for layer_id in range(3)]
    prepared_keys = [prepared_plan.keys_by_layer[layer_id] for layer_id in range(3)]
    assert prepared_groups[0][0] is prepared_groups[1][0] is prepared_groups[2][0]
    assert prepared_groups[0][0].block_ids is prepared_plan.groups[0].block_ids
    assert prepared_groups[0][0].token_counts is prepared_plan.groups[0].token_counts
    assert prepared_keys[0] is prepared_keys[1] is prepared_keys[2]

    direct_calls = []
    direct_projection = arguments_module.key_range_layer_ranges

    def record_direct_projection(group, block_ids, token_counts, *, layer_id):
        direct_calls.append(layer_id)
        return direct_projection(group, block_ids, token_counts, layer_id=layer_id)

    def reject_rebuild(*_args, **_kwargs):
        raise AssertionError("Layerwise Load rebuilt cross-layer execution state")

    monkeypatch.setattr(arguments_module, "key_range_layer_ranges", record_direct_projection)
    monkeypatch.setattr(KVTransferBatch, "for_layer", reject_rebuild)
    monkeypatch.setattr(KVGroupBatch, "selected_keys", reject_rebuild)

    for layer_id in range(2):
        worker.wait_for_layer_load(f"layers.{layer_id}.group.0")
    assert backend_io._load_plan is prepared_plan
    worker.wait_for_layer_load("layers.2.group.0")
    assert direct_calls == [0, 1, 2]
    assert backend_io._load_plan is None
    worker.close()

    assert [call[0] for call in backend.calls].count("batch_copy_get") == 3
    assert resources.closed


def test_layerwise_load_failure_cleans_sessions_and_preserves_block_identity() -> None:
    backend = FakeBackend()
    backend.session_copy_result = [-1]
    worker, resources, _ = make_worker(backend, layerwise=True, store=False, physical_layers=(0,))
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
    begin_step(worker, load=LoadCommandBatch((command,)))

    worker.start_load()
    worker.wait_for_layer_load("layers.0.group.0")
    result = worker.collect_load_result()
    assert result.failed_block_ids == frozenset({7})
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    worker.close()
    assert resources.closed


def test_layerwise_load_keeps_a_bounded_prefetch_window() -> None:
    gates: list[FakeLoadStartGate] = []

    def make_gate():
        gate = FakeLoadStartGate()
        gates.append(gate)
        return gate

    worker, _, backend = make_worker(
        layerwise=True, store=False, physical_layers=(0, 1, 2), start_gate_factory=make_gate
    )
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=LoadCommandBatch((command,)))

    worker.start_load()
    assert not any(call[0] == "batch_copy_get" for call in backend.calls)
    worker.wait_for_layer_load("layers.0.group.0")
    assert [call[0] for call in backend.calls].count("batch_copy_get") == 1
    gates[0].open()
    worker.wait_for_layer_load("layers.1.group.0")
    assert [call[0] for call in backend.calls].count("batch_copy_get") == 2
    gates[1].open()
    worker.wait_for_layer_load("layers.2.group.0")
    worker.close()
    assert [call[0] for call in backend.calls].count("batch_copy_get") == 3


def test_layerwise_load_start_exception_closes_attempted_session_once(monkeypatch) -> None:
    backend = FakeBackend()

    def fail_start(keys):
        backend.calls.append(("batch_get_start", tuple(keys)))
        raise RuntimeError("session start failed")

    monkeypatch.setattr(backend, "batch_get_start", fail_start)
    worker, resources, _ = make_worker(backend, layerwise=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=LoadCommandBatch((command,)))

    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        worker.start_load()
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        worker.close()
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    assert resources.closed


def test_layerwise_store_commits_only_after_all_layers_and_unknown_failure_retains_source() -> None:
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    worker, resources, backend = make_worker(layerwise=True)
    begin_step(worker, store=StoreCommandBatch((command,)))

    worker.save_layer("layers.0.group.0")
    assert worker._layerwise_store_timeline is not None
    worker._layerwise_store_timeline._executor._queue.join()
    assert [call[0] for call in backend.calls].count("batch_copy_put") == 1
    assert "batch_commit" not in [call[0] for call in backend.calls]

    worker.save_layer("layers.1.group.0")
    worker.finish_step()
    assert [call[0] for call in backend.calls].count("batch_commit") == 1
    assert worker.take_released_store_job_ids() == {17}
    worker.close()
    assert resources.closed

    failing_backend = FakeBackend()
    failing_backend.store_session_copy_result = RuntimeError("copy failed")
    failed, failed_resources, _ = make_worker(failing_backend, layerwise=True)
    duplicate = RangeStoreCommand("duplicate", TokenRange(0, 4), ((3,),), (b"a",), 4, 18)
    begin_step(failed, store=StoreCommandBatch((command, duplicate)))
    failed.save_layer("layers.0.group.0")
    failed.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store source release is unknown"):
        failed.finish_step()
    assert failed._pending_store_batch is not None
    assert not failed._pending_store_batch.completions[0].evidence.source_release_confirmed
    assert failed._pending_store_batch.completions[1].evidence.source_release_confirmed
    assert failed.take_released_store_job_ids() == {18}
    assert "batch_commit" not in [call[0] for call in failing_backend.calls]
    assert "batch_revoke" in [call[0] for call in failing_backend.calls]
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        failed.close()
    assert not failed_resources.closed


def test_layerwise_store_reuses_fully_started_batch_and_filters_failed_sessions(monkeypatch) -> None:
    original_select_keys = KVTransferBatch.select_keys
    selections = []

    def record_selection(self, accepted, *, claim_once=False):
        selections.append((accepted, claim_once))
        return original_select_keys(self, accepted, claim_once=claim_once)

    monkeypatch.setattr(KVTransferBatch, "select_keys", record_selection)
    first = RangeStoreCommand("first", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    second = RangeStoreCommand("second", TokenRange(0, 4), ((3,),), (b"b",), 4, 18)

    worker, resources, _ = make_worker(layerwise=True)
    begin_step(worker, store=StoreCommandBatch((first, second)))
    worker.save_layer("layers.0.group.0")
    worker.save_layer("layers.1.group.0")
    worker.finish_step()
    assert selections == []
    worker.close()
    assert resources.closed

    backend = FakeBackend()
    backend.store_session_start_result = [0, -1]
    failed, failed_resources, _ = make_worker(backend, layerwise=True)
    begin_step(failed, store=StoreCommandBatch((first, second)))
    failed.save_layer("layers.0.group.0")
    failed.save_layer("layers.1.group.0")
    failed.finish_step()

    started_keys = next(call[1] for call in backend.calls if call[0] == "batch_put_start")
    copied_keys = [call[1] for call in backend.calls if call[0] == "batch_copy_put"]
    committed_keys = next(call[1] for call in backend.calls if call[0] == "batch_commit")
    assert selections == [({started_keys[0]}, True)]
    assert copied_keys == [(started_keys[0],), (started_keys[0],)]
    assert committed_keys == (started_keys[0],)
    assert failed.take_released_store_job_ids() == {17, 18}
    failed.close()
    assert failed_resources.closed


def test_layerwise_store_reuses_lowered_execution_state_across_layers(monkeypatch) -> None:
    worker, resources, backend = make_worker(layerwise=True, physical_layers=(0, 1, 2))
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    begin_step(worker, store=StoreCommandBatch((command,)))
    assert worker._layerwise_store_timeline is not None
    worker._layerwise_store_timeline._executor._queue.join()

    backend_io = worker_backend_io(worker)
    prepared_plan = backend_io._store_plan
    assert prepared_plan is not None
    session = worker._layerwise_store_timeline._session
    assert session is not None and session.selected is not None
    assert session.selected.layer_store_plan is prepared_plan
    prepared_groups = [prepared_plan.groups_by_layer[layer_id] for layer_id in range(3)]
    prepared_keys = [prepared_plan.keys_by_layer[layer_id] for layer_id in range(3)]
    assert prepared_groups[0][0] is prepared_groups[1][0] is prepared_groups[2][0]
    assert prepared_groups[0][0].block_ids is session.selected.groups[0].block_ids
    assert prepared_groups[0][0].token_counts is session.selected.groups[0].token_counts
    assert prepared_keys[0] is prepared_keys[1] is prepared_keys[2]

    direct_calls = []
    direct_projection = arguments_module.key_range_layer_ranges

    def record_direct_projection(group, block_ids, token_counts, *, layer_id):
        direct_calls.append(layer_id)
        return direct_projection(group, block_ids, token_counts, layer_id=layer_id)

    def reject_rebuild(*_args, **_kwargs):
        raise AssertionError("Layerwise Store rebuilt cross-layer execution state")

    monkeypatch.setattr(arguments_module, "key_range_layer_ranges", record_direct_projection)
    monkeypatch.setattr(KVTransferBatch, "for_layer", reject_rebuild)
    monkeypatch.setattr(KVGroupBatch, "selected_keys", reject_rebuild)

    for layer_id in range(3):
        worker.save_layer(f"layers.{layer_id}.group.0")
    worker.finish_step()

    assert direct_calls == [0, 1, 2]
    assert [call[0] for call in backend.calls].count("batch_copy_put") == 3
    assert worker.take_released_store_job_ids() == {17}
    worker.close()
    assert resources.closed


def test_layerwise_non_leader_discards_store_rows_in_worker_business_path() -> None:
    topology = make_topology(tp_rank=1, tp_size=2, put_step=2)
    worker, resources, _ = make_worker(topology=topology, layerwise=True)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)

    assert worker._layerwise_store_leader is False
    assert worker._store_candidate_rows(command) == (([], [], []),)

    worker.close()
    assert resources.closed


def test_layerwise_store_prepares_asynchronously_behind_layer_jobs(monkeypatch) -> None:
    backend = FakeBackend()
    admission_started = threading.Event()
    release_admission = threading.Event()

    def blocking_exists(keys):
        backend.calls.append(("exists", tuple(keys)))
        admission_started.set()
        if not release_admission.wait(timeout=5):
            raise TimeoutError("test did not release Layerwise admission")
        return [0] * len(keys)

    monkeypatch.setattr(backend, "exists", blocking_exists)
    worker, resources, _ = make_worker(backend, layerwise=True, requires_exists_before_put=True)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    begin_completed = threading.Event()
    begin_errors = []

    def begin() -> None:
        try:
            begin_step(worker, store=StoreCommandBatch((command,)))
        except BaseException as error:
            begin_errors.append(error)
        finally:
            begin_completed.set()

    caller = threading.Thread(target=begin)
    caller.start()
    try:
        assert admission_started.wait(timeout=2)
        returned_before_admission = begin_completed.wait(timeout=0.2)
        assert returned_before_admission
        worker.save_layer("layers.0.group.0")
    finally:
        release_admission.set()
        caller.join(timeout=5)

    assert not begin_errors
    worker.save_layer("layers.1.group.0")
    worker.finish_step()
    assert [
        call[0] for call in backend.calls if call[0] in {"exists", "batch_put_start", "batch_copy_put", "batch_commit"}
    ] == [
        "exists",
        "batch_put_start",
        "batch_copy_put",
        "batch_copy_put",
        "batch_commit",
    ]
    worker.close()
    assert resources.closed


def test_layerwise_store_materializes_only_missing_rows_and_skips_full_hits(monkeypatch) -> None:
    command = RangeStoreCommand("present", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    missing = RangeStoreCommand("missing", TokenRange(0, 4), ((3,),), (b"b",), 4, 18)
    backend = FakeBackend()
    backend.presence = [1, 0]
    events = []

    def make_event():
        event = FakeEvent()
        events.append(event)
        return event

    worker, resources, _ = make_worker(
        backend, layerwise=True, requires_exists_before_put=True, source_ready_event_factory=make_event
    )
    begin_step(worker, store=StoreCommandBatch((command, missing)))
    worker.save_layer("layers.0.group.0")
    worker.save_layer("layers.1.group.0")
    worker.finish_step()

    start = next(call for call in backend.calls if call[0] == "batch_put_start")
    copies = [call for call in backend.calls if call[0] == "batch_copy_put"]
    assert len(start[1]) == 1
    assert [call[1] for call in copies] == [start[1], start[1]]
    assert [call[2] for call in copies] == [((1192,),), ((2192,),)]
    assert all(event.recorded and event.synchronized for event in events)
    assert worker.take_released_store_job_ids() == {17, 18}
    worker.close()
    assert resources.closed

    full_hit_backend = FakeBackend()
    full_hit_events = []

    def make_full_hit_event():
        event = FakeEvent()
        full_hit_events.append(event)
        return event

    full_hit, full_hit_resources, _ = make_worker(
        full_hit_backend,
        layerwise=True,
        requires_exists_before_put=True,
        source_ready_event_factory=make_full_hit_event,
    )

    def reject_batch(*_args, **_kwargs):
        raise AssertionError("Full-hit Layerwise Store must not materialize a transfer batch")

    monkeypatch.setattr(full_hit, "_materialize_store_candidates", reject_batch)
    begin_step(full_hit, store=StoreCommandBatch((command,)))
    full_hit.save_layer("layers.0.group.0")
    full_hit.save_layer("layers.1.group.0")
    full_hit.finish_step()
    assert [call[0] for call in full_hit_backend.calls].count("exists") == 1
    assert not any(call[0].startswith("batch_") for call in full_hit_backend.calls)
    assert all(event.recorded and not event.synchronized for event in full_hit_events)
    assert full_hit.take_released_store_job_ids() == {17}
    full_hit.close()
    assert full_hit_resources.closed


def test_layerwise_store_aggregates_terminal_evidence_across_layers() -> None:
    worker, resources, _ = make_worker(layerwise=True)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    begin_step(worker, store=StoreCommandBatch((command,)))
    worker.save_layer("layers.0.group.0")
    worker.save_layer("layers.1.group.0")

    context = worker._active_step
    assert context is not None
    worker._finalize_layerwise_store(context)
    completions = worker.fence_previous_store()

    assert len(completions) == 1
    evidence = completions[0].evidence.transfer_evidence
    assert len(evidence) == 1
    assert evidence[0].source.physical_layer_ids == (0, 1)
    assert evidence[0].result_code == 0
    assert evidence[0].source_release_confirmed

    worker.end_step()
    worker.close()
    assert resources.closed


def test_worker_close_reports_incomplete_layerwise_store_but_releases_safe_source() -> None:
    worker, resources, backend = make_worker(layerwise=True)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    begin_step(worker, store=StoreCommandBatch((command,)))
    worker.save_layer("layers.0.group.0")

    worker.close()

    assert worker.take_released_store_job_ids() == {17}
    assert worker._pending_store_batch is None
    assert resources.closed
    assert "batch_revoke" in [call[0] for call in backend.calls]
    assert "batch_commit" not in [call[0] for call in backend.calls]


def test_worker_owns_active_step_and_async_completion_lifecycle() -> None:
    worker, _, _ = make_worker(async_load=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))

    with pytest.raises(RuntimeError, match="has not begun"):
        worker.start_load()
    begin_step(worker, load=LoadCommandBatch((command,)))
    with pytest.raises(RuntimeError, match="has not ended"):
        worker.begin_step(KVTransferStep())
    worker.start_load()
    worker.end_step()
    begin_step(worker)
    assert isinstance(worker._asynchronous_load_timeline, AsynchronousLoadTimeline)
    worker._asynchronous_load_timeline._executor._queue.join()
    result = worker.collect_load_result()
    assert result.completed_request_ids == {"request"}
    assert worker.collect_load_result().completed_request_ids == set()
    worker.end_step()
    worker.close()


def test_worker_selects_concrete_timelines_from_bound_configuration() -> None:
    synchronous, synchronous_resources, _ = make_worker(store=False)
    asynchronous, asynchronous_resources, _ = make_worker(async_load=True)
    layerwise, layerwise_resources, _ = make_worker(layerwise=True)

    try:
        assert isinstance(synchronous._synchronous_load_timeline, SynchronousLoadTimeline)
        assert synchronous._asynchronous_store_timeline is None
        assert isinstance(asynchronous._asynchronous_load_timeline, AsynchronousLoadTimeline)
        assert isinstance(asynchronous._asynchronous_store_timeline, AsynchronousStoreTimeline)
        assert isinstance(layerwise._layerwise_load_timeline, LayerwiseLoadTimeline)
        assert isinstance(layerwise._layerwise_store_timeline, LayerwiseStoreTimeline)
    finally:
        synchronous.close()
        asynchronous.close()
        layerwise.close()
    assert synchronous_resources.closed
    assert asynchronous_resources.closed
    assert layerwise_resources.closed


def test_async_load_failure_drains_pending_work_and_rejects_overlap(monkeypatch) -> None:
    backend = FakeBackend()

    def fail_get(*_args):
        raise RuntimeError("backend get failed")

    monkeypatch.setattr(backend, "get", fail_get)
    worker, _, _ = make_worker(backend, async_load=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=LoadCommandBatch((command,)))
    worker.start_load()
    worker.end_step()

    begin_step(worker, load=LoadCommandBatch((replace(command, request_id="other"), command)))
    with pytest.raises(RuntimeError, match="already has a pending asynchronous Load"):
        worker.start_load()
    assert worker._pending_load_request_ids == {"request"}
    assert isinstance(worker._asynchronous_load_timeline, AsynchronousLoadTimeline)
    worker._asynchronous_load_timeline._executor._queue.join()
    result = worker.collect_load_result()
    assert result.completed_request_ids == {"request"}
    assert result.failed_block_ids == {1}
    assert [(location.group_id, location.block_id) for location in result.failed_locations] == [(0, 1)]
    worker.close()
