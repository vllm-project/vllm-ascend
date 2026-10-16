"""Layerwise copies, publication barriers, source safety, and bounded prefetch."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeLoadStartGate,
    begin_step,
    make_backend_spec,
    make_topology,
    make_worker,
)
from tests.ut.distributed.ascend_store.v1.worker.bulk_fixtures import (
    make_align_state_topology,
    make_multi_spec_caches,
    make_sparse_group_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.gva_fixtures import (
    FakeGVABackend,
    make_gva_spec,
    make_gva_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    GVALayerwiseProjectionBinder,
    KeyRangeLayerwiseProjectionBinder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupRequest,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import resolve_group_layers
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import base as worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.io import gva as gva_io
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.io import key_range as key_range_io
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    GVALayerwiseWorker,
    KeyRangeLayerwiseWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.resources import (
    GVAObjectLayout,
    KVPoolResources,
)


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_layerwise_load_with_no_selected_objects_does_not_open_sessions(use_gva, monkeypatch) -> None:
    worker, resources, _ = make_gva_worker() if use_gva else make_worker(layerwise=True)

    def unexpected_session(command):
        pytest.fail("A Load with no selected objects must not open a Backend session")

    monkeypatch.setattr(worker._load_executor, "submit", unexpected_session)
    try:
        command = LoadCommand("request", TokenRange(0, 0), ((),), ())
        begin_step(worker, load=(command,))
        worker.start_load()
        assert not worker.collect_load_result().failed_locations
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
@pytest.mark.parametrize("failure_stage", (None, "prepare", "synchronize"))
def test_layerwise_store_prepares_before_wait_and_reads_only_ready_sources(use_gva, failure_stage, monkeypatch) -> None:
    trace = []

    class TraceEvent(FakeEvent):
        def record(self) -> None:
            trace.append("record")
            super().record()

        def synchronize(self) -> None:
            trace.append("wait")
            if failure_stage == "synchronize":
                raise RuntimeError("source readiness failed")
            super().synchronize()

    event = TraceEvent()
    topology = make_topology(physical_layers=(0,))
    module: ModuleType
    if use_gva:
        worker, resources, backend = make_gva_worker(topology=topology, source_ready_event_factory=lambda: event)
        module, materializer = gva_io, "materialize_store_layer_gva"
        copy_method = "batch_copy"
    else:
        worker, resources, backend = make_worker(
            layerwise=True, topology=topology, source_ready_event_factory=lambda: event
        )
        module, materializer = key_range_io, "materialize_store_layer_ranges"
        copy_method = "batch_copy_put"
    prepare = getattr(module, materializer)
    copy = getattr(backend, copy_method)

    def prepare_arguments(*args):
        trace.append("prepare")
        assert not event.synchronized
        if failure_stage == "prepare":
            raise RuntimeError("argument preparation failed")
        return prepare(*args)

    def copy_ready_source(*args):
        assert event.synchronized
        trace.append("copy")
        return copy(*args)

    monkeypatch.setattr(module, materializer, prepare_arguments)
    monkeypatch.setattr(backend, copy_method, copy_ready_source)
    try:
        command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
        begin_step(worker, store=(command,))
        worker.save_layer("layers.0.group.0")
        worker.finish_step()
        expected = ["record", "prepare"]
        if failure_stage != "prepare":
            expected.append("wait")
        if failure_stage is None:
            expected.append("copy")
        assert trace == expected
        assert worker.take_released_store_job_ids() == {17}
        assert any(call[0] in ("publish", "batch_commit") for call in backend.calls) == (failure_stage is None)
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("with_load", (False, True), ids=("cold", "external_load"))
def test_key_range_hybrid_starts_store_before_attention_without_duplicate_saves(with_load, monkeypatch) -> None:
    monkeypatch.setattr(attention_fence, "_attention_compute_start_gate", None)
    monkeypatch.setattr(torch.npu, "Event", Mock)
    topology = make_topology(group_ids=(0, 1))
    worker, resources, backend = make_worker(layerwise=True, topology=topology)
    store = RangeStoreCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"b",), 4, 17)
    load = (LoadCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",)),) if with_load else ()
    try:
        begin_step(worker, load=load, store=(store,))
        assert worker._session is None
        worker.start_load()
        session = worker._session
        assert worker._store_executor is not None
        for layer_id in (0, 1):
            layer_name = f"layers.{layer_id}.group.0"
            worker.wait_for_layer_load(layer_name)
            if session is None:
                session = worker._session
            assert worker._session is session
            gate = attention_fence.get_attention_compute_start_gate()
            worker.wait_for_layer_load(f"layers.{layer_id}.group.1")
            assert attention_fence.get_attention_compute_start_gate() is gate
            assert sum(call[0] == "batch_copy_put" for call in backend.calls) == layer_id
            with attention_fence.attention_transfer_window():
                # This point models attention compute: KV is ready, but the post-compute hook has not run.
                worker._store_executor.submit(lambda: None).result()
                assert sum(call[0] == "batch_copy_put" for call in backend.calls) == layer_id + 1
            worker.save_layer(layer_name)
            worker.save_layer(f"layers.{layer_id}.group.1")
        worker.finish_step()
        assert sum(call[0] == "batch_copy_put" for call in backend.calls) == 2
        assert sum(call[0] == "batch_copy_get" for call in backend.calls) == (2 if with_load else 0)
        assert sum(call[0] == "batch_commit" for call in backend.calls) == 1
        assert not worker.collect_load_result().failed_locations
        assert worker.take_released_store_job_ids() == {17}
        gate = attention_fence.get_attention_compute_start_gate()
        worker.end_step()
        assert gate.on_start is None
        assert worker._session is None
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("uniform_state_group", (False, True), ids=("mamba", "uniform_mamba"))
def test_hybrid_recurrent_layer_without_range_store_does_not_record_source(uniform_state_group, monkeypatch) -> None:
    monkeypatch.setattr(attention_fence, "_attention_compute_start_gate", None)
    monkeypatch.setattr(torch.npu, "Event", Mock)
    topology = make_align_state_topology(attention_block_size=4, state_block_size=4, tp_size=1)
    group = topology.groups[1]
    state_spec = replace(group.kv_cache_spec, mamba_cache_mode="all")
    if uniform_state_group:
        state_spec = UniformTypeKVCacheSpecs.from_specs({group.layer_names[0]: state_spec})
    topology = replace(topology, groups=(topology.groups[0], replace(group, kv_cache_spec=state_spec)))
    source_events = []

    def record_source():
        event = FakeEvent()
        source_events.append(event)
        return event

    worker, resources, backend = make_worker(
        layerwise=True, topology=topology, source_ready_event_factory=record_source
    )
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",), 4, 17)
    try:
        begin_step(worker, store=(command,))
        worker.start_load()
        assert worker._store_executor is not None
        for layer_id, name in enumerate(("layers.0.attention", "layers.1.state")):
            worker.wait_for_layer_load(name)
            with attention_fence.attention_transfer_window():
                worker._store_executor.submit(lambda: None).result()
                assert len(source_events) == 1, "Recurrent KV is not ready at attention entry"
            worker.save_layer(name)
            worker._store_executor.submit(lambda: None).result()
            assert len(source_events) == 1, "Range Store has no recurrent task to record or submit"
        worker.finish_step()
        # Range Store excludes recurrent state; this change must not enable checkpoint transfer.
        assert sum(call[0] == "batch_copy_put" for call in backend.calls) == 1
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("use_gva", (False, True), ids=("single_group_key_range", "hybrid_gva"))
def test_other_layerwise_routes_keep_post_compute_store(use_gva, monkeypatch) -> None:
    monkeypatch.setattr(attention_fence, "_attention_compute_start_gate", None)
    topology = make_topology(group_ids=(0, 1) if use_gva else (0,), physical_layers=(0,))
    if use_gva:
        worker, resources, backend = make_gva_worker(topology=topology)
    else:
        worker, resources, backend = make_worker(layerwise=True, topology=topology)
    blocks = tuple((index + 1,) for index in range(len(topology.groups)))
    command = RangeStoreCommand("request", TokenRange(0, 4), blocks, (b"a",), 4, 17)
    try:
        begin_step(worker, store=(command,))
        worker.wait_for_layer_load("layers.0.group.0")
        with attention_fence.attention_transfer_window():
            assert not any(call[0] in ("copy", "batch_copy_put") for call in backend.calls)
        worker.save_layer("layers.0.group.0")
        worker.finish_step()
        assert sum(call[0] in ("copy", "batch_copy_put") for call in backend.calls) == 1
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("finish", ("fallback", "end_step", "close"))
def test_hybrid_attention_store_callback_cannot_outlive_its_step(finish, monkeypatch) -> None:
    monkeypatch.setattr(attention_fence, "_attention_compute_start_gate", None)
    topology = make_topology(group_ids=(0, 1), physical_layers=(0,))
    worker, resources, backend = make_worker(layerwise=True, topology=topology)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",), 4, 17)
    try:
        begin_step(worker, store=(command,))
        worker.wait_for_layer_load("layers.0.group.0")
        gate = attention_fence.get_attention_compute_start_gate()
        assert gate.on_start is not None
        if finish == "fallback":
            # Some attention implementations do not expose the early window.
            worker.save_layer("layers.0.group.0")
            worker.finish_step()
            assert sum(call[0] == "batch_copy_put" for call in backend.calls) == 1
        if finish != "close":
            worker.end_step()
        else:
            worker.close()
        assert gate.on_start is None
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
@pytest.mark.parametrize(
    "retention_interval,stored_swa_boundaries,middle_end", ((0, (80, 96), 0), (64, (48, 64, 80, 96), 80))
)
def test_layerwise_lookup_recovers_sparse_objects_after_publication(
    use_gva, retention_interval, stored_swa_boundaries, middle_end
) -> None:
    class PublishedKeyRangeBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.readable_keys: set[str] = set()

        def exists(self, keys):
            self.calls.append(("exists", tuple(keys)))
            return [int(key in self.readable_keys) for key in keys]

        def commit_key_range_store(self, keys):
            codes = super().commit_key_range_store(keys)
            self.readable_keys.update(key for key, code in zip(keys, codes, strict=True) if code == 0)
            return codes

    topology = make_sparse_group_topology(block_size=16, sliding_window=33)
    backend: FakeGVABackend | PublishedKeyRangeBackend = FakeGVABackend() if use_gva else PublishedKeyRangeBackend()
    backend_spec = make_gva_spec() if use_gva else make_backend_spec(requires_exists_before_put=True)
    resources = KVPoolResources(
        backend,
        backend_spec,
        8,
        topology.transfer_groups,
        gva_layout=GVAObjectLayout(0, 2, 0, 1, 1) if use_gva else None,
    )
    binder_type = GVALayerwiseProjectionBinder if use_gva else KeyRangeLayerwiseProjectionBinder
    binder = binder_type(
        topology, 256, lambda group, value, head, stage: f"g{group}:{value}", retention_interval=retention_interval
    )
    worker_type = GVALayerwiseWorker if use_gva else KeyRangeLayerwiseWorker
    worker = worker_type(topology, binder, resources, source_ready_event_factory=FakeEvent)
    block_hashes = tuple(bytes([index]) for index in range(1, 9))
    caches = {group.layer_names[0]: torch.zeros((8, 2, 16, 1, 1)) for group in topology.transfer_groups}
    try:
        worker.bind_kv_caches(caches)
        command = RangeStoreCommand(
            "producer", TokenRange(0, 96), ((), (1, 2, 3, 4, 5, 6), (), (1, 2, 3, 4, 5, 6)), block_hashes[:6], 97, 17
        )
        begin_step(worker, store=(command,))
        for group in topology.transfer_groups:
            worker.save_layer(group.layer_names[0])
        worker.finish_step()
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
        if isinstance(backend, FakeGVABackend):
            assert backend.native_store is not None
            readable_keys = {key for key, region in backend.native_store.objects.items() if region[2]}
        else:
            readable_keys = backend.readable_keys
        swa_keys = tuple(key for key in readable_keys if key.startswith("g3:"))
        assert sorted(int(key.split(":")[1], 16) * 16 for key in swa_keys) == list(stored_swa_boundaries)
        for start, end, expected_end in ((0, 128, 96), (0, 80, middle_end), (64, 128, 96)):
            result = worker.lookup(LookupRequest(TokenRange(start, end), (1, 3), block_hashes))
            assert result.available_end_token == expected_end
    finally:
        worker.close()
    assert backend.closed


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_layerwise_store_copies_all_cache_entries_at_each_physical_layer(use_gva) -> None:
    cache_config, caches = make_multi_spec_caches()
    base = make_topology(group_ids=(0, 2), physical_layers=(2, 3))
    topology = replace(
        base,
        cache_transfer_granularity=32,
        hash_block_size=32,
        groups=tuple(
            replace(
                group,
                kv_cache_spec=cache_config.kv_cache_groups[group.group_id].kv_cache_spec,
                layers=resolve_group_layers(cache_config.kv_cache_groups[group.group_id].layer_names, 4),
            )
            for group in base.groups
        ),
    )
    backend = FakeGVABackend() if use_gva else FakeBackend()
    backend_spec = make_gva_spec() if use_gva else make_backend_spec()
    resources = KVPoolResources(
        backend,
        backend_spec,
        cache_config.num_blocks,
        topology.transfer_groups,
        gva_layout=GVAObjectLayout(0, 4, 0, 1, 1) if use_gva else None,
    )
    binder_type = GVALayerwiseProjectionBinder if use_gva else KeyRangeLayerwiseProjectionBinder
    binder = binder_type(topology, 64, lambda group, value, head, stage: f"g{group}:{value}")
    worker_type = GVALayerwiseWorker if use_gva else KeyRangeLayerwiseWorker
    worker = worker_type(topology, binder, resources, source_ready_event_factory=FakeEvent)
    source_blocks = {0: 1, 2: 3}
    try:
        worker.bind_kv_caches(caches)
        command = RangeStoreCommand("request", TokenRange(0, 32), ((1,), (), (3,)), (b"a",), 32, 17)
        begin_step(worker, store=(command,))
        worker.save_layer("model.layers.2.attn")
        worker.save_layer("model.layers.3.swa_cache")
        worker.finish_step()
        assert worker.take_released_store_job_ids() == {17}
        if use_gva:
            assert isinstance(backend, FakeGVABackend) and backend.native_store is not None
            copies = [call for call in backend.native_store.calls if call[0] == "copy"]
            object_bases = {
                group.group_id: next(
                    region[0]
                    for key, region in backend.native_store.objects.items()
                    if key.startswith(f"g{group.group_id}:")
                )
                for group in topology.transfer_groups
            }
        else:
            copies = [call for call in backend.calls if call[0] == "batch_copy_put"]
        assert len(copies) == 2
        offsets = {0: 0, 2: 0}
        for layer_id, copy in zip((2, 3), copies, strict=True):
            addresses, sizes, remote_offsets = [], [], []
            for group in topology.transfer_groups:
                layer = next(layer for layer in group.layers if layer.physical_layer_id == layer_id)
                group_addresses, group_sizes, group_offsets = [], [], []
                for name in layer.layer_names:
                    block = caches[name][source_blocks[group.group_id]]
                    group_addresses.append(block.data_ptr())
                    group_sizes.append(block.numel() * block.element_size())
                    group_offsets.append(offsets[group.group_id])
                    offsets[group.group_id] += group_sizes[-1]
                addresses.append(tuple(group_addresses))
                sizes.append(tuple(group_sizes))
                remote_offsets.append(tuple(group_offsets))
            if use_gva:
                assert copy[1] == tuple(
                    object_bases[group.group_id] + offset
                    for group, row in zip(topology.transfer_groups, remote_offsets, strict=True)
                    for offset in row
                )
                assert copy[2] == tuple(address for row in addresses for address in row)
                assert copy[3] == tuple(size for row in sizes for size in row)
            else:
                assert copy[2:5] == (tuple(addresses), tuple(sizes), tuple(remote_offsets))
        assert worker.lookup(LookupRequest(TokenRange(0, 32), (0, 2), (b"a",))).available_end_token == 32
    finally:
        worker.close()
    assert backend.closed


@pytest.mark.parametrize(
    ("physical_layers", "expected_addresses"),
    (((0, 1, 2), (1064, 2064, 3064)), ((2, 5, 9), (3064, 6064, 10064))),
    ids=("contiguous", "sparse"),
)
def test_layerwise_load_copies_each_layer_within_one_session(physical_layers, expected_addresses) -> None:
    worker, resources, backend = make_worker(
        layerwise=True,
        store=False,
        physical_layers=physical_layers,
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=(command,))
    worker.start_load()
    for layer_id in physical_layers:
        worker.wait_for_layer_load(f"layers.{layer_id}.group.0")
        worker.wait_for_layer_load(f"layers.{layer_id}.group.0")
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    worker.close()

    session_calls = [call[0] for call in backend.calls if call[0] in ("batch_get_start", "batch_get_end")]
    copy_calls = [call for call in backend.calls if call[0] == "batch_copy_get"]
    assert session_calls == ["batch_get_start", "batch_get_end"]
    assert [call[2] for call in copy_calls] == [((address,),) for address in expected_addresses]
    assert [call[4] for call in copy_calls] == [((0,),), ((32,),), ((64,),)]
    assert resources.closed


@pytest.mark.parametrize("native_result", ([-1], RuntimeError("range copy failed")), ids=("failed", "exception"))
@pytest.mark.parametrize("physical_layer_id", (0, 5), ids=("contiguous", "sparse"))
def test_layerwise_load_failure_closes_session_and_reports_original_block(native_result, physical_layer_id) -> None:
    backend = FakeBackend()
    backend.session_copy_result = native_result
    worker, resources, _ = make_worker(backend, layerwise=True, store=False, physical_layers=(physical_layer_id,))
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
    try:
        begin_step(worker, load=(command,))
        worker.start_load()
        worker.wait_for_layer_load(f"layers.{physical_layer_id}.group.0")
        worker.wait_for_layer_load(f"layers.{physical_layer_id}.group.0")
        result = worker.collect_load_result()

        assert result.failed_block_ids == {7}
        assert [(location.group_id, location.block_id) for location in result.failed_locations] == [(0, 7)]
        assert [call[0] for call in backend.calls].count("batch_get_end") == 1
        worker.end_step()
    finally:
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
    begin_step(worker, load=(command,))

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
    original_start = backend.batch_get_start

    def fail_start(keys):
        backend.calls.append(("batch_get_start", tuple(keys)))
        raise RuntimeError("session start failed")

    monkeypatch.setattr(backend, "batch_get_start", fail_start)
    worker, resources, _ = make_worker(
        backend, layerwise=True, store=False, start_gate_factory=lambda: FakeLoadStartGate(opened=True)
    )
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=(command,))

    worker.start_load()
    assert worker.collect_load_result().failed_block_ids == {1}
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    worker.end_step()
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    monkeypatch.setattr(backend, "batch_get_start", original_start)
    begin_step(worker, load=(command,))
    worker.start_load()
    worker.wait_for_layer_load("layers.0.group.0")
    worker.wait_for_layer_load("layers.1.group.0")
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    worker.close()
    assert [call[0] for call in backend.calls].count("batch_get_end") == 2
    assert resources.closed


def test_layerwise_missing_load_hook_aborts_prefetch_and_next_step_can_load() -> None:
    gates = []

    def make_gate():
        gate = FakeLoadStartGate()
        gates.append(gate)
        return gate

    worker, resources, backend = make_worker(layerwise=True, store=False, start_gate_factory=make_gate)
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    try:
        begin_step(worker, load=(command,))
        worker.start_load()
        worker.wait_for_layer_load("layers.0.group.0")
        with pytest.raises(RuntimeError, match=r"did not reach physical layers \[1\]"):
            worker.collect_load_result()
        assert sum(call[0] == "batch_copy_get" for call in backend.calls) == 1
        assert sum(call[0] == "batch_get_end" for call in backend.calls) == 1
        worker.end_step()

        begin_step(worker, load=(command,))
        worker.start_load()
        worker.wait_for_layer_load("layers.0.group.0")
        gates[-1].open()
        worker.wait_for_layer_load("layers.1.group.0")
        assert not worker.collect_load_result().failed_locations
        assert sum(call[0] == "batch_copy_get" for call in backend.calls) == 3
        assert sum(call[0] == "batch_get_end" for call in backend.calls) == 2
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


def test_layerwise_failed_load_finish_is_observed_without_repeating_backend_cleanup() -> None:
    worker, resources, backend = make_worker(layerwise=True, store=False, physical_layers=(0,))
    backend.session_end_result = -1
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=(command,))
    worker.start_load()
    with pytest.raises(RuntimeError, match="batch_get_end failed"):
        worker.wait_for_layer_load("layers.0.group.0")
    with pytest.raises(RuntimeError, match="batch_get_end failed"):
        worker.close()
    assert sum(call[0] == "batch_get_end" for call in backend.calls) == 1
    assert worker._load_executor.stopped
    assert resources.closed


def test_layerwise_new_store_waits_for_its_single_session_but_empty_decode_does_not(monkeypatch) -> None:
    backend = FakeBackend()
    finalization_started = threading.Event()
    allow_finalization = threading.Event()
    original_revoke = backend.batch_revoke

    def blocking_revoke(keys):
        finalization_started.set()
        assert allow_finalization.wait(timeout=5)
        return original_revoke(keys)

    monkeypatch.setattr(backend, "batch_revoke", blocking_revoke)
    worker, resources, _ = make_worker(backend, layerwise=True)
    first = RangeStoreCommand("first", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    second = RangeStoreCommand("second", TokenRange(0, 4), ((3,),), (b"b",), 4, 18)
    prepared = threading.Event()
    failures = []

    def prepare_second_store():
        try:
            begin_step(worker, store=(second,))
        except BaseException as error:
            failures.append(error)
        finally:
            prepared.set()

    caller = threading.Thread(target=prepare_second_store)
    try:
        # Already-ready finalization is asynchronous. No layer copy was supplied, so it revokes, not publishes.
        worker.begin_step(KVTransferStep(store=StoreCommandBatch(source_ready_commands=(first,))))
        worker.finish_step()
        worker.end_step()
        assert finalization_started.wait(timeout=2)
        assert worker.fence_previous_store(frozenset()) == ()
        begin_step(worker)
        worker.finish_step()
        worker.end_step()
        assert worker.take_released_store_job_ids() == set()

        caller.start()
        assert not prepared.wait(timeout=0.05), "A new Store must not reuse the old session before finalization"
        assert worker.take_released_store_job_ids() == set()
        assert not resources.closed
        allow_finalization.set()
        assert prepared.wait(timeout=2)
        assert not failures
        assert worker.take_released_store_job_ids() == {17}
        assert worker.take_released_store_job_ids() == set()

        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
        worker.finish_step()
        assert worker.take_released_store_job_ids() == {18}
        assert worker.take_released_store_job_ids() == set()
        worker.end_step()
        assert [call[0] for call in backend.calls].count("batch_revoke") == 1
        assert [call[0] for call in backend.calls].count("batch_commit") == 1
    finally:
        allow_finalization.set()
        if caller.ident is not None:
            caller.join(timeout=5)
            assert not caller.is_alive()
        worker.close()
    assert resources.closed


def test_layerwise_store_commits_only_after_all_layers_and_unknown_failure_retains_source() -> None:
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    worker, resources, backend = make_worker(layerwise=True)
    begin_step(worker, store=(command,))

    worker.save_layer("layers.0.group.0")
    assert worker._store_executor is not None
    worker._store_executor.submit(lambda: None).result()
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
    begin_step(failed, store=(command, duplicate))
    failed.save_layer("layers.0.group.0")
    failed.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store source release is unknown"):
        failed.finish_step()
    assert failed._pending_store_batch is not None
    assert not failed._pending_store_batch.future.result()[0].evidence.source_release_confirmed
    assert failed._pending_store_batch.future.result()[1].evidence.source_release_confirmed
    assert failed.take_released_store_job_ids() == {18}
    assert "batch_commit" not in [call[0] for call in failing_backend.calls]
    assert "batch_revoke" in [call[0] for call in failing_backend.calls]
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        failed.close()
    assert not failed_resources.closed


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_layerwise_escaped_copy_error_preserves_later_jobs_and_retains_unknown_sources(use_gva, monkeypatch) -> None:
    worker, resources, backend = make_gva_worker() if use_gva else make_worker(layerwise=True)
    original = worker._layerwise_backend_io.store_layer
    reached = []

    def copy_then_raise(source_ready, layer_id):
        result = original(source_ready, layer_id)
        reached.append(layer_id)
        if layer_id == 0:
            raise RuntimeError("escaped layer copy error")
        return result

    monkeypatch.setattr(worker._layerwise_backend_io, "store_layer", copy_then_raise)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    worker.save_layer("layers.0.group.0")
    worker.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store source release is unknown"):
        worker.finish_step()
    assert reached == [0, 1]
    assert not worker.take_released_store_job_ids()
    assert not any(call[0] in ("publish", "batch_commit") for call in backend.calls)
    assert worker._pending_store_batch is not None
    evidence = worker._pending_store_batch.future.result()[0].evidence
    assert not evidence.succeeded
    assert not evidence.source_release_confirmed
    assert str(evidence.error) == "escaped layer copy error"
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        worker.close()
    assert worker._store_executor.stopped
    assert worker._load_executor.stopped
    assert not resources.closed


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_layerwise_escaped_preparation_error_skips_copies_releases_sources_and_next_step_recovers(
    use_gva, monkeypatch
) -> None:
    worker, resources, backend = make_gva_worker() if use_gva else make_worker(layerwise=True)
    original = worker._open_store

    def prepare_then_raise(progress):
        original(progress)
        raise RuntimeError("escaped Store preparation error")

    monkeypatch.setattr(worker, "_open_store", prepare_then_raise)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    try:
        begin_step(worker, store=(command,))
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
        worker.finish_step()
        assert worker.take_released_store_job_ids() == {17}
        assert not any(call[0] in ("copy", "batch_copy_put", "publish", "batch_commit") for call in backend.calls)
        if use_gva:
            assert worker.lookup(LookupRequest(TokenRange(0, 4), (0,), (b"a",))).available_end_token == 0
        worker.end_step()

        monkeypatch.setattr(worker, "_open_store", original)
        next_command = RangeStoreCommand("next", TokenRange(0, 4), ((3,),), (b"b",), 4, 18)
        begin_step(worker, store=(next_command,))
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
        worker.finish_step()
        assert worker.take_released_store_job_ids() == {18}
        assert sum(call[0] in ("publish", "batch_commit") for call in backend.calls) == 1
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("start_results", ([0, 0], [0, -1]), ids=("all_started", "partial_start"))
def test_layerwise_store_copies_and_commits_only_successfully_started_keys(start_results) -> None:
    backend = FakeBackend()
    backend.store_session_start_result = start_results
    worker, resources, _ = make_worker(backend, layerwise=True)
    first = RangeStoreCommand("first", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    second = RangeStoreCommand("second", TokenRange(0, 4), ((3,),), (b"b",), 4, 18)
    begin_step(worker, store=(first, second))
    worker.save_layer("layers.0.group.0")
    worker.save_layer("layers.1.group.0")
    worker.finish_step()

    started_keys = next(call[1] for call in backend.calls if call[0] == "batch_put_start")
    accepted = tuple(key for key, code in zip(started_keys, start_results, strict=True) if code == 0)
    assert [call[1] for call in backend.calls if call[0] == "batch_copy_put"] == [accepted, accepted]
    assert next(call[1] for call in backend.calls if call[0] == "batch_commit") == accepted
    assert worker.take_released_store_job_ids() == {17, 18}
    worker.close()
    assert resources.closed


def test_layerwise_non_leader_discards_store_rows_in_worker_business_path() -> None:
    topology = make_topology(tp_rank=1, tp_size=2, put_step=2)
    worker, resources, backend = make_worker(topology=topology, layerwise=True)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)

    begin_step(worker, store=(command,))
    worker.save_layer("layers.0.group.0")
    worker.save_layer("layers.1.group.0")
    worker.finish_step()
    assert not any(call[0] in ("batch_put_start", "batch_copy_put", "batch_commit") for call in backend.calls)
    assert worker.take_released_store_job_ids() == {17}

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
            begin_step(worker, store=(command,))
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


@pytest.mark.parametrize(
    "observed_layers,fail_second_request",
    (((), False), ((0,), False), ((0, 1), False), ((0, 1, 2), False), ((0, 1, 2), True)),
)
def test_layerwise_store_aggregates_terminal_evidence_across_layers(observed_layers, fail_second_request) -> None:
    topology = make_topology(group_ids=(0, 1), physical_layers=(0, 1, 2))
    topology = replace(
        topology,
        groups=(
            replace(topology.groups[0], layers=topology.groups[0].layers[:2]),
            replace(topology.groups[1], layers=topology.groups[1].layers[1:]),
        ),
    )
    backend = FakeBackend()
    if fail_second_request:
        backend.store_session_commit_result = [0, -1, 0, -1]
    worker, resources, _ = make_worker(backend, layerwise=True, topology=topology)
    commands = (
        RangeStoreCommand("first", TokenRange(0, 4), ((1,), (2,)), (b"a",), 4, 17),
        RangeStoreCommand("second", TokenRange(0, 4), ((3,), (4,)), (b"b",), 4, 18),
    )
    layer_names = ("layers.0.group.0", "layers.1.group.0", "layers.2.group.1")
    expected_layers = {
        group_id: tuple(layer for layer in layers if layer in observed_layers)
        for group_id, layers in ((0, (0, 1)), (1, (1, 2)))
    }
    try:
        begin_step(worker, store=commands)
        for layer_id in observed_layers:
            worker.save_layer(layer_names[layer_id])
        context = worker._active_step
        assert context is not None
        worker._finalize_layerwise_store(context)
        completions = worker.fence_previous_store()

        assert [completion.request_id for completion in completions] == ["first", "second"]
        for request_index, completion in enumerate(completions):
            evidence = completion.evidence.transfer_evidence
            assert [
                (item.source.group_id, item.source.block_id, item.source.physical_layer_ids) for item in evidence
            ] == [
                (group_id, commands[request_index].block_ids_by_group[group_id][0], layers)
                for group_id, layers in expected_layers.items()
                if layers
            ]
            succeeded = len(observed_layers) == 3 and not (fail_second_request and request_index == 1)
            assert completion.evidence.succeeded == succeeded
            assert completion.evidence.source_release_confirmed
            if len(observed_layers) == 3:
                assert all(item.result_code == (0 if succeeded else -1) for item in evidence)
        assert worker.take_released_store_job_ids() == {17, 18}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


def test_worker_close_reports_incomplete_layerwise_store_but_releases_safe_source() -> None:
    worker, resources, backend = make_worker(layerwise=True)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    worker.save_layer("layers.0.group.0")

    worker.close()

    assert worker.take_released_store_job_ids() == {17}
    assert worker._pending_store_batch is None
    assert resources.closed
    assert "batch_revoke" in [call[0] for call in backend.calls]
    assert "batch_commit" not in [call[0] for call in backend.calls]


def test_gva_worker_publishes_after_all_layers_and_reports_job_release() -> None:
    worker, resources, store = make_gva_worker()
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, store_job_id=17)
    begin_step(worker, store=(command,))

    worker.save_layer("layers.0.group.0")
    assert worker._store_executor is not None
    worker._store_executor.submit(lambda: None).result()
    assert [call[0] for call in store.calls].count("copy") == 1
    assert not any(call[0] == "publish" for call in store.calls)

    worker.save_layer("layers.1.group.0")
    worker._store_executor.submit(lambda: None).result()
    worker.finish_step()
    assert [call[0] for call in store.calls if call[0] in ("alloc", "copy", "publish")] == [
        "alloc",
        "copy",
        "copy",
        "publish",
    ]
    assert all(region[2] for region in store.objects.values())
    assert worker.take_released_store_job_ids() == {17}
    worker.close()
    assert resources.closed


@pytest.mark.parametrize("failure", ("allocation", "copy", "publication"))
def test_gva_store_failure_is_a_cache_miss_and_next_step_can_publish(failure: str, monkeypatch, caplog) -> None:
    monkeypatch.setattr(worker_module, "logger", logging.getLogger(__name__))
    worker, resources, store = make_gva_worker()
    if failure == "allocation":
        store.allocation_result = [0]
    elif failure == "copy":
        store.copy_result = -9
    else:
        store.commit_result = [-8]
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, store_job_id=17)
    begin_step(worker, store=(command,))
    worker.save_layer("layers.0.group.0")
    worker.save_layer("layers.1.group.0")
    worker.finish_step()

    assert "KV cache Store failed for request request (job 17)" in caplog.text
    if failure == "allocation":
        assert "Store session start failed" in caplog.text
        assert "result code -1" in caplog.text
    assert worker.take_released_store_job_ids() == {17}
    if failure == "copy":
        assert not any(call[0] == "publish" for call in store.calls)
    assert not any(region[2] for region in store.objects.values())
    assert worker.lookup(LookupRequest(TokenRange(0, 4), (0,), (b"a",))).available_end_token == 0
    worker.end_step()

    store.allocation_result = None
    store.copy_result = 0
    store.commit_result = None
    next_command = RangeStoreCommand("next", TokenRange(0, 4), ((3,),), (b"b",), 4, store_job_id=18)
    begin_step(worker, store=(next_command,))
    worker.save_layer("layers.0.group.0")
    worker.save_layer("layers.1.group.0")
    worker.finish_step()
    assert worker.take_released_store_job_ids() == {18}
    assert worker.lookup(LookupRequest(TokenRange(0, 4), (0,), (b"b",))).available_end_token == 4
    worker.close()
    assert resources.closed
