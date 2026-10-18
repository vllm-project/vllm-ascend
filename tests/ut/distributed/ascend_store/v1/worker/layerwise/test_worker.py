"""Store admission, failure precedence, and source ownership across layer callbacks."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import replace
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeLoadStartGate,
    begin_step,
    make_backend_spec,
    make_reachability,
    make_topology,
    make_worker,
)
from tests.ut.distributed.ascend_store.v1.worker.bulk.fixtures import (
    make_multi_spec_caches,
    make_sparse_group_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.layerwise.fixtures import (
    FakeGVABackend,
    make_gva_spec,
    make_gva_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    GVAObjectSessions,
    ObjectSessions,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.resolve import (
    resolve_group_layers,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    LookupRequest,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import (
    KVPoolResources,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    GVALayerwiseWorker,
    KeyRangeLayerwiseWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import gva as gva_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import key_range as key_range_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import transfer as layer_transfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.transfer import (
    LayerStoreResult,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.worker import (
    LayerwiseWorker,
    _StoreState,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVTransferBatch,
)


def test_layer_results_update_only_current_keys_and_keep_scalar_success_constant_time():
    class NoScanDict(dict):
        def __iter__(self):
            pytest.fail("Layer results must not scan accumulated state")

        def items(self):
            pytest.fail("Layer results must not scan accumulated state")

        def keys(self):
            pytest.fail("Layer results must not scan accumulated state")

        def values(self):
            pytest.fail("Layer results must not scan accumulated state")

    class NoScanKeys(list):
        def __iter__(self):
            pytest.fail("Scalar success must not expand into per-key results")

    state = _StoreState(())
    state.failure_codes = NoScanDict({"older": -1})
    state.errors_by_key = NoScanDict()
    LayerwiseWorker._record_store_result(state, NoScanKeys(["first", "second"]), LayerStoreResult(0, True))
    LayerwiseWorker._record_store_result(state, ["first", "second"], LayerStoreResult((-9, 0), (True, True)))
    error = RuntimeError("copy completion unknown")
    LayerwiseWorker._record_store_result(state, ["first", "second"], LayerStoreResult(None, False, error))
    assert state.failure_codes == {"older": -1, "first": -9, "second": None}
    assert state.unsafe_keys == {"first", "second"}
    assert state.errors_by_key == {"first": error, "second": error}


@pytest.mark.parametrize("start_codes", ((0, 0), (0, -7), (-6, -7)))
@pytest.mark.parametrize("failure_stage", (None, "start", "prepare_layers"))
def test_store_admission_keeps_failed_requests_separate_from_copy_sources(start_codes, failure_stage, monkeypatch):
    backend = FakeBackend()
    backend.store_session_start_result = list(start_codes)
    worker, resources, _ = make_worker(backend, layerwise=True, topology=make_topology(physical_layers=(2, 5, 9)))
    commands = (
        RangeStoreCommand("first", 0, 4, ((1,),), (b"a",), 4, 17),
        RangeStoreCommand("second", 0, 4, ((3,),), (b"b",), 4, 18),
        RangeStoreCommand("duplicate", 0, 4, ((5,),), (b"a",), 4, 19),
    )
    error = RuntimeError(f"failed at {failure_stage}")
    if failure_stage == "start":
        start = worker._backend_sessions.start_store

        def start_then_raise(*args):
            start(*args)
            raise error

        monkeypatch.setattr(worker._backend_sessions, "start_store", start_then_raise)
    elif failure_stage == "prepare_layers":

        def fail_preparation(batch):
            raise error

        monkeypatch.setattr(worker, "_prepare_store_layers", fail_preparation)
    try:
        begin_step(worker, store=commands)
        for layer_id in (2, 2, 5, 5, 9, 9):
            worker.save_layer(f"layers.{layer_id}.group.0")
        worker._finalize_layerwise_store(worker._active_step)
        completions = worker.fence_previous_store()
        assert [completion.request_id for completion in completions] == ["first", "second", "duplicate"]
        for index, completion in enumerate(completions[:2]):
            copied = start_codes[index] == 0 and failure_stage is None
            assert completion.evidence.succeeded == copied
            assert completion.evidence.source_release_confirmed
            evidence = completion.evidence.transfer_evidence
            assert len(evidence) == int(copied)
            if copied:
                assert evidence[0].source.block_id == commands[index].block_ids_by_group[0][0]
                assert evidence[0].source.physical_layer_ids == (2, 5, 9)
                assert evidence[0].result_code == 0
            elif failure_stage == "start" or start_codes[index] == 0:
                assert completion.evidence.error is error
            else:
                assert f"result code {start_codes[index]}" in str(completion.evidence.error)
        assert completions[2].evidence.succeeded
        assert completions[2].evidence.source_release_confirmed
        assert completions[2].evidence.transfer_evidence == ()
        started_keys = next(call[1] for call in backend.calls if call[0] == "batch_put_start")
        copied_keys = tuple(key for key, code in zip(started_keys, start_codes, strict=True) if code == 0)
        copies = [call[1] for call in backend.calls if call[0] == "batch_copy_put"]
        assert copies == ([copied_keys] * 3 if copied_keys and failure_stage is None else [])
        commits = [call[1] for call in backend.calls if call[0] == "batch_commit"]
        assert commits == ([copied_keys] if copied_keys and failure_stage is None else [])
        assert worker.take_released_store_job_ids() == {17, 18, 19}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize(
    ("copy_codes", "commit_code", "revoke_code", "expected_code"),
    (
        ((-9, -10), 0, 0, -9),
        ((-9, 0), 0, -8, -8),
        ((0, 0), -7, -8, -7),
        ((0, 0), None, -8, -8),
        ((0, 0), None, 0, None),
        ((0, 0), 0, 0, 0),
    ),
)
def test_store_result_precedence_does_not_hide_copy_source_release(
    copy_codes, commit_code, revoke_code, expected_code, monkeypatch
):
    backend = FakeBackend()
    backend.store_session_commit_result = [commit_code] if commit_code is not None else None
    backend.store_session_revoke_result = [revoke_code]
    worker, resources, _ = make_worker(backend, layerwise=True)
    copy = backend.batch_copy_put
    codes = iter(copy_codes)

    def copy_with_next_result(*args):
        backend.store_session_copy_result = [next(codes)]
        return copy(*args)

    monkeypatch.setattr(backend, "batch_copy_put", copy_with_next_result)
    commit_error = RuntimeError("commit failed")
    if commit_code is None:
        commit = backend.batch_commit

        def commit_then_raise(keys):
            commit(keys)
            raise commit_error

        monkeypatch.setattr(backend, "batch_commit", commit_then_raise)
    try:
        command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
        begin_step(worker, store=(command,))
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
        worker._finalize_layerwise_store(worker._active_step)
        (completion,) = worker.fence_previous_store()
        evidence = completion.evidence
        assert evidence.succeeded == (copy_codes == (0, 0) and commit_code == 0)
        assert evidence.source_release_confirmed
        assert len(evidence.transfer_evidence) == 1
        assert evidence.transfer_evidence[0].result_code == expected_code
        assert evidence.transfer_evidence[0].source.physical_layer_ids == (0, 1)
        if commit_code is None:
            assert evidence.error is commit_error
        calls = [call[0] for call in backend.calls]
        assert calls.count("batch_commit") == int(copy_codes == (0, 0))
        assert calls.count("batch_revoke") == int(not evidence.succeeded)
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
    finally:
        if revoke_code != 0:
            with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
                worker.close()
            assert not resources.closed
            assert worker._backend_sessions.pending_store_keys
            backend.store_session_revoke_result = [0]
        worker.close()
    assert resources.closed


def test_store_candidate_failure_completes_every_command_without_handing_off_sources(monkeypatch):
    worker, resources, backend = make_worker(layerwise=True)
    error = RuntimeError("candidate preparation failed")

    def fail_candidates(commands):
        raise error

    monkeypatch.setattr(worker, "_build_store_candidates", fail_candidates)
    try:
        commands = (
            RangeStoreCommand("first", 0, 4, ((1,),), (b"a",), 4, 17),
            RangeStoreCommand("second", 0, 4, ((3,),), (b"b",), 4, 18),
        )
        begin_step(worker, store=commands)
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
        worker._finalize_layerwise_store(worker._active_step)
        completions = worker.fence_previous_store()
        assert len(completions) == 2
        for completion in completions:
            assert not completion.evidence.succeeded
            assert completion.evidence.error is error
            assert completion.evidence.source_release_confirmed
            assert completion.evidence.transfer_evidence == ()
        assert not any(call[0] in ("batch_put_start", "batch_copy_put", "batch_commit") for call in backend.calls)
        assert worker.take_released_store_job_ids() == {17, 18}
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
        module, materializer = gva_worker, "materialize_layer_gva"
        copy_method = "batch_copy"
    else:
        worker, resources, backend = make_worker(
            layerwise=True, topology=topology, source_ready_event_factory=lambda: event
        )
        module, materializer = key_range_worker, "materialize_store_layer_ranges"
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
        command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
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
        cleanup_unconfirmed = use_gva and failure_stage is not None
        with pytest.raises(RuntimeError, match="cleanup is unconfirmed") if cleanup_unconfirmed else nullcontext():
            worker.close()
    assert resources.closed is not cleanup_unconfirmed
    assert bool(worker._backend_sessions.pending_store_keys) == cleanup_unconfirmed


@pytest.mark.parametrize("use_gva", (False, True), ids=("single_group_key_range", "hybrid_gva"))
def test_other_layerwise_routes_keep_post_compute_store(use_gva, monkeypatch) -> None:
    monkeypatch.setattr(attention_fence, "_attention_compute_start_gate", None)
    topology = make_topology(group_ids=(0, 1) if use_gva else (0,), physical_layers=(0,))
    if use_gva:
        worker, resources, backend = make_gva_worker(topology=topology)
    else:
        worker, resources, backend = make_worker(layerwise=True, topology=topology)
    blocks = tuple((index + 1,) for index in range(len(topology.groups)))
    command = RangeStoreCommand("request", 0, 4, blocks, (b"a",), 4, 17)
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
    resources = KVPoolResources(backend, backend_spec, 8, topology.transfer_groups)
    reachability = make_reachability(topology, 256, retention_interval=retention_interval)
    worker_type = GVALayerwiseWorker if use_gva else KeyRangeLayerwiseWorker
    worker = worker_type(topology, resources, reachability=reachability, source_ready_event_factory=FakeEvent)
    block_hashes = tuple(bytes([index]) for index in range(1, 9))
    caches = {group.layer_names[0]: torch.zeros((8, 2, 16, 1, 1)) for group in topology.transfer_groups}
    try:
        worker.bind_kv_caches(caches)
        command = RangeStoreCommand(
            "producer", 0, 96, ((), (1, 2, 3, 4, 5, 6), (), (1, 2, 3, 4, 5, 6)), block_hashes[:6], 97, 17
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
        swa_keys = tuple(key for key in readable_keys if "@group:3@" in key)
        assert sorted(int(key.rsplit("@", 1)[1], 16) * 16 for key in swa_keys) == list(stored_swa_boundaries)
        for start, end, expected_end in ((0, 128, 96), (0, 80, middle_end), (64, 128, 96)):
            result = worker.lookup(LookupRequest(start, end, (1, 3), block_hashes))
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
    resources = KVPoolResources(backend, backend_spec, cache_config.num_blocks, topology.transfer_groups)
    reachability = make_reachability(topology)
    worker_type = GVALayerwiseWorker if use_gva else KeyRangeLayerwiseWorker
    worker = worker_type(topology, resources, reachability=reachability, source_ready_event_factory=FakeEvent)
    source_blocks = {0: 1, 2: 3}
    try:
        worker.bind_kv_caches(caches)
        command = RangeStoreCommand("request", 0, 32, ((1,), (), (3,)), (b"a",), 32, 17)
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
                    if f"@group:{group.group_id}@" in key
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
        assert worker.lookup(LookupRequest(0, 32, (0, 2), (b"a",))).available_end_token == 32
    finally:
        worker.close()
    assert backend.closed


def test_layerwise_load_keeps_a_bounded_prefetch_window() -> None:
    gates: list[FakeLoadStartGate] = []

    def make_gate():
        gate = FakeLoadStartGate()
        gates.append(gate)
        return gate

    worker, _, backend = make_worker(
        layerwise=True, store=False, physical_layers=(0, 1, 2), start_gate_factory=make_gate
    )
    command = LoadCommand("request", 0, 4, ((1,),), (b"a",))
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


def test_layerwise_missing_load_hook_aborts_prefetch_and_next_step_can_load() -> None:
    gates = []

    def make_gate():
        gate = FakeLoadStartGate()
        gates.append(gate)
        return gate

    worker, resources, backend = make_worker(layerwise=True, store=False, start_gate_factory=make_gate)
    command = LoadCommand("request", 0, 4, ((1,),), (b"a",))
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
    first = RangeStoreCommand("first", 0, 4, ((1,),), (b"a",), 4, 17)
    second = RangeStoreCommand("second", 0, 4, ((3,),), (b"b",), 4, 18)
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
    command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
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
    duplicate = RangeStoreCommand("duplicate", 0, 4, ((3,),), (b"a",), 4, 18)
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
    original = worker._copy_store_layer
    reached = []

    def copy_then_raise(source_ready, layer_id):
        result = original(source_ready, layer_id)
        reached.append(layer_id)
        if layer_id == 0:
            raise RuntimeError("escaped layer copy error")
        return result

    monkeypatch.setattr(worker, "_copy_store_layer", copy_then_raise)
    command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
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
    command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
    try:
        begin_step(worker, store=(command,))
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
        worker.finish_step()
        assert worker.take_released_store_job_ids() == {17}
        assert not any(call[0] in ("copy", "batch_copy_put", "publish", "batch_commit") for call in backend.calls)
        if use_gva:
            assert worker.lookup(LookupRequest(0, 4, (0,), (b"a",))).available_end_token == 0
        worker.end_step()

        monkeypatch.setattr(worker, "_open_store", original)
        next_command = RangeStoreCommand("next", 0, 4, ((3,),), (b"b",), 4, 18)
        begin_step(worker, store=(next_command,))
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
        worker.finish_step()
        assert worker.take_released_store_job_ids() == {18}
        assert sum(call[0] in ("publish", "batch_commit") for call in backend.calls) == 1
        worker.end_step()
    finally:
        with pytest.raises(RuntimeError, match="cleanup is unconfirmed") if use_gva else nullcontext():
            worker.close()
    assert resources.closed is not use_gva
    assert bool(worker._backend_sessions.pending_store_keys) == use_gva


def test_layerwise_non_leader_discards_store_rows_in_worker_business_path() -> None:
    topology = make_topology(tp_rank=1, tp_size=2, put_step=2)
    worker, resources, backend = make_worker(topology=topology, layerwise=True)
    command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)

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
    command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
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
    ] == ["exists", "batch_put_start", "batch_copy_put", "batch_copy_put", "batch_commit"]
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
        RangeStoreCommand("first", 0, 4, ((1,), (2,)), (b"a",), 4, 17),
        RangeStoreCommand("second", 0, 4, ((3,), (4,)), (b"b",), 4, 18),
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
    command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    worker.save_layer("layers.0.group.0")

    worker.close()

    assert worker.take_released_store_job_ids() == {17}
    assert worker._pending_store_batch is None
    assert resources.closed
    assert "batch_revoke" in [call[0] for call in backend.calls]
    assert "batch_commit" not in [call[0] for call in backend.calls]


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_worker_uses_backend_supplied_session_owner(monkeypatch, use_gva):
    sessions = Mock(spec=GVAObjectSessions if use_gva else ObjectSessions)
    sessions.has_load_sessions = False
    sessions.pending_store_keys = ()
    create = Mock(return_value=sessions)
    backend_type = FakeGVABackend if use_gva else FakeBackend
    monkeypatch.setattr(backend_type, "create_sessions", create)
    # A Worker constructing the old concrete class would bypass the factory.
    validate = Mock(side_effect=AssertionError("Worker must not choose a native session implementation"))
    monkeypatch.setattr(backend_type, "validate_gva_support" if use_gva else "validate_key_range_support", validate)

    worker, resources, _ = make_gva_worker() if use_gva else make_worker(layerwise=True)
    try:
        create.assert_called_once_with()
        assert worker._backend_sessions is sessions
        assert resources._backend_sessions is sessions
        validate.assert_not_called()
    finally:
        worker.close()
    sessions.close.assert_called_once_with()


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_worker_propagates_session_factory_failure(monkeypatch, use_gva):
    backend_type = FakeGVABackend if use_gva else FakeBackend
    create = Mock(side_effect=RuntimeError("native sessions unavailable"))
    monkeypatch.setattr(backend_type, "create_sessions", create)

    with pytest.raises(RuntimeError, match="native sessions unavailable"):
        if use_gva:
            make_gva_worker()
        else:
            make_worker(layerwise=True)
    create.assert_called_once_with()


@pytest.mark.parametrize("gva", (False, True), ids=("key_range", "gva"))
def test_all_successful_session_admission_reuses_batch_without_reselecting_objects(gva, monkeypatch):
    if gva:
        worker, _, backend = make_gva_worker()
    else:
        worker, _, backend = make_worker(layerwise=True)
    commands = (LoadCommand("first", 0, 4, ((1,),), (b"same",)), LoadCommand("second", 0, 4, ((7,),), (b"same",)))
    batch = worker._build_load_batch(commands)
    if gva:
        backend.objects.update({key: (10000, batch.groups[0].object_size, True) for key in batch.selected_keys()})
    select = Mock(side_effect=AssertionError("Successful admission must reuse its input selection"))
    monkeypatch.setattr(KVTransferBatch, "select_keys", select)
    try:
        assert worker._submit_load(batch) == ()
        assert worker._load_plan.batch is batch
        assert worker._load_plan.groups[0].request_indices == (0, 1)
        assert worker._load_initial_completions == ()
        select.assert_not_called()
    finally:
        worker.close()


@pytest.mark.parametrize("route", ("key_range", "gva"))
@pytest.mark.parametrize("native_result", (-9, RuntimeError("copy failed")))
def test_layer_copy_failure_preserves_unknown_code_and_releases_load_sessions(route, native_result, monkeypatch):
    topology = make_topology(physical_layers=(2, 5))
    if route == "gva":
        worker, resources, backend = make_gva_worker(topology=topology)
        worker._start_gate_factory = lambda: FakeLoadStartGate(opened=True)
    else:
        worker, resources, backend = make_worker(
            topology=topology, layerwise=True, store=False, start_gate_factory=lambda: FakeLoadStartGate(opened=True)
        )
    command = LoadCommand("request", 0, 4, ((3,),), (b"a",))
    if route == "gva":
        batch = worker._build_load_batch((command,))
        backend.objects[batch.selected_keys()[0]] = (10000, 64, True)
        backend.copy_result = native_result
    else:
        backend.session_copy_result = native_result if isinstance(native_result, Exception) else (native_result,)
    observed_codes = []
    record = worker._record_load_completion

    def record_failures(context, completion):
        observed_codes.extend(item.result_code for item in completion.failures)
        record(context, completion)

    monkeypatch.setattr(worker, "_record_load_completion", record_failures)
    try:
        begin_step(worker, load=(command,))
        worker.start_load()
        for layer in (2, 2, 5, 5):
            worker.wait_for_layer_load(f"layers.{layer}.group.0")
        expected_code = None if isinstance(native_result, Exception) else native_result
        assert observed_codes == [expected_code, expected_code]
        assert not worker._backend_sessions.has_load_sessions
        if route == "key_range":
            assert sum(call[0] == "batch_get_end" for call in backend.calls) == 1
        context = worker._active_step
        assert [(source.block_id, source.physical_layer_ids) for source in context.failed_sources] == [
            (3, (2,)),
            (3, (5,)),
        ]
        result = worker.collect_load_result()
        assert result.failed_block_ids == {3}
        assert [(item.group_id, item.block_id) for item in result.failed_locations] == [(0, 3)]
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("scenario", ("no_commands", "no_candidates", "all_present", "all_starts_failed"))
def test_known_empty_store_never_creates_a_step_session_or_layer_work(scenario, monkeypatch):
    backend = FakeBackend()
    if scenario == "all_starts_failed":
        backend.store_session_start_result = [-7]
    record = Mock(side_effect=FakeEvent)
    gate = Mock(side_effect=FakeLoadStartGate)
    worker, resources, _ = make_worker(
        backend,
        layerwise=True,
        physical_layers=(2, 5, 9),
        requires_exists_before_put=scenario == "all_present",
        source_ready_event_factory=record,
        start_gate_factory=gate,
    )
    prepare = Mock(side_effect=worker._prepare_store_layers)
    monkeypatch.setattr(worker, "_prepare_store_layers", prepare)
    end_token = 0 if scenario == "no_candidates" else 4
    command = RangeStoreCommand("request", 0, end_token, ((1,),), (b"a",), 4, 17)
    commands = () if scenario == "no_commands" else (command,)
    try:
        begin_step(worker, store=commands)
        # The empty result is known before the hooks; this barrier is not part of dispatch.
        worker._store_executor.submit(lambda: None).result(timeout=5)
        assert worker._session is None
        submit = Mock(side_effect=worker._store_executor.submit)
        monkeypatch.setattr(worker._store_executor, "submit", submit)
        for layer in (2, 2, 5, 9):
            worker.wait_for_layer_load(f"layers.{layer}.group.0")
            worker.save_layer(f"layers.{layer}.group.0")
        assert worker._session is None
        submit.assert_not_called()
        record.assert_not_called()
        gate.assert_not_called()
        prepare.assert_not_called()
        worker.finish_step()
        worker.fence_previous_store()
        calls = [call[0] for call in backend.calls]
        assert calls.count("batch_put_start") == int(scenario == "all_starts_failed")
        assert "batch_copy_put" not in calls
        assert "batch_commit" not in calls
        assert worker.take_released_store_job_ids() == ({17} if commands else set())
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_all_present_store_exits_before_materialization_or_backend_sessions(use_gva, monkeypatch):
    if use_gva:
        worker, resources, backend = make_gva_worker()
        command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
        candidates = worker._build_store_candidates((command,))
        for group in candidates.groups:
            object_size = worker._mappings[group.group_id].local_group_block_bytes
            for axis in group.key_axes:
                for key in axis:
                    backend.objects[key] = (10000, object_size, True)
    else:
        worker, resources, backend = make_worker(layerwise=True, requires_exists_before_put=True)
        command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
    materialize = Mock(side_effect=AssertionError("All-present keys must exit before materialization"))
    start = Mock(side_effect=AssertionError("Presence checks must not acquire Backend sessions"))
    monkeypatch.setattr(worker, "_materialize_store_candidates", materialize)
    monkeypatch.setattr(worker._backend_sessions, "start_store", start)
    record = Mock(side_effect=FakeEvent)
    monkeypatch.setattr(worker, "_source_ready_event_factory", record)
    try:
        begin_step(worker, store=(command,))
        worker._store_executor.submit(lambda: None).result(timeout=5)
        for name in worker._layer_ids_by_name:
            worker.wait_for_layer_load(name)
            worker.save_layer(name)
        assert worker._session is None
        record.assert_not_called()
        worker.finish_step()
        assert worker.take_released_store_job_ids() == {17}
        materialize.assert_not_called()
        start.assert_not_called()
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("scenario", ("no_commands", "no_objects", "all_starts_failed"))
def test_known_empty_load_never_creates_a_step_session_or_compute_gates(scenario):
    backend = FakeBackend()
    if scenario == "all_starts_failed":
        backend.session_start_result = [-7]
    gate = Mock(side_effect=FakeLoadStartGate)
    worker, resources, _ = make_worker(backend, layerwise=True, start_gate_factory=gate)
    end_token = 0 if scenario == "no_objects" else 4
    command = LoadCommand("request", 0, end_token, ((1,),), (b"a",))
    commands = () if scenario == "no_commands" else (command,)
    try:
        begin_step(worker, load=commands)
        worker.start_load()
        for layer in (0, 0, 1):
            worker.wait_for_layer_load(f"layers.{layer}.group.0")
        assert worker._session is None
        gate.assert_not_called()
        calls = [call[0] for call in backend.calls]
        assert calls.count("batch_get_start") == int(scenario == "all_starts_failed")
        assert "batch_copy_get" not in calls
        assert "batch_get_end" not in calls
        worker.collect_load_result()
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("start_codes", ((0, 0), (0, -7), (-6, -7)))
@pytest.mark.parametrize("use_gva", (False, True), ids=("key_range", "gva"))
def test_store_builds_shared_layer_information_only_after_backend_admission(start_codes, use_gva, monkeypatch):
    topology = make_topology(physical_layers=(2, 5, 9))
    if use_gva:
        worker, resources, backend = make_gva_worker(topology=topology)
        backend.allocation_result = [10000 * (index + 1) if code == 0 else 0 for index, code in enumerate(start_codes)]
    else:
        backend = FakeBackend()
        backend.store_session_start_result = list(start_codes)
        worker, resources, _ = make_worker(backend, layerwise=True, topology=topology)
    prepare = Mock(side_effect=worker._prepare_store_layers)
    monkeypatch.setattr(worker, "_prepare_store_layers", prepare)
    make_group = layer_transfer.make_layer_transfer_group

    def make_admitted_group(group, **kwargs):
        assert any(call[0] == ("alloc" if use_gva else "batch_put_start") for call in backend.calls)
        assert len(group.selected_keys()) == start_codes.count(0)
        return make_group(group, **kwargs)

    shared_group = Mock(side_effect=make_admitted_group)
    monkeypatch.setattr(layer_transfer, "make_layer_transfer_group", shared_group)
    plans = []
    copy = worker._copy_store_layer

    def copy_with_shared_plan(event, layer):
        plans.append(worker._store_plan)
        return copy(event, layer)

    monkeypatch.setattr(worker, "_copy_store_layer", copy_with_shared_plan)
    commands = (
        RangeStoreCommand("first", 0, 4, ((1,),), (b"a",), 4, 17),
        RangeStoreCommand("second", 0, 4, ((2,),), (b"b",), 4, 18),
    )
    try:
        begin_step(worker, store=commands)
        for layer in (2, 5, 9):
            worker.save_layer(f"layers.{layer}.group.0")
        worker.finish_step()
        assert shared_group.call_count == int(0 in start_codes)
        assert prepare.call_count == int(0 in start_codes)
        assert len(plans) == (3 if 0 in start_codes else 0)
        assert all(plan is plans[0] for plan in plans)
        copies = [call for call in backend.calls if call[0] == ("copy" if use_gva else "batch_copy_put")]
        assert all(len(call[1]) == start_codes.count(0) for call in copies)
        assert worker.take_released_store_job_ids() == {17, 18}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


def test_store_admission_does_not_block_model_hooks_and_keeps_recorded_source_boundary(monkeypatch):
    worker, resources, backend = make_worker(layerwise=True, physical_layers=(0,))
    admitted = threading.Event()
    continue_admission = threading.Event()
    original = worker._admitted_store_keys
    records = []

    def admission(keys):
        admitted.set()
        assert continue_admission.wait(timeout=5)
        return original(keys)

    def event():
        recorded = FakeEvent()
        records.append(recorded)
        return recorded

    monkeypatch.setattr(worker, "_admitted_store_keys", admission)
    monkeypatch.setattr(worker, "_source_ready_event_factory", event)
    try:
        command = RangeStoreCommand("request", 0, 4, ((1,),), (b"a",), 4, 17)
        begin_step(worker, store=(command,))
        assert admitted.wait(timeout=2)
        worker.save_layer("layers.0.group.0")
        assert len(records) == 1 and records[0].recorded and not records[0].synchronized
        assert not any(call[0] == "batch_put_start" for call in backend.calls)
        continue_admission.set()
        worker.finish_step()
        assert records[0].synchronized
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
    finally:
        continue_admission.set()
        worker.close()
    assert resources.closed


def test_blocked_store_does_not_block_load_and_retains_state_after_end_step(monkeypatch):
    worker, resources, backend = make_worker(layerwise=True, start_gate_factory=FakeLoadStartGate)
    store_started = threading.Event()
    release_store = threading.Event()
    copy = worker._copy_store_layer

    def blocked_copy(event, layer):
        if layer == 0:
            store_started.set()
            assert release_store.wait(timeout=5)
        return copy(event, layer)

    monkeypatch.setattr(worker, "_copy_store_layer", blocked_copy)
    try:
        begin_step(
            worker,
            load=(LoadCommand("load", 0, 4, ((2,),), (b"b",)),),
            store=(RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17),),
        )
        worker.start_load()
        worker.wait_for_layer_load("layers.0.group.0")
        worker.save_layer("layers.0.group.0")
        assert store_started.wait(timeout=2)
        worker._session._get_or_create_compute_start_gate(0).open()
        worker.wait_for_layer_load("layers.1.group.0")
        worker.save_layer("layers.1.group.0")
        assert sum(call[0] == "batch_copy_get" for call in backend.calls) == 2
        assert not any(call[0] == "batch_copy_put" for call in backend.calls)
        gate = worker._session._get_or_create_compute_start_gate(1)
        callback = Mock()
        gate.on_start = callback
        worker.end_step()
        assert worker._session is None and worker._store_state is None
        assert gate.on_start is None
        callback.assert_not_called()
        assert worker._pending_store_batch is not None and not worker._pending_store_batch.future.done()
        assert not resources.closed
        assert worker.take_released_store_job_ids() == set()
        release_store.set()
        (completion,) = worker.fence_previous_store()
        assert completion.evidence.succeeded and completion.evidence.source_release_confirmed
        assert completion.evidence.transfer_evidence[0].source.physical_layer_ids == (0, 1)
        assert worker.take_released_store_job_ids() == {17}
    finally:
        release_store.set()
        worker.close()
    assert resources.closed


def test_escaped_store_task_error_revokes_objects_without_claiming_source_release(monkeypatch):
    worker, resources, backend = make_worker(layerwise=True, physical_layers=(0,))
    error = RuntimeError("copy escaped before source completion was confirmed")

    def escaped_copy(event, layer):
        raise error

    monkeypatch.setattr(worker, "_copy_store_layer", escaped_copy)
    begin_step(worker, store=(RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17),))
    worker.save_layer("layers.0.group.0")
    worker._finalize_layerwise_store(worker._active_step)
    (completion,) = worker._pending_store_batch.future.result(timeout=5)
    assert completion.evidence.error is error
    assert not completion.evidence.succeeded and not completion.evidence.source_release_confirmed
    assert any(call[0] == "batch_revoke" for call in backend.calls)
    assert not any(call[0] == "batch_commit" for call in backend.calls)
    with pytest.raises(RuntimeError, match="Store source release is unknown"):
        worker.fence_previous_store()
    assert worker.take_released_store_job_ids() == set()
    with pytest.raises(RuntimeError, match="cannot continue after a fatal Store failure"):
        worker.close()
    assert not resources.closed
