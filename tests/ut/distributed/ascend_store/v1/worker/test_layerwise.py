"""Layerwise copies, publication barriers, source safety, and bounded prefetch."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeLoadStartGate,
    begin_step,
    make_backend_spec,
    make_topology,
    make_worker,
)
from tests.ut.distributed.ascend_store.v1.worker.bulk_fixtures import make_multi_spec_caches
from tests.ut.distributed.ascend_store.v1.worker.gva_fixtures import (
    FakeGVABackend,
    make_gva_spec,
    make_gva_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    GVALayerwiseProjectionBinder,
    KeyRangeLayerwiseProjectionBinder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupRequest,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    LoadCommand,
    RangeStoreCommand,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import resolve_group_layers
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import base as worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    GVALayerwiseWorker,
    KeyRangeLayerwiseWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.resources import (
    GVAObjectLayout,
    KVPoolResources,
)


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


def test_layerwise_load_copies_each_layer_within_one_session() -> None:
    worker, resources, backend = make_worker(
        layerwise=True,
        store=False,
        physical_layers=(0, 1, 2),
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=(command,))
    worker.start_load()
    for layer_id in range(3):
        worker.wait_for_layer_load(f"layers.{layer_id}.group.0")
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    worker.close()

    session_calls = [call[0] for call in backend.calls if call[0] in ("batch_get_start", "batch_get_end")]
    copy_calls = [call for call in backend.calls if call[0] == "batch_copy_get"]
    assert session_calls == ["batch_get_start", "batch_get_end"]
    assert [call[2] for call in copy_calls] == [((1064,),), ((2064,),), ((3064,),)]
    assert [call[4] for call in copy_calls] == [((0,),), ((32,),), ((64,),)]
    assert resources.closed


@pytest.mark.parametrize("native_result", ([-1], RuntimeError("range copy failed")), ids=("failed", "exception"))
def test_layerwise_load_failure_closes_session_and_reports_original_block(native_result) -> None:
    backend = FakeBackend()
    backend.session_copy_result = native_result
    worker, resources, _ = make_worker(backend, layerwise=True, store=False, physical_layers=(0,))
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
    try:
        begin_step(worker, load=(command,))
        worker.start_load()
        worker.wait_for_layer_load("layers.0.group.0")
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

    def fail_start(keys):
        backend.calls.append(("batch_get_start", tuple(keys)))
        raise RuntimeError("session start failed")

    monkeypatch.setattr(backend, "batch_get_start", fail_start)
    worker, resources, _ = make_worker(backend, layerwise=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=(command,))

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
    begin_step(worker, store=(command,))

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
    begin_step(failed, store=(command, duplicate))
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


def test_layerwise_store_aggregates_terminal_evidence_across_layers() -> None:
    worker, resources, _ = make_worker(layerwise=True)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
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
    assert worker._layerwise_store_timeline is not None
    worker._layerwise_store_timeline._executor._queue.join()
    assert [call[0] for call in store.calls].count("copy") == 1
    assert not any(call[0] == "publish" for call in store.calls)

    worker.save_layer("layers.1.group.0")
    worker._layerwise_store_timeline._executor._queue.join()
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
