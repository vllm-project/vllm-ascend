"""Tests for worker.layerwise.key_range."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeLoadStartGate,
    begin_step,
    make_topology,
    make_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    LoadCommand,
    RangeStoreCommand,
)


@pytest.mark.parametrize("failed_group", (0, 3))
def test_key_range_merged_failure_identifies_every_original_request_destination(failed_group):
    topology = make_topology(group_ids=(0, 3), physical_layers=(2, 5))
    worker, resources, backend = make_worker(topology=topology, layerwise=True, store=False)
    commands = (
        LoadCommand("first", 0, 4, ((1,), (), (), (2,)), (b"same",)),
        LoadCommand("second", 0, 4, ((4,), (), (), (5,)), (b"same",)),
    )
    try:
        batch = worker._build_load_batch(commands)
        prepared = worker._prepare_load_layers(batch)
        backend.session_copy_result = (-7, 0) if failed_group == 0 else (0, -7)
        completions = prepared.copy_layer(2)
        assert [completion.request_id for completion in completions] == ["first", "second"]
        expected_blocks = (1, 4) if failed_group == 0 else (2, 5)
        for completion, block_id in zip(completions, expected_blocks, strict=True):
            (failure,) = completion.failures
            assert failure.result_code == -7
            assert (failure.source.group_id, failure.source.local_block_id) == (failed_group, block_id)
            assert failure.source.physical_layer_ids == (2,)
        copy = next(call for call in backend.calls if call[0] == "batch_copy_get")
        assert len(copy[1]) == 2  # Two Group keys, each merged across both requests.
    finally:
        worker.close()
    assert resources.closed


def test_prepared_load_keeps_its_key_merges_and_destinations_after_another_preparation():
    worker, resources, backend = make_worker(layerwise=True, store=False, physical_layers=(2,))
    commands = (
        LoadCommand("first", 0, 4, ((1,),), (b"same",)),
        LoadCommand("second", 0, 4, ((4,),), (b"same",)),
    )
    try:
        prepared = worker._prepare_load_layers(worker._build_load_batch(commands))
        other = worker._prepare_load_layers(
            worker._build_load_batch((LoadCommand("other", 0, 4, ((7,),), (b"other",)),))
        )
        backend.session_copy_result = (-7,)
        completions = prepared.copy_layer(2)
        assert [completion.request_id for completion in completions] == ["first", "second"]
        assert [completion.failures[0].source.local_block_id for completion in completions] == [1, 4]
        copy = next(call for call in backend.calls if call[0] == "batch_copy_get")
        assert copy[1] == tuple(dict.fromkeys(prepared.plan.batch.selected_keys()))
        assert set(copy[1]).isdisjoint(other.plan.batch.selected_keys())
        assert copy[2] == (tuple(resources.kv_caches["layers.2.group.0"][block].data_ptr() for block in (1, 4)),)
    finally:
        worker.close()


@pytest.mark.parametrize("with_load", (False, True), ids=("cold", "external_load"))
def test_key_range_hybrid_starts_store_before_attention_without_duplicate_saves(with_load, monkeypatch) -> None:
    monkeypatch.setattr(attention_fence, "_attention_compute_start_gate", None)
    monkeypatch.setattr(torch.npu, "Event", Mock)
    topology = make_topology(group_ids=(0, 1))
    worker, resources, backend = make_worker(layerwise=True, topology=topology)
    store = RangeStoreCommand("request", 0, 4, ((1,), (2,)), (b"b",), 4, 17)
    load = (LoadCommand("request", 0, 4, ((1,), (2,)), (b"a",)),) if with_load else ()
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


@pytest.mark.parametrize("finish", ("fallback", "end_step", "close"))
def test_hybrid_attention_store_callback_cannot_outlive_its_step(finish, monkeypatch) -> None:
    monkeypatch.setattr(attention_fence, "_attention_compute_start_gate", None)
    topology = make_topology(group_ids=(0, 1), physical_layers=(0,))
    worker, resources, backend = make_worker(layerwise=True, topology=topology)
    command = RangeStoreCommand("request", 0, 4, ((1,), (2,)), (b"a",), 4, 17)
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


@pytest.mark.parametrize(
    "physical_layers",
    ((0, 1, 2), (2, 5, 9)),
    ids=("contiguous", "sparse"),
)
def test_layerwise_load_copies_each_layer_within_one_session(physical_layers) -> None:
    worker, resources, backend = make_worker(
        layerwise=True,
        store=False,
        physical_layers=physical_layers,
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    command = LoadCommand("request", 0, 4, ((1,),), (b"a",))
    begin_step(worker, load=(command,))
    worker.start_load()
    for layer_id in physical_layers:
        worker.wait_for_layer_load(f"layers.{layer_id}.group.0")
        worker.wait_for_layer_load(f"layers.{layer_id}.group.0")
    assert not worker.collect_load_result().failed_locations
    worker.end_step()
    expected_addresses = [
        ((resources.kv_caches[f"layers.{layer}.group.0"][1].data_ptr(),),) for layer in physical_layers
    ]
    worker.close()

    session_calls = [call[0] for call in backend.calls if call[0] in ("batch_get_start", "batch_get_end")]
    copy_calls = [call for call in backend.calls if call[0] == "batch_copy_get"]
    assert session_calls == ["batch_get_start", "batch_get_end"]
    assert [call[2] for call in copy_calls] == expected_addresses
    assert [call[4] for call in copy_calls] == [((0,),), ((32,),), ((64,),)]
    assert resources.closed


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
    command = LoadCommand("request", 0, 4, ((1,),), (b"a",))
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


def test_layerwise_failed_load_finish_keeps_backend_ownership_for_close_retry() -> None:
    worker, resources, backend = make_worker(layerwise=True, store=False, physical_layers=(0,))
    backend.session_end_result = -1
    command = LoadCommand("request", 0, 4, ((1,),), (b"a",))
    begin_step(worker, load=(command,))
    worker.start_load()
    with pytest.raises(RuntimeError, match="batch_get_end failed"):
        worker.wait_for_layer_load("layers.0.group.0")
    with pytest.raises(RuntimeError, match="batch_get_end failed"):
        worker.close()
    assert sum(call[0] == "batch_get_end" for call in backend.calls) == 2
    assert worker._load_executor.stopped
    assert not resources.closed
    assert worker._backend_sessions.has_load_sessions
    backend.session_end_result = 0
    # The execution error is still reported, independently of successful resource retry.
    with pytest.raises(RuntimeError, match="batch_get_end failed"):
        worker.close()
    assert sum(call[0] == "batch_get_end" for call in backend.calls) == 3
    assert not worker._backend_sessions.has_load_sessions
    assert resources.closed


@pytest.mark.parametrize("copy_codes", ((0, 0), (0, -7)))
def test_partial_load_admission_preserves_original_failure_blocks(copy_codes):
    worker, resources, backend = make_worker(
        layerwise=True,
        store=False,
        physical_layers=(0, 1, 2),
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    backend.session_start_result = [-1, 0, 0]
    backend.session_copy_result = copy_codes
    commands = (LoadCommand("first", 0, 8, ((1, 2),), (b"a", b"b")), LoadCommand("second", 0, 4, ((7,),), (b"c",)))
    try:
        begin_step(worker, load=commands)
        worker.start_load()
        for layer in range(3):
            worker.wait_for_layer_load(f"layers.{layer}.group.0")
        result = worker.collect_load_result()
        assert result.failed_block_ids == ({1} if copy_codes[1] == 0 else {1, 7})
        assert len([call for call in backend.calls if call[0] == "batch_copy_get"]) == 3
        assert len([call for call in backend.calls if call[0] == "batch_get_end"]) == 1
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed
