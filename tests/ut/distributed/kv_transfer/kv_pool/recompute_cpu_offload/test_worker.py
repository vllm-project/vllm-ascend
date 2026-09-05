# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheTensor

from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.metadata import (  # noqa: E402
    RecomputeCPUOffloadMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.worker import (  # noqa: E402
    RecomputeCPUOffloadWorker,
)


def test_recompute_cpu_offload_worker_metadata_and_empty_transfers():
    worker = RecomputeCPUOffloadWorker.__new__(RecomputeCPUOffloadWorker)
    worker._connector_metadata = None
    worker._pending_load_event_indices = set()
    worker._submitted_load_event_indices = set()
    worker._completed_store_events = {}
    worker._load_events = []
    worker._load_hwm = -1
    worker.load_stream = None
    worker._load_stream_waited = False

    metadata = RecomputeCPUOffloadMetadata(
        preempt_store_event=1,
        preempt_load_event=2,
        preempt_load_event_to_reqs={2: ["req-1"]},
    )
    worker.bind_connector_metadata(metadata)
    assert worker._connector_metadata is metadata
    assert worker._pending_load_event_indices == {2}

    worker._submit_transfer([], [], 1, is_store=True)
    assert worker.build_connector_worker_meta().completed_store_events == {1: 1}
    assert worker.build_connector_worker_meta() is None

    worker._submit_transfer([], [], 2, is_store=False)
    assert worker.get_finished(set()) == (None, {"req-1"})
    assert worker.get_finished(set()) == (None, None)

    worker.clear_connector_metadata()
    assert worker._connector_metadata is None


def test_recompute_cpu_offload_worker_preempt_and_load_entrypoints():
    worker = RecomputeCPUOffloadWorker.__new__(RecomputeCPUOffloadWorker)
    worker._submit_transfer = MagicMock()
    worker._flush_and_sync_all = MagicMock()
    worker._connector_metadata = None
    metadata = RecomputeCPUOffloadMetadata(
        need_flush=True,
        preempt_store_event=3,
        preempt_store_gpu_blocks=[1],
        preempt_store_cpu_blocks=[2],
        preempt_load_event=4,
        preempt_load_gpu_blocks=[5],
        preempt_load_cpu_blocks=[6],
    )

    worker.handle_preemptions(metadata)

    worker._flush_and_sync_all.assert_called_once_with()
    worker._submit_transfer.assert_called_once_with(
        [1],
        [2],
        3,
        is_store=True,
        sync=True,
    )

    worker._submit_transfer.reset_mock()
    worker.start_load_kv()
    worker._submit_transfer.assert_not_called()

    worker._connector_metadata = metadata
    worker.start_load_kv()
    worker._submit_transfer.assert_called_once_with(
        [6],
        [5],
        4,
        is_store=False,
        sync=True,
    )


def test_recompute_cpu_offload_worker_wait_for_layer_load_once():
    worker = RecomputeCPUOffloadWorker.__new__(RecomputeCPUOffloadWorker)
    stream = MagicMock()
    current_stream = MagicMock()
    worker.load_stream = stream
    worker._connector_metadata = RecomputeCPUOffloadMetadata(preempt_load_event=1)
    worker._load_stream_waited = False

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.worker.torch.npu.current_stream",
        return_value=current_stream,
    ):
        worker.wait_for_layer_load()
        worker.wait_for_layer_load()

    current_stream.wait_stream.assert_called_once_with(stream)
    assert worker._load_stream_waited is True


@pytest.fixture
def cpu_allocation(monkeypatch):
    zeros = torch.zeros
    allocator = MagicMock(side_effect=lambda *args, **kwargs: zeros(*args, **{**kwargs, "pin_memory": False}))
    monkeypatch.setattr(torch, "zeros", allocator)
    return allocator


@pytest.mark.parametrize("capacity", [1, 32])
def test_registration_deduplicates_aliases_and_scales_blocks(cpu_allocation, capacity):
    first = torch.arange(16, dtype=torch.int8).reshape(4, 4)
    second = torch.arange(8, dtype=torch.int8).reshape(2, 4)
    config = SimpleNamespace(
        num_blocks=2,
        kv_cache_tensors=[KVCacheTensor(size=24, shared_by=["a", "b"]), KVCacheTensor(size=99, shared_by=[])],
    )
    worker = RecomputeCPUOffloadWorker(SimpleNamespace(), config, capacity)
    worker.register_kv_caches({"a": (first, second, first), "b": second, "c": first})

    assert worker.num_cpu_blocks == max(1, 2 * capacity // 24)
    assert worker.block_size_scale == {"a.0": 2, "a.1": 1}
    assert set(worker.gpu_kv_caches) == {"a.0", "a.1"}
    assert worker.gpu_kv_caches["a.0"].data_ptr() == first.data_ptr()
    assert worker.cpu_kv_caches["a.0"].shape == (worker.num_cpu_blocks * 2, 4)
    assert worker.cpu_kv_caches["a.1"].shape == (worker.num_cpu_blocks, 4)
    assert torch.count_nonzero(worker.cpu_kv_caches["a.0"]) == 0
    assert all(call.kwargs["pin_memory"] for call in cpu_allocation.call_args_list)
    assert torch.npu.Stream.call_count == 2


def test_empty_and_single_tensor_registration(cpu_allocation):
    config = SimpleNamespace(num_blocks=2, kv_cache_tensors=[KVCacheTensor(size=8, shared_by=["a"])])
    worker = RecomputeCPUOffloadWorker(SimpleNamespace(), config, 16)
    worker.register_kv_caches({})
    assert worker.gpu_kv_caches is None
    torch.npu.Stream.assert_not_called()

    tensor = torch.arange(8, dtype=torch.int8).reshape(2, 4)
    worker.register_kv_caches({"a": tensor, "alias": tensor})
    assert worker.num_cpu_blocks == 4
    assert list(worker.cpu_kv_caches) == ["a"]
    assert worker.cpu_kv_caches["a"].shape == (4, 4)


@pytest.mark.parametrize(("is_store", "sync"), [(True, True), (False, True), (False, False)])
def test_transfer_copies_real_tensor_blocks_and_reports_completion(is_store, sync):
    worker = RecomputeCPUOffloadWorker(SimpleNamespace(), None, 0)
    worker.gpu_kv_caches = {"plain": torch.arange(12).reshape(3, 4), "scaled": torch.arange(24).reshape(6, 4)}
    worker.cpu_kv_caches = {"plain": torch.full((3, 4), -1), "scaled": torch.full((6, 4), -2)}
    worker.block_size_scale = {"plain": 1, "scaled": 2}
    worker.load_stream, worker.store_stream = MagicMock(), MagicMock()
    metadata = RecomputeCPUOffloadMetadata(preempt_load_event=4, preempt_load_event_to_reqs={4: ["req-a", "req-b"]})
    worker.bind_connector_metadata(metadata)
    source = worker.gpu_kv_caches if is_store else worker.cpu_kv_caches
    destination = worker.cpu_kv_caches if is_store else worker.gpu_kv_caches
    expected_plain = source["plain"][1].clone()
    expected_scaled = source["scaled"][2:4].clone()
    untouched_plain = destination["plain"][:2].clone()
    untouched_scaled = destination["scaled"][:4].clone()

    worker._submit_transfer([1], [2], 4, is_store=is_store, sync=sync)

    assert torch.equal(destination["plain"][2], expected_plain)
    assert torch.equal(destination["scaled"][4:6], expected_scaled)
    assert torch.equal(destination["plain"][:2], untouched_plain)
    assert torch.equal(destination["scaled"][:4], untouched_scaled)
    event = torch.npu.Event.return_value
    event.record.assert_called_once_with(worker.store_stream if is_store else worker.load_stream)
    if sync:
        event.synchronize.assert_called_once_with()
    else:
        event.synchronize.assert_not_called()
        event.query.return_value = False
        assert worker.get_finished(set()) == (None, None)
        assert worker._load_events == [(4, event)]
        event.query.return_value = True
    if is_store:
        assert worker.build_connector_worker_meta().completed_store_events == {4: 1}
    else:
        assert worker.get_finished(set()) == (None, {"req-a", "req-b"})
        assert worker._pending_load_event_indices == set()
        assert worker._submitted_load_event_indices == set()
    assert worker._load_events == []


def test_invalid_and_duplicate_events_do_not_issue_copies():
    worker = RecomputeCPUOffloadWorker(SimpleNamespace(), None, 0)
    worker._submit_transfer([], [], -1, is_store=False)
    assert worker._submitted_load_event_indices == set()
    worker._submit_transfer([], [], 2, is_store=False)
    worker._submit_transfer([0], [0], 2, is_store=False)
    assert worker._load_hwm == 2
    torch.npu.synchronize.assert_not_called()
    with pytest.raises(AssertionError):
        worker._submit_transfer([0], [], 3, is_store=True)
    assert worker._completed_store_events == {}


def test_flush_synchronizes_all_events_and_clears_submission_state():
    worker = RecomputeCPUOffloadWorker(SimpleNamespace(), None, 0)
    events = [MagicMock(), MagicMock()]
    worker._load_events = [(2, events[0]), (4, events[1])]
    worker._submitted_load_event_indices = {2, 4}
    worker.handle_preemptions(RecomputeCPUOffloadMetadata(need_flush=True))
    for event in events:
        event.synchronize.assert_called_once_with()
    assert worker._load_hwm == 4
    assert worker._load_events == []
    assert worker._submitted_load_event_indices == set()
    worker.handle_preemptions(RecomputeCPUOffloadMetadata(need_flush=False))
    assert worker._load_hwm == 4


@pytest.mark.parametrize("metadata", [None, RecomputeCPUOffloadMetadata()])
def test_wait_without_load_metadata_does_not_insert_barrier(metadata):
    worker = RecomputeCPUOffloadWorker(SimpleNamespace(), None, 0)
    worker.load_stream = MagicMock()
    if metadata is not None:
        worker.bind_connector_metadata(metadata)
    worker.wait_for_layer_load()
    assert worker.get_finished(set()) == (None, None)
    assert worker._pending_load_event_indices == set()
    torch.npu.current_stream.assert_not_called()
