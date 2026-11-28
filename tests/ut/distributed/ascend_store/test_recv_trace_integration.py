# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import AttentionLoadGate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVTransferThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker


@pytest.mark.parametrize("error", [ValueError("serialization"), OSError("write")])
def test_trace_close_failure_does_not_skip_lease_cleanup(error):
    trace = MagicMock()
    trace.close.side_effect = error
    receiver = MagicMock()
    receiver.is_alive.return_value = False
    receiver.recv_trace = trace
    store = MagicMock()
    store.batch_remove_lease.return_value = 0
    worker = SimpleNamespace(
        _closed=False, kv_send_thread=None, kv_recv_thread=receiver, _step_load_keys={"lease"}, m_store=store
    )
    LayerwiseKVPoolWorker.shutdown(worker)
    store.batch_remove_lease.assert_called_once_with(["lease"])
    store.close.assert_called_once()
    assert worker._closed
    assert not worker._step_load_keys


def test_prefetch_submits_layer_load_task():
    receiver = MagicMock()
    worker = SimpleNamespace(
        kv_recv_thread=receiver,
        layer_load_tasks=[[SimpleNamespace(group_id=0)]],
        prefetch_layer_map={},
        current_layer=0,
        next_layer_to_submit=0,
        num_prefetch_layers=0,
        num_layers=1,
    )
    LayerwiseKVPoolWorker._submit_ready_layer_loads(worker)
    task = receiver.add_request.call_args.args[0]
    assert task.layer_id == 0
    assert task.transfer_tasks == worker.layer_load_tasks[0]
    assert task.wait_for_save_layers == ()
    assert task.load_gate is None
    assert task.compute_boundary is None


@pytest.mark.parametrize("traced", [False, True])
def test_reuse_saves_keep_events_and_wait_for_every_owner(traced):
    trace = MagicMock() if traced else None
    cpu_events = [threading.Event(), threading.Event()]
    for event in cpu_events:
        event.set()
    device_events = [MagicMock(), MagicMock()]
    worker = SimpleNamespace(
        recv_trace=trace,
        _stopping=threading.Event(),
        layer_save_finished_events=cpu_events,
        sync_save_events=device_events,
        save_failure_checker=MagicMock(),
        external_slot_release_waiter=MagicMock(),
    )
    worker._trace_mark = lambda event, source=None: KVTransferThread._trace_mark(worker, event, source)
    worker._wait_for_event = lambda *args: KVTransferThread._wait_for_event(worker, *args)
    assert KVCacheStoreLayerRecvingThread._wait_for_reuse_saves(worker, SimpleNamespace(wait_for_save_layers=(0, 1)))
    assert all(event.is_set() for event in cpu_events)
    for event in device_events:
        event.synchronize.assert_called_once_with()
    assert [call.args for call in worker.external_slot_release_waiter.call_args_list] == [(0,), (1,)]
    if trace is not None:
        assert [call.args for call in trace.mark.call_args_list] == [
            (event, source)
            for source in (0, 1)
            for event in (
                "save_cpu_wait_begin",
                "save_cpu_wait_end",
                "save_npu_wait_begin",
                "save_npu_wait_end",
                "external_slot_wait_begin",
                "external_slot_wait_end",
            )
        ]


@pytest.mark.parametrize("result", [0, 128, -1])
def test_range_backend_result_is_recorded(result):
    trace = MagicMock()
    worker = SimpleNamespace(
        _active_load_indices=None,
        _load_abort_event=threading.Event(),
        _stagger_h2d_submit=MagicMock(),
        _range_transfer_batches=KVTransferThread._range_transfer_batches,
        max_transfer_blocks=0,
        max_transfer_bytes=0,
        m_store=MagicMock(),
        _trace_mark=trace.mark,
        _mark_invalid_range_indices=MagicMock(),
        final_layer_id=1,
    )
    worker.m_store.batch_copy_get.return_value = [result]
    metadata = SimpleNamespace(layer_id=0, keys=["key"], all_buffers=[[1000]], all_sizes=[[128]], all_offsets=[[0]])
    KVCacheStoreLayerRecvingThread._handle_range_request(worker, metadata, SimpleNamespace())
    assert [call.args[0] for call in trace.mark.call_args_list] == (
        ["backend_submit", "backend_complete"] + (["backend_error"] if result < 0 else [])
    )
    assert worker._active_load_indices == (set() if result < 0 else {0})
    if result < 0:
        worker._mark_invalid_range_indices.assert_called_once_with(metadata, [0])
    else:
        worker._mark_invalid_range_indices.assert_not_called()


def test_transfer_wait_uses_load_gate_observer():
    gate = AttentionLoadGate()
    gate._event = MagicMock()
    trace = MagicMock()
    worker = SimpleNamespace(recv_trace=trace, _stopping=threading.Event())
    assert KVTransferThread._wait_for_event(worker, gate)
    gate._event.synchronize.assert_called_once_with()
    assert [call.args[0] for call in trace.mark.call_args_list] == [
        "gate_record_wait_begin",
        "gate_record_wait_end",
        "gate_npu_wait_begin",
        "gate_npu_wait_end",
    ]


@pytest.mark.parametrize("traced", [False, True])
def test_compute_boundary_trace_preserves_receive_completion(traced):
    trace = MagicMock() if traced else None
    boundary = MagicMock()
    finished = threading.Event()
    worker = SimpleNamespace(
        recv_trace=trace,
        _wait_for_reuse_saves=lambda task: True,
        external_slot_release_waiter=None,
        layer_load_finished_events=[finished],
        request_queue=MagicMock(),
    )
    worker._trace_mark = lambda event: KVTransferThread._trace_mark(worker, event)
    task = SimpleNamespace(layer_id=0, compute_boundary=boundary, load_gate=None, transfer_tasks=[])
    KVCacheStoreLayerRecvingThread._handle_range_layer_task(worker, task)
    boundary.synchronize.assert_called_once_with()
    assert finished.is_set()
    worker.request_queue.task_done.assert_called_once_with()
    if trace is not None:
        assert [call.args[0] for call in trace.mark.call_args_list] == [
            "compute_boundary_wait_begin",
            "compute_boundary_wait_end",
        ]


@pytest.mark.parametrize("traced", [False, True])
def test_shutdown_closes_trace_after_join_and_before_store(traced):
    lifecycle = MagicMock()
    lifecycle.receiver.recv_trace = lifecycle.trace if traced else None
    lifecycle.sender.is_alive.return_value = True
    lifecycle.receiver.is_alive.return_value = True
    lifecycle.store.batch_remove_lease.return_value = 0
    worker = SimpleNamespace(
        _closed=False,
        kv_send_thread=lifecycle.sender,
        kv_recv_thread=lifecycle.receiver,
        m_store=lifecycle.store,
        _step_load_keys={"key"},
    )
    LayerwiseKVPoolWorker.shutdown(worker)
    LayerwiseKVPoolWorker.shutdown(worker)
    expected = [
        call.sender.stop(),
        call.receiver.stop(),
        call.sender.is_alive(),
        call.sender.join(),
        call.receiver.is_alive(),
        call.receiver.join(),
    ]
    if traced:
        expected.append(call.trace.close())
    expected.extend([call.store.batch_remove_lease(["key"]), call.store.close()])
    assert lifecycle.mock_calls == expected
    assert worker._closed
    assert not worker._step_load_keys
