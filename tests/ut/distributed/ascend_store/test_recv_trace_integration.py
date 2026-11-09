# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreKeyLayerRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from tools.kv_pool.analyze_recv_trace import summarize


@pytest.mark.parametrize("result", [[0], [1], None])
def test_legacy_key_backend_result_is_recorded(result):
    events: list[dict] = []

    def mark(name):
        events.append({"task_id": 0, "ts_ns": len(events), "event": name})

    database = MagicMock()
    database.prepare_value_layer.return_value = (1000, 128, None)
    database._make_key_by_hash.return_value.split_layers.return_value = [MagicMock(), MagicMock()]
    store = MagicMock()
    store.get.return_value = result
    worker = SimpleNamespace(
        recv_trace=object(),
        _trace_mark=mark,
        token_database=database,
        final_layer_id=1,
        layer_offset=0,
        _get_block_size=lambda group: 16,
        tp_rank=0,
        tp_size=1,
        m_store=store,
        layer_load_finished_events=[threading.Event()],
        request_queue=MagicMock(),
        get_event=threading.Event(),
    )
    request = SimpleNamespace(req_id="request", is_last_chunk=True, block_hashes=["hash"], block_ids=[0])
    block = SimpleNamespace(request=request, start_block=0, end_block=1)
    task = SimpleNamespace(
        wait_for_save_layer=None,
        layer_id=0,
        attention_start_gate=None,
        transfer_tasks=[SimpleNamespace(block_ranges=[block])],
    )
    KVCacheStoreKeyLayerRecvingThread._handle_request(worker, task)
    report = summarize({"schema_version": 1, "dropped": 0, "events": events})
    assert bool(report["errors"]) == (result != [0])
    assert worker.layer_load_finished_events[0].is_set()
    worker.request_queue.task_done.assert_called_once()


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


def test_prefetch_submits_layer_load_task_for_key_receiver():
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
    assert task.wait_for_save_layer is None
    assert task.attention_start_gate is None
