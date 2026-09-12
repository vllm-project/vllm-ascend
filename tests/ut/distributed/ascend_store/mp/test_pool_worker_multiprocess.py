import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LayerBlockRange,
    LayerTransferTask,
    ReqMeta,
)


def request():
    return ReqMeta("request", 4, [[1, 3]], [b"a" * 32, b"b" * 32], can_save=True)


def test_worker_process_configuration_preserves_supported_modes():
    parallel_config = SimpleNamespace(
        rank=3,
        data_parallel_index=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=2,
    )
    cases = (
        ("hybrid", dict(use_hybrid=True), False),
        ("compressed", dict(use_compress=True), True),
        ("tp_mismatch", dict(tp_mismatch=True), False),
    )
    for name, flags, lazy_init in cases:
        case = unittest.TestCase()
        try:
            worker = make_worker(case)
            worker.use_multiprocess = True
            worker.use_hybrid = flags.get("use_hybrid", False)
            worker.use_compress = flags.get("use_compress", False)
            worker.tp_mismatch = flags.get("tp_mismatch", False)
            with (
                case.subTest(name=name),
                patch(
                    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.KVTransferProcess"
                ) as factory,
            ):
                worker._init_backend(parallel_config, worker._extra_config)

                config = factory.call_args.args[0]
                assert worker.transfer_process is factory.return_value
                assert worker.m_store is factory.return_value
                assert config["global_rank"] == 7
                assert config["tp_rank"] == worker.tp_rank
                assert config["kv_role"] == "kv_producer"
                assert config["lazy_init"] is lazy_init
        finally:
            case.doCleanups()

    # Exercise the user-facing config path as well as the direct mode matrix.
    case = unittest.TestCase()
    try:
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.KVTransferProcess") as factory:
            worker = make_worker(
                case,
                use_layerwise=True,
                extra_config={"backend": "mooncake", "use_multiprocess": True},
            )
        assert worker.transfer_process is factory.return_value
        assert worker.m_store is factory.return_value
    finally:
        case.doCleanups()


@pytest.mark.parametrize(
    "use_gva,use_key_major_ranges,sender_name,receiver_name",
    [
        (False, False, "KVCacheStoreKeyLayerSendingProcessAdapter", "KVCacheStoreKeyLayerRecvingProcessAdapter"),
        (True, False, "KVCacheStoreLayerSendingProcessAdapter", "KVCacheStoreLayerRecvingProcessAdapter"),
        (False, True, "KVCacheStoreLayerSendingProcessAdapter", "KVCacheStoreLayerRecvingProcessAdapter"),
    ],
)
def test_layerwise_worker_selects_matching_process_adapters(
    use_gva,
    use_key_major_ranges,
    sender_name,
    receiver_name,
):
    case = unittest.TestCase()
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.adapter"
    try:
        worker = make_worker(case, kv_role="kv_both", use_layerwise=True)
        worker.transfer_process = MagicMock()
        worker.m_store = worker.transfer_process
        worker.use_layerwise_transfer = use_gva
        worker.use_block_key_layerwise = use_key_major_ranges
        worker._transfer_threads_started = False
        worker.page_size_bytes = 2
        with (
            patch(f"{module}.{sender_name}") as sender,
            patch(f"{module}.{receiver_name}") as receiver,
            patch.object(torch.npu, "Event"),
            patch.object(worker, "_build_group_layer_builders", return_value=[MagicMock()]),
        ):
            worker._start_kv_transfer_threads()
        assert worker.kv_send_thread is sender.return_value
        assert worker.kv_recv_thread is receiver.return_value
        worker.transfer_process.bind_adapters.assert_called_once_with((sender.return_value, receiver.return_value))
        sender.return_value.start.assert_not_called()
        receiver.return_value.start.assert_not_called()
    finally:
        case.doCleanups()


def test_process_layerwise_wait_for_save_drains_adapter():
    case = unittest.TestCase()
    try:
        worker = make_worker(case, use_layerwise=True)
        worker.transfer_process = MagicMock()
        worker.kv_send_thread = MagicMock()

        worker.wait_for_save(MagicMock())

        worker.kv_send_thread.wait_for_pending.assert_called_once_with()
        worker.kv_send_thread.request_queue.join.assert_not_called()
    finally:
        case.doCleanups()


def test_process_layer_save_records_a_fresh_event_before_submission(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

    worker = KVPoolWorker.__new__(KVPoolWorker)
    old_event = MagicMock()
    fresh_event = MagicMock()
    order = []
    fresh_event.record.side_effect = lambda: order.append("record")
    send_thread = MagicMock()
    send_thread.add_request.side_effect = lambda _tasks: order.append("submit")
    req = request()
    task = LayerTransferTask(0, [LayerBlockRange(req, 0, 1)])
    worker.current_layer = 0
    worker.num_layers = 2
    worker.sync_save_events = [old_event, MagicMock()]
    worker.layer_save_finished_events = [threading.Event(), threading.Event()]
    worker.kv_send_thread = send_thread
    worker.transfer_process = MagicMock()
    worker.layer_save_tasks = [[task], []]
    worker.prefetch_layer_map = {}
    monkeypatch.setattr(torch.npu, "Event", MagicMock(return_value=fresh_event))

    worker.save_kv_layer(MagicMock())

    torch.npu.Event.assert_called_once_with()
    assert worker.sync_save_events[0] is fresh_event
    assert order == ["record", "submit"]
    send_thread.add_stored_request.assert_called_once_with(req.req_id)
