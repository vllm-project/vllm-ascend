import threading
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
    MooncakeSessionTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.adapter import (
    KVCacheStoreKeyLayerSendingProcessAdapter,
    KVCacheStoreLayerRecvingProcessAdapter,
    KVCacheStoreLayerSendingProcessAdapter,
    KVCacheStoreRecvingProcessAdapter,
    KVCacheStoreSendingProcessAdapter,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer import KVTransferProcess


def database():
    db = ChunkedTokenDatabase([KeyMetadata("test", 1, 2, 1, 3)], [2], None, hash_block_size=2)
    db.set_group_buffers({0: [100]}, {0: [2]}, {0: [2]})
    return db


def request(req_id="request"):
    return ReqMeta(req_id, 4, [[1, 3]], [b"a" * 32, b"b" * 32], can_save=True)


def process_endpoint(cls):
    process = MagicMock()
    process.client.timeout = 1
    process.client.wait.side_effect = lambda future: future.result(timeout=1)
    process.submit_request.side_effect = lambda *args: Future()
    endpoint = cls(MagicMock(), database(), [2], 0, process=process)
    return process, endpoint


def completed(future, *, invalid_blocks=(), operations=()):
    future.set_result(
        {
            "finished": True,
            "events": [],
            "invalid_blocks": list(invalid_blocks),
            "operations": list(operations),
        }
    )


def test_block_adapters_publish_only_current_completions_and_metrics():
    process, sender = process_endpoint(KVCacheStoreSendingProcessAdapter)
    futures: list[Future] = [Future(), Future(), Future()]
    process.submit_request.side_effect = futures
    req = request()
    for _ in range(2):
        sender.add_stored_request(req.req_id)
        sender.add_request(req)
    completed(futures[0])
    assert sender.get_and_clear_finished_requests() == set()
    sender.delete_finished_stored_request(req.req_id)
    sender.discard_finished_requests({req.req_id})
    sender.add_stored_request(req.req_id)
    sender.add_request(req)
    completed(futures[1])
    assert sender.get_stored_request_count(req.req_id) == 1
    completed(futures[2])
    sender.wait_for_pending()
    assert sender.get_and_clear_finished_requests() == {req.req_id}
    assert not sender._generations

    # A completion from a preempted load generation must not leak either a
    # finished request or its invalid blocks into the next generation.
    process, receiver = process_endpoint(KVCacheStoreRecvingProcessAdapter)
    future: Future = Future()
    process.submit_request.side_effect = [future]
    receiver.add_request(request())
    receiver.discard_finished_requests({"request"})
    completed(future, invalid_blocks=[1])
    assert receiver.get_and_clear_finished_requests() == set()
    assert not receiver._invalid_block_ids

    # Current completions still forward the child's backend measurements.
    process = MagicMock()
    process.client.timeout = 1
    metrics_future: Future = Future()
    process.submit_request.return_value = metrics_future
    record_operation = MagicMock()
    receiver = KVCacheStoreRecvingProcessAdapter(
        MagicMock(),
        database(),
        [2],
        0,
        process=process,
        record_operation=record_operation,
    )
    receiver.add_request(request())

    completed(metrics_future, operations=[("load_get", 0.25, 3)])
    receiver.wait_for_pending()

    record_operation.assert_called_once_with("load_get", 0.25, 3)


def test_failed_async_transfer_is_raised_by_waiter():
    process, sender = process_endpoint(KVCacheStoreSendingProcessAdapter)
    future: Future = Future()
    process.submit_request.side_effect = [future]
    sender.add_request(request())
    future.set_exception(RuntimeError("child died"))
    with pytest.raises(RuntimeError, match="asynchronous transfer"):
        sender.wait_for_pending()


def test_layer_adapters_apply_child_outcomes_to_parent_state():
    tracker = MooncakeSessionTracker()
    tracker.register_put_keys("request", [("committed", 0), ("revoked", 1)])
    started_keys = {"committed", "revoked"}
    sender = KVCacheStoreLayerSendingProcessAdapter(
        m_store=MagicMock(),
        token_database=database(),
        block_size=2,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        page_size_bytes=2,
        ready_event=threading.Event(),
        num_layers=1,
        layer_save_finished_events=[threading.Event()],
        sync_save_events=[MagicMock()],
        group_builders=[MagicMock()],
        put_started_keys=started_keys,
        session_tracker=tracker,
        process=MagicMock(),
    )
    try:
        sender._complete_store(
            {
                "completed_req_ids": [],
                "finished_req_ids": [],
                "events": [],
                "committed_keys": ["committed"],
                "revoked_keys": ["revoked"],
            },
            [],
            0,
        )
        assert tracker.prepare_load_entries("request", []) == [("committed", 0)]
        assert started_keys == set()
    finally:
        sender.close()

    invalid_blocks: set[int] = set()
    load_aborted = threading.Event()
    receiver = KVCacheStoreLayerRecvingProcessAdapter(
        m_store=MagicMock(),
        token_database=database(),
        block_size=2,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        page_size_bytes=2,
        ready_event=threading.Event(),
        get_event=threading.Event(),
        layer_load_finished_events=[threading.Event()],
        layer_save_finished_events=[threading.Event()],
        sync_save_events=[MagicMock()],
        num_layers=1,
        group_builders=[MagicMock()],
        invalid_block_ids=invalid_blocks,
        load_abort_event=load_aborted,
        process=MagicMock(),
    )
    try:
        receiver._complete_load(
            {
                "finished_req_ids": [],
                "events": [],
                "invalid_blocks": [3],
                "load_aborted": True,
            },
            0,
        )
        assert invalid_blocks == {3}
        assert load_aborted.is_set()
    finally:
        receiver.close()


def test_layer_receive_process_preserves_parent_ordering():
    order = []

    def record(item, result):
        order.append(item)
        return result

    process = MagicMock()
    process.client.timeout = 1
    process.client.wait.side_effect = lambda _future: record(
        "wait_child", {"completed_req_ids": ["request"], "finished_req_ids": [], "events": []}
    )
    process.submit_layer_request.side_effect = lambda *_args: record("submit_child", Future())
    save_finished = MagicMock()
    save_finished.wait.side_effect = lambda timeout: record("wait_save", True)
    save_finished.clear.side_effect = lambda: order.append("clear_save")
    sync_event = MagicMock()
    sync_event.synchronize.side_effect = lambda: order.append("sync_save")
    gate = MagicMock()
    gate.wait.side_effect = lambda timeout: record("wait_attention", True)
    receiver = KVCacheStoreLayerRecvingProcessAdapter(
        m_store=MagicMock(),
        token_database=database(),
        block_size=2,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        page_size_bytes=2,
        ready_event=threading.Event(),
        get_event=threading.Event(),
        layer_load_finished_events=[threading.Event()],
        layer_save_finished_events=[save_finished],
        sync_save_events=[sync_event],
        num_layers=1,
        group_builders=[MagicMock()],
        external_slot_release_waiter=lambda _layer_id: order.append("release_slot"),
        process=process,
    )
    receiver._stagger_h2d_submit = lambda _layer_id: order.append("stagger_load")
    task = LayerTransferTask(0, [LayerBlockRange(request(), 0, 1)])
    data = LayerLoadTask(0, [task], 0, gate)
    try:
        receiver._coordinate_load(data)
    finally:
        receiver.close()
    assert order == [
        "wait_save",
        "sync_save",
        "clear_save",
        "wait_attention",
        "stagger_load",
        "release_slot",
        "submit_child",
        "wait_child",
    ]


def test_key_layer_sending_process_completes_parent_request_state():
    process = MagicMock()
    process.client.timeout = 1
    future: Future[dict[str, object]] = Future()
    process.submit_layer_request.return_value = future
    process.client.wait.side_effect = lambda child_future: child_future.result(timeout=1)
    save_finished = threading.Event()
    sender = KVCacheStoreKeyLayerSendingProcessAdapter(
        m_store=MagicMock(),
        token_database=database(),
        block_size=2,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        ready_event=threading.Event(),
        num_layers=1,
        layer_save_finished_events=[save_finished],
        sync_save_events=[MagicMock()],
        process=process,
    )
    task = LayerTransferTask(0, [LayerBlockRange(request(), 0, 1)])
    tasks = [task]
    sender.add_stored_request("request")
    sender.add_request(tasks)
    future.set_result(
        {
            "completed_req_ids": ["request"],
            "finished_req_ids": ["request"],
            "events": [],
        }
    )
    sender.wait_for_pending()
    assert sender.get_and_clear_finished_requests() == {"request"}
    assert "request" not in sender.stored_requests
    assert save_finished.is_set()
    assert tasks == []


def test_layer_send_process_waits_on_parent_event_before_child_submission():
    order = []

    def submit_child(*_args):
        order.append("submit_child")
        return Future()

    def wait_child(_future):
        order.append("wait_child")
        return {
            "completed_req_ids": [],
            "finished_req_ids": [],
            "events": [],
        }

    process = MagicMock()
    process.client.timeout = 1
    process.submit_layer_request.side_effect = submit_child
    process.client.wait.side_effect = wait_child
    save_event = MagicMock()
    save_event.synchronize.side_effect = lambda: order.append("sync_parent_event")
    sender = KVCacheStoreKeyLayerSendingProcessAdapter(
        m_store=MagicMock(),
        token_database=database(),
        block_size=2,
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        put_step=1,
        ready_event=threading.Event(),
        num_layers=1,
        layer_save_finished_events=[threading.Event()],
        sync_save_events=[save_event],
        process=process,
    )
    tasks = [LayerTransferTask(0, [LayerBlockRange(request(), 0, 1)])]
    try:
        sender.add_request(tasks)
        sender.wait_for_pending()
    finally:
        sender.close()

    assert order == ["sync_parent_event", "submit_child", "wait_child"]
    process.submit_layer_request.assert_called_once()
    assert process.submit_layer_request.call_args.args[0] == "store"
    assert process.submit_layer_request.call_args.args[2] == 0


def test_block_send_process_waits_on_parent_event_before_child_submission():
    order = []

    def submit_child(*_args):
        order.append("submit_child")
        return Future()

    def wait_child(_future):
        order.append("wait_child")
        return {"finished": True, "events": [], "invalid_blocks": [], "operations": []}

    process = MagicMock()
    process.client.timeout = 1
    process.submit_request.side_effect = submit_child
    process.client.wait.side_effect = wait_child
    req = request()
    req.current_event = MagicMock()
    req.current_event.synchronize.side_effect = lambda: order.append("sync_parent_event")
    sender = KVCacheStoreSendingProcessAdapter(MagicMock(), database(), [2], 0, process=process)
    try:
        sender.add_request(req)
        sender.wait_for_pending()
    finally:
        sender.close()

    assert order == ["sync_parent_event", "submit_child", "wait_child"]


def test_block_request_does_not_cross_process_with_worker_event():
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer.TransferProcess"):
        parent = KVTransferProcess({})
    parent.cache = MagicMock()
    req = request()
    req.current_event = MagicMock()

    parent.submit_request("store", req)

    payload = parent.client.submit.call_args.args[1]
    assert "current_event" not in payload
