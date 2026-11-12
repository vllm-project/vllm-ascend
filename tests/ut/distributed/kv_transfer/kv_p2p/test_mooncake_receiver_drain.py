# SPDX-License-Identifier: Apache-2.0
"""Real receiver queue/executor semantics, including delayed final ACK."""

import queue
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import KVCacheRecvingThread


def make_receiver():
    receiver = KVCacheRecvingThread.__new__(KVCacheRecvingThread)
    threading.Thread.__init__(receiver, daemon=True)
    receiver.ready_event = threading.Event()
    receiver.request_queue = queue.Queue()
    receiver._managed_shutdown = True
    receiver._accept_lock = threading.Lock()
    receiver._stopping = False
    receiver._executor_stopped = threading.Event()
    receiver._executor_stop_thread = None
    receiver._executor_stop_error = None
    receiver.executor = ThreadPoolExecutor(max_workers=2)
    receiver.peer_request_queues = defaultdict(deque)
    receiver.active_peer_request_handlers = set()
    receiver.peer_request_queues_lock = threading.Lock()
    vars(receiver)["_mark_request_task_submitted"] = lambda request: None
    return receiver


def enqueue(receiver, index):
    receiver.add_request(str(index), str(index), [], [], "engine", "peer", 1, 0, 1)


def test_drain_waits_for_resubmitted_work_and_final_ack():
    receiver = make_receiver()
    first = threading.Event()
    proceed = threading.Event()
    ack_started = threading.Event()
    ack = threading.Event()
    completed = []

    def handle(request):
        index = int(request["request_id"])
        if index == 0:
            first.set()
            assert proceed.wait(5)
        completed.append(index)
        receiver.request_queue.task_done()
        if index == 7:
            ack_started.set()
            assert ack.wait(5)

    receiver._handle_request = handle
    receiver.start()
    enqueue(receiver, 0)
    assert first.wait(5)
    for index in range(1, 8):
        enqueue(receiver, index)
    proceed.set()
    assert ack_started.wait(5)
    # Eight jobs exceed the five-job peer-handler batch and force resubmission.
    with pytest.raises(TimeoutError, match="acknowledgements"):
        receiver.drain_and_stop(timeout=0.2)
    with pytest.raises(RuntimeError, match="shutting down"):
        enqueue(receiver, 8)
    assert completed == list(range(8))
    ack.set()
    receiver.drain_and_stop(timeout=5)
    receiver.drain_and_stop(timeout=5)
    assert not receiver.is_alive()


def make_connector(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_hybrid_connector as module

    connector = object.__new__(module.MooncakeConnector)
    engine = object()
    receiver = SimpleNamespace(engine=engine, drain_and_stop=Mock())
    worker = SimpleNamespace(kv_role="kv_consumer", kv_recv_thread=receiver, kv_send_thread=None, engine=engine)
    monkeypatch.setattr(connector, "connector_worker", worker, raising=False)
    connector._kv_transfer_config = SimpleNamespace(kv_connector="MooncakeHybridConnector")
    connector._vllm_config = SimpleNamespace(ec_transfer_config=None)
    engine_state = SimpleNamespace(transfer_engine=engine, release_exclusive_engine=Mock())
    monkeypatch.setattr(module, "global_te", engine_state)
    monkeypatch.setattr(module, "get_ascend_config", lambda: SimpleNamespace(engram_local_vmm=True))
    return connector, worker, receiver, engine_state


def test_connector_retains_engine_when_receiver_does_not_drain(monkeypatch):
    connector, worker, receiver, engine_state = make_connector(monkeypatch)
    receiver.drain_and_stop.side_effect = TimeoutError("pending ACK")
    with pytest.raises(TimeoutError, match="pending ACK"):
        connector.shutdown()
    engine_state.release_exclusive_engine.assert_not_called()
    assert connector.connector_worker is worker
    assert receiver.engine is worker.engine is engine_state.transfer_engine


def test_connector_releases_only_after_receiver_drains(monkeypatch):
    connector, worker, receiver, engine_state = make_connector(monkeypatch)

    def release(owner, engine):
        receiver.drain_and_stop.assert_called_once()
        assert owner is worker and receiver.engine is worker.engine is engine_state.transfer_engine is engine
        engine_state.transfer_engine = None

    engine_state.release_exclusive_engine.side_effect = release
    connector.shutdown()
    assert connector.connector_worker is None
    assert receiver.engine is worker.engine is engine_state.transfer_engine is None
    connector.shutdown()


def test_connector_retains_engine_on_shared_ownership(monkeypatch):
    connector, worker, receiver, engine_state = make_connector(monkeypatch)
    connector._vllm_config.ec_transfer_config = object()
    with pytest.raises(RuntimeError, match="shared"):
        connector.shutdown()
    receiver.drain_and_stop.assert_not_called()
    engine_state.release_exclusive_engine.assert_not_called()
    assert receiver.engine is worker.engine is engine_state.transfer_engine


def test_connector_retains_engine_on_unregister_failure(monkeypatch):
    connector, worker, receiver, engine_state = make_connector(monkeypatch)
    engine_state.release_exclusive_engine.side_effect = RuntimeError("unregister failed")
    with pytest.raises(RuntimeError, match="unregister failed"):
        connector.shutdown()
    receiver.drain_and_stop.assert_called_once()
    assert connector.connector_worker is worker
    assert receiver.engine is worker.engine is engine_state.transfer_engine


def test_producer_waits_for_peer_completion_before_stopping():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
        KVCacheSendingThread,
        KVCacheTaskTracker,
    )

    sender = KVCacheSendingThread.__new__(KVCacheSendingThread)
    threading.Thread.__init__(sender, daemon=True)
    sender._stop_requested = threading.Event()
    sender.task_tracker = KVCacheTaskTracker(managed_shutdown=True)
    sender.task_tracker.add_req_to_process("pending")
    vars(sender)["run"] = lambda: sender._stop_requested.wait(5)
    sender.start()
    with pytest.raises(TimeoutError, match="producer requests"):
        sender.drain_and_stop(timeout=0.1)
    assert sender.is_alive() and not sender._stop_requested.is_set()
    sender.task_tracker.update_done_task_count("pending")
    sender.drain_and_stop(timeout=5)
    assert not sender.is_alive()


def test_producer_engine_is_retained_when_remote_read_is_pending(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    connector, worker, receiver, engine_state = make_connector(monkeypatch)
    sender = SimpleNamespace(drain_and_stop=Mock(side_effect=TimeoutError("pending remote read")))
    worker.kv_role = "kv_producer"
    worker.kv_recv_thread = None
    worker.kv_send_thread = sender
    with pytest.raises(TimeoutError, match="pending remote read"):
        connector.shutdown()
    engine_state.release_exclusive_engine.assert_not_called()
    assert connector.connector_worker is worker
    assert worker.engine is engine_state.transfer_engine


def test_producer_does_not_treat_expired_request_as_completed():
    import time

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
        KVCacheSendingThread,
        KVCacheTaskTracker,
    )

    sender = KVCacheSendingThread.__new__(KVCacheSendingThread)
    threading.Thread.__init__(sender, daemon=True)
    sender._stop_requested = threading.Event()
    sender.task_tracker = KVCacheTaskTracker(managed_shutdown=True)
    sender.task_tracker.add_req_to_process("expired")
    sender.task_tracker.add_delayed_request("expired", time.time() - 10**9)
    assert sender.task_tracker.get_and_clear_finished_requests() == {"expired"}
    with pytest.raises(RuntimeError, match="unknown transfer state"):
        sender.drain_and_stop(timeout=0.1)
    assert not sender._stop_requested.is_set()


def test_default_sender_uses_blocking_receive_without_polling():
    from unittest.mock import Mock

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import KVCacheSendingThread

    sender = KVCacheSendingThread.__new__(KVCacheSendingThread)
    sender._managed_shutdown = False
    sender._stop_requested = threading.Event()
    vars(sender)["metadata"] = None
    sock = Mock()
    sock.recv_multipart.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        sender.run_busy_loop(sock)
    sock.poll.assert_not_called()
    sock.recv_multipart.assert_called_once()


def test_default_request_admission_keeps_original_unlocked_path():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import KVCacheTaskTracker

    class UnexpectedLock:
        def __enter__(self):
            raise AssertionError("default request path took the VMM lock")

    tracker = KVCacheTaskTracker()
    vars(tracker)["done_task_lock"] = UnexpectedLock()
    tracker.add_req_to_process("request")
    assert tracker.reqs_to_process == {"request"}
    receiver = make_receiver()
    receiver._managed_shutdown = False
    vars(receiver)["_accept_lock"] = UnexpectedLock()
    enqueue(receiver, 7)
    assert receiver.request_queue.get_nowait()["request_id"] == "7"
    receiver.executor.shutdown()


def test_default_receiver_continues_after_none_request():
    receiver = make_receiver()
    receiver._managed_shutdown = False
    received = []

    def submit(request):
        received.append(request)
        raise KeyboardInterrupt

    vars(receiver)["_submit_request"] = submit
    receiver.request_queue.put(None)
    receiver.request_queue.put({"request_id": "next"})
    with pytest.raises(KeyboardInterrupt):
        receiver.run()
    assert received == [{"request_id": "next"}]
    receiver.executor.shutdown()


@pytest.mark.parametrize("connector,ec", [("MultiConnector", None), ("MooncakeHybridConnector", object())])
def test_vmm_rejects_shared_engine_before_connector_initialization(monkeypatch, connector, ec):
    from types import SimpleNamespace

    from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_hybrid_connector as module

    monkeypatch.setattr(module, "get_ascend_config", lambda: SimpleNamespace(engram_local_vmm=True))
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector=connector),
        ec_transfer_config=ec,
    )
    with pytest.raises(ValueError, match="exclusive MooncakeHybridConnector"):
        module.MooncakeConnector(config, module.KVConnectorRole.WORKER)


@pytest.mark.parametrize(
    "enabled,connector,ec",
    [(False, "MultiConnector", object()), (True, "MooncakeHybridConnector", None)],
)
def test_connector_init_preserves_supported_configurations(monkeypatch, enabled, connector, ec):
    from types import SimpleNamespace

    from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_hybrid_connector as module

    monkeypatch.setattr(module, "get_ascend_config", lambda: SimpleNamespace(engram_local_vmm=enabled))
    monkeypatch.setattr(module.KVConnectorBase_V1, "__init__", lambda self, *args: None)
    monkeypatch.setattr(module, "MooncakeConnectorWorker", lambda *args: "worker-created")
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector=connector, engine_id="test"),
        ec_transfer_config=ec,
    )
    instance = module.MooncakeConnector(config, module.KVConnectorRole.WORKER)
    assert instance.connector_worker == "worker-created"
