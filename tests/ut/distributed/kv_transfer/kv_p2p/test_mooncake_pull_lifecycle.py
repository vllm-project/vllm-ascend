"""Lifecycle/shutdown tests for the pull mooncake v1 connector.

The tests build thread/worker/connector instances with ``__new__`` and only wire
the fields that the lifecycle code touches, so no vLLM worker bootstrap or
network handshake is required. No ZMQ sockets are created here.
"""

import queue
import threading
from typing import Any
from unittest.mock import MagicMock

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    SHUTDOWN_JOIN_TIMEOUT_SECONDS,
    KVCacheRecvingThread,
    KVCacheSendingThread,
    MooncakeConnector,
    MooncakeConnectorWorker,
)


def _make_worker(recv_thread, send_thread):
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker._shutdown_complete = False
    worker._shutdown_lock = threading.Lock()
    worker.engine_id = "engine-test"
    worker.kv_recv_thread = recv_thread
    worker.kv_send_thread = send_thread
    return worker


def _make_thread_mock(name, events, stop_error=None, join_error=None):
    thread = MagicMock(name=name)
    thread.name = name
    thread.is_alive.return_value = False

    def _stop():
        if stop_error is not None:
            raise stop_error
        events.append("stop:" + name)

    def _join(timeout=None):
        if join_error is not None:
            raise join_error
        events.append("join:" + name)

    thread.stop.side_effect = _stop
    thread.join.side_effect = _join
    return thread


def test_worker_shutdown_stops_threads_and_resources():
    events: list[Any] = []
    send_thread = _make_thread_mock("send", events)
    recv_thread = _make_thread_mock("recv", events)
    recv_thread.executor = MagicMock()
    recv_thread.executor.shutdown.side_effect = lambda wait: events.append("executor")
    recv_thread.remote_socket_pool = MagicMock()
    recv_thread.remote_socket_pool.close.side_effect = lambda: events.append("pool")

    worker = _make_worker(recv_thread, send_thread)
    worker.shutdown()
    worker.shutdown()

    # Idempotent: every action ran exactly once despite two shutdown() calls.
    assert recv_thread.stop.call_count == 1
    assert send_thread.stop.call_count == 1
    assert recv_thread.join.call_count == 1
    assert send_thread.join.call_count == 1
    assert recv_thread.executor.shutdown.call_count == 1
    assert recv_thread.remote_socket_pool.close.call_count == 1
    assert recv_thread.join.call_args.kwargs["timeout"] == SHUTDOWN_JOIN_TIMEOUT_SECONDS
    assert send_thread.join.call_args.kwargs["timeout"] == SHUTDOWN_JOIN_TIMEOUT_SECONDS
    recv_thread.executor.shutdown.assert_called_once_with(wait=True)
    # Order: stop -> join -> executor -> socket pool.
    assert events == ["stop:recv", "stop:send", "join:recv", "join:send", "executor", "pool"]


def test_worker_shutdown_without_registered_threads_is_safe():
    worker = _make_worker(None, None)
    worker.shutdown()
    worker.shutdown()


def test_worker_shutdown_tolerates_resource_failures():
    events: list[Any] = []
    send_thread = _make_thread_mock("send", events, stop_error=RuntimeError("stop failed"))
    recv_thread = _make_thread_mock("recv", events, join_error=RuntimeError("join failed"))
    recv_thread.executor = MagicMock()
    recv_thread.executor.shutdown.side_effect = RuntimeError("executor failed")
    recv_thread.remote_socket_pool = MagicMock()
    recv_thread.remote_socket_pool.close.side_effect = lambda: events.append("pool")

    worker = _make_worker(recv_thread, send_thread)
    worker.shutdown()  # best-effort: exceptions are logged, never raised

    # A failure in one step does not stop the remaining steps, or the retry-safe flag.
    assert send_thread.join.call_count == 1
    assert recv_thread.executor.shutdown.call_count == 1
    assert recv_thread.remote_socket_pool.close.call_count == 1
    assert worker._shutdown_complete is True


def test_connector_shutdown_delegates_and_is_idempotent():
    connector = MooncakeConnector.__new__(MooncakeConnector)
    connector._shutdown_complete = False
    connector.connector_worker = MagicMock()
    connector.connector_scheduler = None

    connector.shutdown()
    connector.shutdown()

    connector.connector_worker.shutdown.assert_called_once_with()


def test_connector_shutdown_scheduler_side_is_safe():
    connector = MooncakeConnector.__new__(MooncakeConnector)
    connector._shutdown_complete = False
    connector.connector_worker = None
    connector.connector_scheduler = MagicMock()

    connector.shutdown()
    connector.connector_scheduler.shutdown.assert_not_called()


def test_recving_thread_stop_exits_run_loop():
    thread = KVCacheRecvingThread.__new__(KVCacheRecvingThread)
    thread._stop_event = threading.Event()
    thread.request_queue = queue.Queue()
    thread.ready_event = threading.Event()
    thread._submit_request = MagicMock()

    runner = threading.Thread(target=thread.run, name="KVCacheRecvingThreadTest")
    runner.start()
    assert thread.ready_event.wait(2.0)
    thread.stop()
    runner.join(2.0)
    assert not runner.is_alive()
    assert thread._submit_request.call_count == 0


def test_recving_thread_run_consumes_requests_and_none_marker():
    thread = KVCacheRecvingThread.__new__(KVCacheRecvingThread)
    thread._stop_event = threading.Event()
    thread.request_queue = queue.Queue()
    thread.ready_event = threading.Event()
    received: list[Any] = []
    thread._submit_request = received.append

    thread.request_queue.put({"request_id": "r1"})
    thread.request_queue.put(None)
    thread.request_queue.put({"request_id": "r2"})

    runner = threading.Thread(target=thread.run, name="KVCacheRecvingThreadConsumeTest")
    runner.start()
    assert thread.ready_event.wait(2.0)
    deadline = threading.Event()
    for _ in range(100):
        if len(received) == 2:
            break
        deadline.wait(0.02)
    thread.stop()
    runner.join(2.0)
    assert not runner.is_alive()
    assert received == [{"request_id": "r1"}, {"request_id": "r2"}]


def test_sending_thread_stop_sets_event_and_is_idempotent():
    thread = KVCacheSendingThread.__new__(KVCacheSendingThread)
    thread._stop_event = threading.Event()

    thread.stop()
    assert thread._stop_event.is_set()
    thread.stop()
    assert thread._stop_event.is_set()


def test_shutdown_defaults_work_for_subclass_without_super_init():
    # Mirrors the d2rh facade, which overrides __init__ without super().__init__()
    # and therefore does not initialize the shutdown flag itself.
    class _D2RHStyleFacade(MooncakeConnector):
        def __init__(self):
            self.connector_worker = MagicMock()
            self.connector_scheduler = None

    facade = _D2RHStyleFacade.__new__(_D2RHStyleFacade)
    _D2RHStyleFacade.__init__(facade)

    assert facade._shutdown_complete is False
    facade.shutdown()
    facade.shutdown()

    assert facade.connector_worker.shutdown.call_count == 1
