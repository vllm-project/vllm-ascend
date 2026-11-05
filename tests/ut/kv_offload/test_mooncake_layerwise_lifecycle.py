"""Lifecycle/shutdown tests for the layerwise mooncake connector.

Instances are built with ``__new__`` and only the fields touched by the
lifecycle code are wired, so no vLLM worker bootstrap, NPU device or ZMQ socket
is required.
"""

import threading
from unittest.mock import MagicMock

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    SHUTDOWN_JOIN_TIMEOUT_SECONDS,
    KVCacheRecvingLayerThread,
    KVCacheSendingLayerThread,
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
)


def _make_scheduler(events):
    scheduler = MooncakeLayerwiseConnectorScheduler.__new__(MooncakeLayerwiseConnectorScheduler)
    scheduler._shutdown_complete = False
    scheduler.engine_id = "engine-test"
    scheduler.executor = MagicMock()
    scheduler.executor.shutdown.side_effect = lambda wait: events.append("executor:" + str(wait))
    scheduler.metaserver_client = MagicMock()
    scheduler.metaserver_client.close.side_effect = lambda: events.append("client")
    return scheduler


def test_scheduler_shutdown_order_and_idempotent():
    events = []
    scheduler = _make_scheduler(events)

    scheduler.shutdown()
    scheduler.shutdown()

    assert events == ["executor:True", "client"]
    assert scheduler.executor.shutdown.call_count == 1
    assert scheduler.metaserver_client.close.call_count == 1


def test_scheduler_shutdown_tolerates_failures():
    events = []
    scheduler = _make_scheduler(events)
    scheduler.executor.shutdown.side_effect = RuntimeError("executor failed")
    scheduler.metaserver_client.close.side_effect = RuntimeError("client failed")

    scheduler.shutdown()  # best-effort: exceptions are logged, never raised
    scheduler.shutdown()

    assert scheduler.executor.shutdown.call_count == 1
    assert scheduler.metaserver_client.close.call_count == 1


def _make_layer_thread_mock(name, events, stop_error=None, join_error=None):
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


def _make_worker(recv_thread, send_thread):
    worker = MooncakeLayerwiseConnectorWorker.__new__(MooncakeLayerwiseConnectorWorker)
    worker._shutdown_complete = False
    worker._shutdown_lock = threading.Lock()
    worker.engine_id = "engine-test"
    worker.kv_recv_layer_thread = recv_thread
    worker.kv_send_layer_thread = send_thread
    return worker


def test_worker_shutdown_stops_layer_threads():
    events = []
    recv_thread = _make_layer_thread_mock("recv", events)
    send_thread = _make_layer_thread_mock("send", events)
    worker = _make_worker(recv_thread, send_thread)

    worker.shutdown()
    worker.shutdown()

    assert events == ["stop:recv", "stop:send", "join:recv", "join:send"]
    assert recv_thread.stop.call_count == 1
    assert send_thread.stop.call_count == 1
    assert recv_thread.join.call_count == 1
    assert send_thread.join.call_count == 1
    assert recv_thread.join.call_args.kwargs["timeout"] == SHUTDOWN_JOIN_TIMEOUT_SECONDS
    assert send_thread.join.call_args.kwargs["timeout"] == SHUTDOWN_JOIN_TIMEOUT_SECONDS


def test_worker_shutdown_without_threads_is_safe():
    worker = _make_worker(None, None)
    worker.shutdown()
    worker.shutdown()


def test_worker_shutdown_tolerates_failures():
    events = []
    recv_thread = _make_layer_thread_mock("recv", events, join_error=RuntimeError("join failed"))
    send_thread = _make_layer_thread_mock("send", events, stop_error=RuntimeError("stop failed"))
    worker = _make_worker(recv_thread, send_thread)

    worker.shutdown()  # best-effort: exceptions are logged, never raised

    assert send_thread.join.call_count == 1
    assert recv_thread.join.call_count == 1
    assert worker._shutdown_complete is True


def test_connector_shutdown_delegates_by_role():
    scheduler_only = MooncakeLayerwiseConnector.__new__(MooncakeLayerwiseConnector)
    scheduler_only.connector_scheduler = MagicMock()
    scheduler_only.connector_worker = None
    scheduler_only.shutdown()
    scheduler_only.shutdown()
    scheduler_only.connector_scheduler.shutdown.assert_called_once_with()

    worker_only = MooncakeLayerwiseConnector.__new__(MooncakeLayerwiseConnector)
    worker_only.connector_scheduler = None
    worker_only.connector_worker = MagicMock()
    worker_only.shutdown()
    worker_only.shutdown()
    worker_only.connector_worker.shutdown.assert_called_once_with()

    no_side = MooncakeLayerwiseConnector.__new__(MooncakeLayerwiseConnector)
    no_side.connector_scheduler = None
    no_side.connector_worker = None
    no_side.shutdown()
    no_side.shutdown()  # idempotent, no resource to release


def test_connector_shutdown_uses_class_default_without_init():
    class _SubclassWithoutSuperInit(MooncakeLayerwiseConnector):
        def __init__(self):
            self.connector_scheduler = None
            self.connector_worker = MagicMock()

    connector = _SubclassWithoutSuperInit.__new__(_SubclassWithoutSuperInit)
    _SubclassWithoutSuperInit.__init__(connector)

    assert connector._shutdown_complete is False
    connector.shutdown()
    connector.shutdown()

    assert connector.connector_worker.shutdown.call_count == 1


def test_layer_threads_stop_is_idempotent():
    sending_thread = KVCacheSendingLayerThread.__new__(KVCacheSendingLayerThread)
    sending_thread._stop_event = threading.Event()
    sending_thread.stop()
    sending_thread.stop()
    assert sending_thread._stop_event.is_set()

    recving_thread = KVCacheRecvingLayerThread.__new__(KVCacheRecvingLayerThread)
    recving_thread._stop_event = threading.Event()
    recving_thread.stop()
    recving_thread.stop()
    assert recving_thread._stop_event.is_set()
