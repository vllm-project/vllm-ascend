"""Lifecycle/shutdown tests for the layerwise mooncake connector.

Instances are built with ``__new__`` and only the fields touched by the
lifecycle code are wired, so no vLLM worker bootstrap, NPU device or ZMQ socket
is required.
"""

import threading
import time
import types
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    SHUTDOWN_JOIN_TIMEOUT_SECONDS,
    KVCacheRecvingLayerThread,
    KVCacheSendingLayerThread,
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise import metaserver
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise import worker as layerwise_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.metaserver import MetaserverClient


def _make_scheduler(events):
    scheduler = MooncakeLayerwiseConnectorScheduler.__new__(MooncakeLayerwiseConnectorScheduler)
    scheduler._shutdown_complete = False
    scheduler.engine_id = "engine-test"
    scheduler.executor = MagicMock()
    scheduler.executor.shutdown.side_effect = lambda **kwargs: events.append(
        "executor:" + str(kwargs["wait"]) + ":" + str(kwargs["cancel_futures"])
    )
    scheduler.metaserver_client = MagicMock()
    scheduler.metaserver_client.close.side_effect = lambda: events.append("client")
    return scheduler


def _fake_vllm_config():
    return types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(get_from_extra_config=lambda *args, **kwargs: {})
    )


def _make_scheduler_with_metaserver(metaserver_client):
    scheduler = MooncakeLayerwiseConnectorScheduler.__new__(MooncakeLayerwiseConnectorScheduler)
    scheduler._shutdown_complete = False
    scheduler.engine_id = "engine-test"
    scheduler.executor = ThreadPoolExecutor(max_workers=1)
    scheduler.metaserver_client = metaserver_client
    return scheduler


def test_scheduler_shutdown_order_and_idempotent():
    events: list[Any] = []
    scheduler = _make_scheduler(events)

    scheduler.shutdown()
    scheduler.shutdown()

    assert events == ["client", "executor:True:True"]
    assert scheduler.executor.shutdown.call_count == 1
    assert scheduler.metaserver_client.close.call_count == 1
    scheduler.executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


def test_scheduler_shutdown_tolerates_failures():
    events: list[Any] = []
    scheduler = _make_scheduler(events)
    scheduler.executor.shutdown.side_effect = RuntimeError("executor failed")
    scheduler.metaserver_client.close.side_effect = RuntimeError("client failed")

    scheduler.shutdown()  # best-effort: exceptions are logged, never raised
    scheduler.shutdown()

    assert scheduler.executor.shutdown.call_count == 1
    assert scheduler.metaserver_client.close.call_count == 1


class _BlockingHttpClient:
    """Fake httpx client whose ``post`` blocks until ``close`` wakes it."""

    def __init__(self):
        self.post_entered = threading.Event()
        self._closed_event = threading.Event()

    def post(self, url, json):
        self.post_entered.set()
        self._closed_event.wait(timeout=10.0)
        raise RuntimeError("request aborted by close")

    def close(self):
        self._closed_event.set()
        raise RuntimeError("close failed")


def test_scheduler_shutdown_bounded_when_metaserver_hangs():
    fake_http = _BlockingHttpClient()
    with patch.object(metaserver.httpx, "Client", return_value=fake_http):
        client = MetaserverClient(_fake_vllm_config())
    scheduler = _make_scheduler_with_metaserver(client)
    future = scheduler.executor.submit(scheduler._access_metaserver, "http://metaserver", {"request_id": "req-1"})
    assert fake_http.post_entered.wait(2.0), "metaserver post never started"

    start = time.monotonic()
    scheduler.shutdown()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0
    assert client._closed is True
    assert future.done()
    with pytest.raises(RuntimeError):
        scheduler.executor.submit(lambda: None)


class _BlockingMetaserverHandler(BaseHTTPRequestHandler):
    """HTTP handler that blocks until the test releases it."""

    received = threading.Event()
    release = threading.Event()

    def do_POST(self):
        type(self).received.set()
        type(self).release.wait(timeout=10.0)

    def log_message(self, format, *args):
        pass


def test_scheduler_shutdown_bounded_with_real_httpx(monkeypatch):
    """A hung metaserver must not block shutdown when the read timeout fires."""
    monkeypatch.setattr(metaserver, "METASERVER_CONNECT_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(metaserver, "METASERVER_READ_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(metaserver, "METASERVER_WRITE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(metaserver, "METASERVER_POOL_TIMEOUT_SECONDS", 0.1)
    _BlockingMetaserverHandler.received.clear()
    _BlockingMetaserverHandler.release.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BlockingMetaserverHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        client = MetaserverClient(_fake_vllm_config())
        scheduler = _make_scheduler_with_metaserver(client)
        url = f"http://127.0.0.1:{server.server_address[1]}/metaserver"
        future = scheduler.executor.submit(scheduler._access_metaserver, url, {"request_id": "req-1"})
        assert _BlockingMetaserverHandler.received.wait(2.0), "metaserver request never arrived"

        start = time.monotonic()
        scheduler.shutdown()
        elapsed = time.monotonic() - start
    finally:
        _BlockingMetaserverHandler.release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(2.0)

    assert elapsed < 5.0
    assert client._closed is True
    assert future.done()
    with pytest.raises(RuntimeError):
        scheduler.executor.submit(lambda: None)


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
    events: list[Any] = []
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


def test_worker_shutdown_closes_remote_sockets_and_contexts():
    worker = _make_worker(None, None)
    sock = MagicMock()
    ctx = MagicMock()
    remote_path = "tcp://remote-host:1234"
    worker.remote_sockets_lock = threading.Lock()
    worker.remote_sockets = {remote_path: deque([sock])}
    worker.remote_socket_contexts = {sock: ctx}
    worker.remote_poller = MagicMock()

    worker.shutdown()

    worker.remote_poller.unregister.assert_called_once_with(sock)
    sock.close.assert_called_once_with(linger=0)
    ctx.destroy.assert_called_once_with(linger=0)
    assert not worker.remote_sockets
    assert not worker.remote_socket_contexts

    worker.shutdown()  # idempotent: no second close
    sock.close.assert_called_once_with(linger=0)
    ctx.destroy.assert_called_once_with(linger=0)


def test_worker_shutdown_closes_tracked_socket_without_pool_entry():
    """Production shape: update_decoder_info never returns the socket to the pool."""
    worker = _make_worker(None, None)
    sock = MagicMock()
    ctx = MagicMock()
    worker.remote_sockets_lock = threading.Lock()
    worker.remote_sockets = {}
    worker.remote_socket_contexts = {sock: ctx}
    worker.remote_poller = MagicMock()

    worker.shutdown()

    worker.remote_poller.unregister.assert_called_once_with(sock)
    sock.close.assert_called_once_with(linger=0)
    ctx.destroy.assert_called_once_with(linger=0)
    assert not worker.remote_sockets
    assert not worker.remote_socket_contexts


def test_get_remote_socket_tracks_context_and_registers_poller(monkeypatch):
    worker = _make_worker(None, None)
    worker.remote_sockets_lock = threading.Lock()
    worker.remote_sockets = defaultdict(deque)
    worker.remote_socket_contexts = {}
    worker.remote_poller = MagicMock()
    worker.timeout = 1.0
    ctx = MagicMock()
    sock = MagicMock()
    fake_zmq = types.SimpleNamespace(
        Context=MagicMock(return_value=ctx),
        REQ=object(),
        SNDTIMEO=object(),
        RCVTIMEO=object(),
        POLLIN=object(),
    )
    monkeypatch.setattr(layerwise_worker, "zmq", fake_zmq)
    monkeypatch.setattr(layerwise_worker, "make_zmq_socket", lambda **kwargs: sock)

    returned = worker._get_remote_socket("host", 1234)

    assert returned is sock
    assert worker.remote_socket_contexts[sock] is ctx
    worker.remote_poller.register.assert_called_once_with(sock, fake_zmq.POLLIN)
    # Both send and recv must be bounded: the shared poller can wake on another
    # peer's socket, so an unbounded recv() would stall the worker thread.
    sock.setsockopt.assert_any_call(fake_zmq.SNDTIMEO, 1000)
    sock.setsockopt.assert_any_call(fake_zmq.RCVTIMEO, 1000)
    assert sock.setsockopt.call_count == 2

    worker._close_remote_sockets()

    worker.remote_poller.unregister.assert_called_once_with(sock)
    sock.close.assert_called_once_with(linger=0)
    ctx.destroy.assert_called_once_with(linger=0)
    assert not worker.remote_socket_contexts


class _TrackingPoller:
    """Poller stand-in that records which sockets still have a registration."""

    def __init__(self):
        self.registered: dict[Any, Any] = {}

    def register(self, sock, event):
        self.registered[sock] = event

    def unregister(self, sock):
        self.registered.pop(sock, None)


def test_update_decoder_info_discards_socket_when_query_fails(monkeypatch):
    """A failed metadata query must release its socket, context and poller entry."""
    worker = _make_worker(None, None)
    worker.remote_sockets_lock = threading.Lock()
    worker.remote_sockets = defaultdict(deque)
    worker.remote_socket_contexts = {}
    worker.remote_poller = _TrackingPoller()
    worker.remote_layer_metadata = defaultdict(dict)
    worker.remote_te_port = defaultdict(dict)
    worker.encoder = MagicMock()
    worker.timeout = 1.0
    ctx = MagicMock()
    sock = MagicMock()
    fake_zmq = types.SimpleNamespace(
        Context=MagicMock(return_value=ctx),
        REQ=object(),
        SNDTIMEO=object(),
        RCVTIMEO=object(),
        POLLIN=object(),
    )
    monkeypatch.setattr(layerwise_worker, "zmq", fake_zmq)
    monkeypatch.setattr(layerwise_worker, "make_zmq_socket", lambda **kwargs: sock)
    monkeypatch.setattr(layerwise_worker, "ensure_zmq_send", MagicMock(side_effect=RuntimeError("send failed")))
    req_meta = types.SimpleNamespace(remote_engine_id="engine-remote", remote_host="host", remote_port=1234)

    with pytest.raises(RuntimeError, match="send failed"):
        worker.update_decoder_info("req-1", req_meta)

    assert not worker.remote_socket_contexts
    assert not worker.remote_poller.registered
    sock.close.assert_called_once_with(linger=0)
    ctx.destroy.assert_called_once_with(linger=0)

    # Releasing again must stay safe: the tracked context was already popped.
    worker._discard_remote_socket(sock)
    assert sock.close.call_count == 2
    ctx.destroy.assert_called_once_with(linger=0)

    # Repeated failures must not accumulate poller entries or contexts.
    ctx2 = MagicMock()
    sock2 = MagicMock()
    fake_zmq.Context = MagicMock(return_value=ctx2)
    monkeypatch.setattr(layerwise_worker, "make_zmq_socket", lambda **kwargs: sock2)
    with pytest.raises(RuntimeError, match="send failed"):
        worker.update_decoder_info("req-2", req_meta)

    assert not worker.remote_socket_contexts
    assert not worker.remote_poller.registered
    sock2.close.assert_called_once_with(linger=0)
    ctx2.destroy.assert_called_once_with(linger=0)


def test_worker_shutdown_tolerates_failures():
    events: list[Any] = []
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
