"""Startup rollback + thread fatal-error tests for the mooncake connectors."""

import threading
import types
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_connector as pull_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_layerwise_connector as layerwise_connector,
)


class _FakeTime:
    """First call returns 0, every later call jumps past the startup timeout."""

    def __init__(self):
        self.calls = 0

    def time(self):
        self.calls += 1
        return 0.0 if self.calls == 1 else 10_000.0

    def sleep(self, seconds):
        return None


def _make_pull_worker():
    worker = pull_connector.MooncakeConnectorWorker.__new__(pull_connector.MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    return worker


def _make_layerwise_worker():
    worker = layerwise_connector.MooncakeLayerwiseConnectorWorker.__new__(
        layerwise_connector.MooncakeLayerwiseConnectorWorker
    )
    worker.shutdown = MagicMock()
    return worker


def test_startup_timeout_constants_are_five_minutes():
    assert pull_connector.KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS == 5 * 60
    assert layerwise_connector.KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS == 5 * 60


def test_pull_wait_until_ready_ready_path_does_not_rollback():
    worker = _make_pull_worker()
    thread = MagicMock()
    thread.is_alive.return_value = True
    ready_event = MagicMock()
    ready_event.is_set.return_value = True

    worker._wait_until_ready(thread, ready_event)

    worker.shutdown.assert_not_called()


def test_pull_wait_until_ready_thread_exit_rolls_back():
    worker = _make_pull_worker()
    thread = MagicMock()
    thread.is_alive.return_value = False
    ready_event = MagicMock()
    ready_event.is_set.return_value = False

    with pytest.raises(RuntimeError) as excinfo:
        worker._wait_until_ready(thread, ready_event)

    assert str(excinfo.value) == "KV Cache sending/receiving thread failed to start."
    worker.shutdown.assert_called_once_with()


def test_pull_wait_until_ready_timeout_rolls_back(monkeypatch):
    worker = _make_pull_worker()
    thread = MagicMock()
    thread.is_alive.return_value = True
    ready_event = MagicMock()
    ready_event.is_set.return_value = False
    monkeypatch.setattr(pull_connector, "time", _FakeTime())

    with pytest.raises(RuntimeError) as excinfo:
        worker._wait_until_ready(thread, ready_event)

    assert str(excinfo.value) == "Timeout waiting for KV Cache thread to be ready."
    worker.shutdown.assert_called_once_with()


def test_layerwise_wait_until_ready_ready_path_does_not_rollback():
    worker = _make_layerwise_worker()
    thread = MagicMock()
    thread.is_alive.return_value = True
    ready_event = MagicMock()
    ready_event.is_set.return_value = True

    worker._wait_until_ready(thread, ready_event)

    worker.shutdown.assert_not_called()


def test_layerwise_wait_until_ready_thread_exit_rolls_back():
    worker = _make_layerwise_worker()
    thread = MagicMock()
    thread.is_alive.return_value = False
    ready_event = MagicMock()
    ready_event.is_set.return_value = False

    with pytest.raises(RuntimeError) as excinfo:
        worker._wait_until_ready(thread, ready_event)

    assert str(excinfo.value) == "KV Cache layer thread failed to start."
    worker.shutdown.assert_called_once_with()


def test_layerwise_wait_until_ready_timeout_rolls_back(monkeypatch):
    worker = _make_layerwise_worker()
    thread = MagicMock()
    thread.is_alive.return_value = True
    ready_event = MagicMock()
    ready_event.is_set.return_value = False
    monkeypatch.setattr(layerwise_connector, "time", _FakeTime())

    with pytest.raises(RuntimeError) as excinfo:
        worker._wait_until_ready(thread, ready_event)

    assert str(excinfo.value) == "Timeout waiting for KV Cache layer thread to be ready."
    worker.shutdown.assert_called_once_with()


def test_pull_get_thread_fatal_error():
    worker = pull_connector.MooncakeConnectorWorker.__new__(pull_connector.MooncakeConnectorWorker)
    worker.kv_recv_thread = None
    worker.kv_send_thread = None
    assert worker.get_thread_fatal_error() is None

    worker.kv_recv_thread = MagicMock(fatal_error=None)
    worker.kv_send_thread = MagicMock(fatal_error=None)
    assert worker.get_thread_fatal_error() is None

    send_error = RuntimeError("send fatal")
    worker.kv_send_thread = MagicMock(fatal_error=send_error)
    assert worker.get_thread_fatal_error() is send_error

    recv_error = RuntimeError("recv fatal")
    worker.kv_recv_thread = MagicMock(fatal_error=recv_error)
    assert worker.get_thread_fatal_error() is recv_error


def test_layerwise_get_thread_fatal_error():
    worker = layerwise_connector.MooncakeLayerwiseConnectorWorker.__new__(
        layerwise_connector.MooncakeLayerwiseConnectorWorker
    )
    worker.kv_recv_layer_thread = None
    worker.kv_send_layer_thread = types.SimpleNamespace(fatal_error=None)
    assert worker.get_thread_fatal_error() is None

    fatal = RuntimeError("layer fatal")
    worker.kv_recv_layer_thread = types.SimpleNamespace(fatal_error=fatal)
    assert worker.get_thread_fatal_error() is fatal


def test_pull_sending_thread_run_records_fatal_error(monkeypatch):
    monkeypatch.setattr(pull_connector, "pull_handshake_port", lambda **kwargs: 12345)
    monkeypatch.setattr(pull_connector, "make_zmq_path", lambda *args: "tcp://127.0.0.1:12345")

    def _broken_ctx(*args, **kwargs):
        raise RuntimeError("zmq setup failed")

    monkeypatch.setattr(pull_connector, "zmq_ctx", _broken_ctx)

    thread = pull_connector.KVCacheSendingThread.__new__(pull_connector.KVCacheSendingThread)
    thread.tp_rank = 0
    thread.pp_rank = 0
    thread.pcp_rank = 0
    thread.pcp_size = 1
    thread.side_channel_port = 30000
    thread.side_channel_host = "127.0.0.1"
    thread.tp_size = 1
    thread.ready_event = threading.Event()
    thread.fatal_error = None

    thread.run()  # baseline behavior: the exception is logged, not propagated

    assert isinstance(thread.fatal_error, RuntimeError)
    assert str(thread.fatal_error) == "zmq setup failed"


def test_layerwise_sending_thread_run_records_fatal_error(monkeypatch):
    monkeypatch.setattr(layerwise_connector, "get_world_group", lambda: types.SimpleNamespace(local_rank=0))
    monkeypatch.setattr(
        layerwise_connector,
        "torch",
        types.SimpleNamespace(
            device=lambda name: name,
            npu=types.SimpleNamespace(set_device=lambda device: None),
        ),
    )

    thread = layerwise_connector.KVCacheSendingLayerThread.__new__(layerwise_connector.KVCacheSendingLayerThread)
    thread._stop_event = threading.Event()
    thread.ready_event = threading.Event()
    thread.send_queue = MagicMock()
    thread.send_queue.get.side_effect = RuntimeError("queue broke")
    thread._handle_request = MagicMock()
    thread.fatal_error = None

    with pytest.raises(RuntimeError) as excinfo:
        thread.run()

    assert str(excinfo.value) == "queue broke"
    assert thread.fatal_error is excinfo.value
    assert thread.ready_event.is_set()
    thread._handle_request.assert_not_called()
