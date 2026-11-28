"""Startup rollback + thread fatal-error tests for the mooncake connectors."""

import contextlib
import threading
import types
from unittest.mock import MagicMock, call

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_connector as pull_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_layerwise_connector as layerwise_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.base import worker as base_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise import threads as layerwise_threads
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise import worker as layerwise_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull import threads as pull_threads
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull import worker as pull_worker
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE


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
    monkeypatch.setattr(base_worker, "time", _FakeTime())

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
    monkeypatch.setattr(base_worker, "time", _FakeTime())

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
    monkeypatch.setattr(pull_threads, "pull_handshake_port", lambda **kwargs: 12345)
    monkeypatch.setattr(pull_threads, "make_zmq_path", lambda *args: "tcp://127.0.0.1:12345")

    def _broken_ctx(*args, **kwargs):
        raise RuntimeError("zmq setup failed")

    monkeypatch.setattr(pull_threads, "zmq_ctx", _broken_ctx)

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
    monkeypatch.setattr(layerwise_threads, "get_world_group", lambda: types.SimpleNamespace(local_rank=0))
    monkeypatch.setattr(
        layerwise_threads,
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


def _is_registered(fake_te: GlobalTE) -> bool:
    """Whether every process-wide buffer currently has a native registration."""
    return bool(fake_te.registered_buffers) and not fake_te._unregistered_ptrs


def _make_registered_global_te(ptrs=(0x1000, 0x2000)):
    """Real GlobalTE with a fake native engine, registered by this test only."""
    engine = MagicMock()
    engine.register_memory.return_value = 0
    engine.unregister_memory.return_value = 0
    fake_te = GlobalTE()
    fake_te.transfer_engine = engine
    fake_te.register_buffer(list(ptrs), [16] * len(ptrs))
    assert _is_registered(fake_te) is True
    return fake_te, engine


def _exited_thread_and_event():
    thread = MagicMock()
    thread.is_alive.return_value = False
    ready_event = MagicMock()
    ready_event.is_set.return_value = False
    return thread, ready_event


def test_pull_startup_failure_rolls_back_this_registration(monkeypatch):
    fake_te, engine = _make_registered_global_te()
    unregister_spy = MagicMock(wraps=fake_te.unregister_buffer)
    monkeypatch.setattr(fake_te, "unregister_buffer", unregister_spy)
    monkeypatch.setattr(pull_worker, "global_te", fake_te)
    worker = _make_pull_worker()
    worker.engine_id = "engine-test"
    worker._registered_kv_buffer = True
    thread, ready_event = _exited_thread_and_event()

    with pytest.raises(RuntimeError):
        worker._wait_until_ready(thread, ready_event)

    assert worker.shutdown.call_count == 1
    assert worker._registered_kv_buffer is False
    assert _is_registered(fake_te) is False
    assert unregister_spy.call_count == 1
    assert engine.unregister_memory.call_args_list == [call(0x1000), call(0x2000)]

    worker._rollback_registered_kv_buffer()  # idempotent: no second unregister
    assert engine.unregister_memory.call_count == 2


def test_pull_rollback_skips_registration_owned_elsewhere(monkeypatch):
    fake_te, engine = _make_registered_global_te()
    monkeypatch.setattr(pull_worker, "global_te", fake_te)
    worker = _make_pull_worker()
    worker.engine_id = "engine-test"
    worker._registered_kv_buffer = False

    worker._rollback_registered_kv_buffer()
    worker._rollback_registered_kv_buffer()

    engine.unregister_memory.assert_not_called()
    assert _is_registered(fake_te) is True


def test_layerwise_startup_failure_rolls_back_this_registration(monkeypatch):
    fake_te, te_engine = _make_registered_global_te((0x5000,))
    unregister_spy = MagicMock(wraps=fake_te.unregister_buffer)
    monkeypatch.setattr(fake_te, "unregister_buffer", unregister_spy)
    monkeypatch.setattr(layerwise_worker, "global_te", fake_te)
    engine = MagicMock()
    engine.unregister_memory.return_value = 0
    worker = _make_layerwise_worker()
    worker.engine_id = "engine-test"
    worker.engine = engine
    worker._registered_engine_buffers = [0x3000, 0x4000]
    worker._registered_kv_buffer = True
    thread, ready_event = _exited_thread_and_event()

    with pytest.raises(RuntimeError):
        worker._wait_until_ready(thread, ready_event)

    assert worker.shutdown.call_count == 1
    assert engine.unregister_memory.call_args_list == [call(0x4000), call(0x3000)]
    assert te_engine.unregister_memory.call_args_list == [call(0x5000)]
    assert worker._registered_engine_buffers == []
    assert worker._registered_kv_buffer is False
    assert _is_registered(fake_te) is False
    assert unregister_spy.call_count == 1

    worker._rollback_registered_memory()  # idempotent: no second unregister
    assert engine.unregister_memory.call_count == 2
    assert te_engine.unregister_memory.call_count == 1


def test_layerwise_rollback_tolerates_unregister_failures(monkeypatch):
    fake_te, te_engine = _make_registered_global_te((0x5000,))
    monkeypatch.setattr(layerwise_worker, "global_te", fake_te)
    engine = MagicMock()
    engine.unregister_memory.side_effect = [RuntimeError("engine unregister failed"), 1]
    worker = _make_layerwise_worker()
    worker.engine_id = "engine-test"
    worker.engine = engine
    worker._registered_engine_buffers = [0x3000, 0x4000]
    worker._registered_kv_buffer = True

    worker._rollback_registered_memory()  # best effort: logs, never raises

    assert engine.unregister_memory.call_args_list == [call(0x4000), call(0x3000)]
    assert te_engine.unregister_memory.call_args_list == [call(0x5000)]
    assert worker._registered_engine_buffers == []
    assert worker._registered_kv_buffer is False
    assert _is_registered(fake_te) is False


class _OrderingThread:
    """Thread stand-in that records stop/join order for the rollback tests."""

    def __init__(self, events, name):
        self.events = events
        self.name = name
        self.fatal_error = None
        # Already exited: _wait_until_ready takes the failure branch at once.
        self._alive = False
        self.executor = MagicMock()
        self.remote_socket_pool = MagicMock()

    def is_alive(self):
        return self._alive

    def stop(self):
        self.events.append("stop:" + self.name)
        self._alive = False

    def join(self, timeout=None):
        self.events.append("join:" + self.name)
        self._alive = False


def test_pull_rollback_runs_after_threads_stop(monkeypatch):
    events: list[str] = []
    fake_te = MagicMock()
    fake_te.unregister_buffer.side_effect = lambda: events.append("unregister")
    monkeypatch.setattr(pull_worker, "global_te", fake_te)
    worker = pull_connector.MooncakeConnectorWorker.__new__(pull_connector.MooncakeConnectorWorker)
    worker.engine_id = "engine-test"
    worker._shutdown_complete = False
    worker._shutdown_lock = threading.Lock()
    worker.kv_recv_thread = _OrderingThread(events, "recv")
    worker.kv_send_thread = None
    worker._registered_kv_buffer = True
    ready_event = MagicMock()
    ready_event.is_set.return_value = False

    with pytest.raises(RuntimeError):
        worker._wait_until_ready(worker.kv_recv_thread, ready_event)

    # unregister_memory disconnects peers, so the thread must be joined first.
    assert events == ["stop:recv", "join:recv", "unregister"]


def test_layerwise_rollback_runs_after_threads_stop(monkeypatch):
    events: list[str] = []
    fake_te = MagicMock()
    fake_te.unregister_buffer.side_effect = lambda: events.append("unregister")
    monkeypatch.setattr(layerwise_worker, "global_te", fake_te)
    worker = layerwise_connector.MooncakeLayerwiseConnectorWorker.__new__(
        layerwise_connector.MooncakeLayerwiseConnectorWorker
    )
    worker.engine_id = "engine-test"
    worker._shutdown_complete = False
    worker._shutdown_lock = threading.Lock()
    worker.kv_send_layer_thread = _OrderingThread(events, "send")
    worker.kv_recv_layer_thread = None
    worker._registered_kv_buffer = True
    worker._registered_engine_buffers = []
    ready_event = MagicMock()
    ready_event.is_set.return_value = False

    with pytest.raises(RuntimeError):
        worker._wait_until_ready(worker.kv_send_layer_thread, ready_event)

    assert events == ["stop:send", "join:send", "unregister"]


class _FakeCacheTensor:
    """Minimal KV-cache tensor surface used by register_kv_caches/create_kv_buffer."""

    def __init__(self, ptr, shape=(1, 8), element_size=2):
        self.shape = shape
        self._ptr = ptr
        self._element_size = element_size

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return 8

    def element_size(self):
        return self._element_size

    def is_contiguous(self):
        return True


_REGIONS = types.SimpleNamespace(ptrs=[0x1000, 0x2000], lengths=[16, 32])

_CLAIM_CASES = [
    (base_worker, _make_pull_worker),
    (base_worker, _make_layerwise_worker),
]


@pytest.mark.parametrize(("module", "worker_factory"), _CLAIM_CASES)
def test_claim_kv_buffer_registration_claims_only_its_own_registration(monkeypatch, module, worker_factory):
    fake_global_te = MagicMock()
    fake_global_te.register_buffer.return_value = True
    monkeypatch.setattr(module, "global_te", fake_global_te)
    worker = worker_factory()
    worker._registered_kv_buffer = False

    worker._claim_kv_buffer_registration(_REGIONS)

    assert worker._registered_kv_buffer is True
    fake_global_te.register_buffer.assert_called_once_with([0x1000, 0x2000], [16, 32])


@pytest.mark.parametrize(("module", "worker_factory"), _CLAIM_CASES)
def test_claim_kv_buffer_registration_ignores_foreign_registration(monkeypatch, module, worker_factory):
    fake_global_te = MagicMock()
    fake_global_te.register_buffer.return_value = False
    monkeypatch.setattr(module, "global_te", fake_global_te)
    worker = worker_factory()
    worker._registered_kv_buffer = False

    worker._claim_kv_buffer_registration(_REGIONS)

    assert worker._registered_kv_buffer is False


@pytest.mark.parametrize(("module", "worker_factory"), _CLAIM_CASES)
def test_claim_kv_buffer_registration_keeps_ownership_on_repeat(monkeypatch, module, worker_factory):
    # A later register_kv_caches call early-returns False; it must not clear the
    # claim made by the call that actually registered the buffers.
    fake_global_te = MagicMock()
    fake_global_te.register_buffer.side_effect = [True, False]
    monkeypatch.setattr(module, "global_te", fake_global_te)
    worker = worker_factory()
    worker._registered_kv_buffer = False

    worker._claim_kv_buffer_registration(_REGIONS)
    worker._claim_kv_buffer_registration(_REGIONS)

    assert worker._registered_kv_buffer is True
    assert fake_global_te.register_buffer.call_count == 2


def test_layerwise_create_kv_buffer_failure_records_only_successful_registrations(monkeypatch):
    engine = MagicMock()
    engine.register_memory.side_effect = [0, 1]
    fake_global_te = MagicMock()
    monkeypatch.setattr(layerwise_worker, "global_te", fake_global_te)
    worker = _make_layerwise_worker()
    worker.engine = engine
    worker.pd_head_ratio = 1
    worker.enable_kv_quant = False
    worker.enable_c8_quant = False
    worker._registered_engine_buffers = []
    worker._registered_kv_buffer = True
    worker.k_buffer = _FakeCacheTensor(0x200000)
    worker.v_buffer = _FakeCacheTensor(0x400000)

    with pytest.raises(RuntimeError, match="memory registration failed"):
        worker.create_kv_buffer((object(), object()))

    assert engine.register_memory.call_args_list == [call(0x200000, 16), call(0x400000, 16)]
    assert worker._registered_engine_buffers == [0x200000]

    worker._rollback_registered_memory()

    assert engine.unregister_memory.call_args_list == [call(0x200000)]
    assert fake_global_te.unregister_buffer.call_count == 1
    assert worker._registered_engine_buffers == []
    assert worker._registered_kv_buffer is False


def test_layerwise_post_registration_failure_rolls_back_and_shuts_down(monkeypatch):
    """The layerwise startup guard covers create_kv_buffer and the steps after it."""
    events: list[str] = []
    fake_global_te = MagicMock()
    fake_global_te.register_buffer.return_value = True
    fake_global_te.unregister_buffer.side_effect = lambda: events.append("unregister")
    monkeypatch.setattr(layerwise_worker, "global_te", fake_global_te)
    # The shared startup claim lives in base/worker.py and reads that module's
    # global_te; the rollback still uses the layerwise worker module.
    monkeypatch.setattr(base_worker, "global_te", fake_global_te)
    monkeypatch.setattr(
        layerwise_worker,
        "collect_storage_merged_register_regions",
        lambda kv_caches: types.SimpleNamespace(ptrs=[0x200000], lengths=[16]),
    )
    monkeypatch.setattr(layerwise_worker, "validate_register_region_count", lambda regions: None)
    worker = _make_layerwise_worker()
    worker.shutdown = MagicMock(side_effect=lambda: events.append("shutdown"))
    worker._registered_kv_buffer = False
    worker._registered_engine_buffers = []
    worker.kv_cache_config = types.SimpleNamespace(
        kv_cache_groups=[types.SimpleNamespace(layer_names=["layer0"], kv_cache_spec=object())],
        kv_cache_tensors=[],
        num_blocks=1,
    )
    worker.pd_head_ratio = 1
    worker.enable_kv_quant = True
    worker.enable_c8_quant = False
    worker.use_attn_mamba_hybrid = False
    worker.attn_resharding_group_idx = set()
    worker.kernel_block_size_scale = [1]
    worker.layer_metadata = {}
    worker.sparse_sfa_c8_layers = set()
    worker.create_kv_buffer = MagicMock(side_effect=RuntimeError("engine register failed"))

    with pytest.raises(RuntimeError, match="engine register failed"):
        worker.register_kv_caches({"layer0": _FakeCacheTensor(0x200000)})

    fake_global_te.register_buffer.assert_called_once_with([0x200000], [16])
    fake_global_te.unregister_buffer.assert_called_once_with()
    worker.shutdown.assert_called_once_with()
    assert worker._registered_kv_buffer is False
    # The startup guard stops the threads before it unregisters native memory.
    assert events == ["shutdown", "unregister"]


def _make_recv_layer_thread():
    thread = layerwise_connector.KVCacheRecvingLayerThread.__new__(layerwise_connector.KVCacheRecvingLayerThread)
    thread.tp_rank = 0
    thread.side_channel_port = 30000
    thread.side_channel_host = "127.0.0.1"
    thread.ready_event = threading.Event()
    thread._stop_event = threading.Event()
    thread.fatal_error = None
    thread.lock = threading.Lock()
    thread.task_tracker = {}
    thread.done_requests = set()
    thread.failed_requests = set()
    thread.metadata = layerwise_connector.MooncakeAgentMetadata(te_rpc_port=0, layer_metadata={})
    return thread


def test_layerwise_recv_thread_setup_failure_records_fatal_error(monkeypatch):
    def _broken_ctx(*args, **kwargs):
        raise RuntimeError("zmq setup failed")

    monkeypatch.setattr(layerwise_threads, "zmq_ctx", _broken_ctx)
    thread = _make_recv_layer_thread()

    thread.run()

    assert isinstance(thread.fatal_error, RuntimeError)
    assert str(thread.fatal_error) == "zmq setup failed"
    assert not thread.ready_event.is_set()

    worker = layerwise_connector.MooncakeLayerwiseConnectorWorker.__new__(
        layerwise_connector.MooncakeLayerwiseConnectorWorker
    )
    worker.kv_recv_layer_thread = thread
    worker.kv_send_layer_thread = None
    assert worker.get_thread_fatal_error() is thread.fatal_error


def test_layerwise_recv_thread_transport_failure_is_fatal(monkeypatch):
    sock = MagicMock()
    sock.poll.side_effect = RuntimeError("poll broke")

    @contextlib.contextmanager
    def _fake_ctx(socket_type, addr):
        yield sock

    monkeypatch.setattr(layerwise_threads, "zmq_ctx", _fake_ctx)
    thread = _make_recv_layer_thread()

    thread.run()

    assert isinstance(thread.fatal_error, RuntimeError)
    assert str(thread.fatal_error) == "poll broke"
    assert thread.ready_event.is_set()


def test_layerwise_recv_thread_send_failure_is_fatal(monkeypatch):
    encoded_done = layerwise_connector.msgspec.msgpack.Encoder().encode(
        (layerwise_connector.DONE_SENDING_MSG, "req-send", 1, "path1")
    )

    class _SendFailureSock:
        def poll(self, timeout):
            return 1

        def recv_multipart(self):
            return [b"identity", b"", encoded_done]

        def send_multipart(self, response):
            raise RuntimeError("send broke")

    sock = _SendFailureSock()

    @contextlib.contextmanager
    def _fake_ctx(socket_type, addr):
        yield sock

    monkeypatch.setattr(layerwise_threads, "zmq_ctx", _fake_ctx)
    thread = _make_recv_layer_thread()

    thread.run()

    assert isinstance(thread.fatal_error, RuntimeError)
    assert str(thread.fatal_error) == "send broke"
    assert thread.done_requests == {"req-send"}


def test_layerwise_recv_thread_decode_error_is_not_fatal(monkeypatch):
    class _DecodeErrorSock:
        def __init__(self):
            self.poll_calls = 0

        def poll(self, timeout):
            self.poll_calls += 1
            if self.poll_calls == 1:
                return 1
            thread._stop_event.set()
            return 0

        def recv_multipart(self):
            return [b"identity", b"", b"\xc1"]

    sock = _DecodeErrorSock()

    @contextlib.contextmanager
    def _fake_ctx(socket_type, addr):
        yield sock

    monkeypatch.setattr(layerwise_threads, "zmq_ctx", _fake_ctx)
    thread = _make_recv_layer_thread()

    thread.run()

    assert thread._stop_event.is_set()
    assert sock.poll_calls == 2
    assert thread.fatal_error is None
