# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""D completion integration with real sockets; NPU registration is mocked."""

import time
from types import SimpleNamespace

import msgspec
import pytest
from zmq.constants import SocketOption, SocketType
from zmq.error import Again
from zmq.sugar.context import Context

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import push_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata


@pytest.fixture
def worker(monkeypatch):
    def initialize(self, *args):
        self.vllm_config = SimpleNamespace(speculative_config=None)
        self.engine_id = "d"
        self.pp_rank = self.pcp_rank = self.tp_rank = 0
        self.num_blocks = 100
        self.side_channel_host = "127.0.0.1"
        self.handshake_port = 0
        self.kv_transfer_config = SimpleNamespace(is_kv_producer=False)

    monkeypatch.setattr(MooncakeBaseConnectorWorker, "__init__", initialize)
    monkeypatch.setattr(MooncakeBaseConnectorWorker, "register_kv_caches", lambda *args: None)
    obj = push_worker.MooncakePushConnectorWorker(
        SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=1)), "d", None
    )
    obj.register_kv_caches({})
    yield obj
    obj.shutdown()
    assert not obj._decode_listener_thread.is_alive()


def metadata():
    result = MooncakeConnectorMetadata()
    result.add_new_push_req("d-request", ([1, 2],), 32, {"num_computed_tokens": 0})
    return result


def completion(**changes):
    return dict(
        kind="done",
        request_id="d-request",
        producer_engine_id="p",
        producer_rank=[0, 0, 0],
        producer_sizes=[1, 1, 1],
        consumer_engine_id="d",
        consumer_rank=[0, 0, 0],
        status="ok",
        **changes,
    )


def send(worker, message):
    with Context() as ctx, ctx.socket(SocketType.REQ) as sock:
        sock.setsockopt(SocketOption.LINGER, 0)
        sock.setsockopt(SocketOption.RCVTIMEO, 2000)
        sock.connect(worker._decode_listener_thread.address)
        sock.send(msgspec.msgpack.encode(message))
        reply = msgspec.msgpack.decode(sock.recv())
    # ACK precedes result publication; wait before inspecting main-thread state.
    deadline = time.monotonic() + 3
    while worker._result_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.001)
    return reply


def finish(worker):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = worker.get_transfer_results()
        if result.finished_recving:
            return result
        time.sleep(0.005)
    pytest.fail("No terminal receive result")


@pytest.mark.parametrize("early", [False, True])
def test_completion_before_or_after_allocation_and_duplicates(worker, early):
    meta = metadata()
    if not early:
        worker.start_load_kv(meta)
    assert send(worker, completion()) == {"kind": "done_ack", "request_id": "d-request"}
    if early:
        assert not worker.get_transfer_results().finished_recving
        worker.start_load_kv(meta)
    worker.start_load_kv(meta)
    assert finish(worker).finished_recving == {"d-request"}
    send(worker, completion())
    worker.start_load_kv(meta)
    assert not worker.get_transfer_results().finished_recving

    failed = MooncakeConnectorMetadata()
    failed.push_failed_requests = {"d-request": False, "no-external-kv": True}
    worker.start_load_kv(failed)
    result = worker.get_transfer_results()
    assert not result.finished_recving
    assert not result.failed_recving
    assert not worker._reqs_recv_state


def test_all_sources_required_and_error_is_reported(worker):
    worker.start_load_kv(metadata())
    first = dict(completion(), producer_sizes=[1, 1, 2], status="error")
    send(worker, first)
    send(worker, first)
    assert not worker.get_transfer_results().finished_recving
    send(worker, dict(first, producer_rank=[0, 0, 1], status="ok"))
    result = finish(worker)
    assert result.finished_recving == result.failed_recving == {"d-request"}


@pytest.mark.parametrize("failure", ["write", "timeout", "dispatch_timeout"])
def test_receive_failure_finishes_request_and_keeps_service_running(worker, failure):
    worker.start_load_kv(metadata())
    if failure == "write":
        send(worker, dict(completion(), status="error", error="P WRITE failed"))
    else:
        if failure == "dispatch_timeout":
            meta = MooncakeConnectorMetadata()
            meta.push_failed_requests = {"d-request": False}
            worker.start_load_kv(meta)
        worker._reqs_recv_state["d-request"].start_time -= push_worker.envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 1
    result = worker.get_transfer_results()
    assert result.finished_recving == result.failed_recving == {"d-request"}
    assert not worker._reqs_recv_state
    assert not worker.get_transfer_results().failed_recving
    assert not worker._stop_event.is_set()

    next_meta = MooncakeConnectorMetadata()
    next_meta.add_new_push_req("next", ([3, 4],), 32, {"num_computed_tokens": 0})
    worker.start_load_kv(next_meta)
    send(worker, dict(completion(), request_id="next"))
    result = finish(worker)
    assert result.finished_recving == {"next"}
    assert not result.failed_recving

    # A late DONE must not reopen a request already retired by timeout.
    send(worker, completion())
    result = worker.get_transfer_results()
    assert not result.finished_recving
    assert not result.failed_recving
    assert "d-request" not in worker._reqs_recv_state


@pytest.mark.parametrize("early", [False, True])
def test_failed_done_is_not_lost_when_same_source_reports_success(worker, early):
    if not early:
        worker.start_load_kv(metadata())
    send(worker, dict(completion(), status="error"))
    send(worker, completion())
    if early:
        assert not worker.get_transfer_results().finished_recving
        worker.start_load_kv(metadata())
    result = finish(worker)
    assert result.finished_recving == result.failed_recving == {"d-request"}


@pytest.mark.parametrize(
    "frames",
    [
        [b"\xc1"],
        [b"\x91" * 10000 + b"\xc0"],
        [msgspec.msgpack.encode({"kind": "done"})],
        [b"extra", b"payload"],
    ],
    ids=["malformed-msgpack", "decoder-recursion", "missing-fields", "multipart-payload"],
)
def test_invalid_done_does_not_stop_listener_or_finish_request(worker, frames):
    worker.start_load_kv(metadata())
    with Context() as ctx, ctx.socket(SocketType.REQ) as sock:
        sock.setsockopt(SocketOption.LINGER, 0)
        sock.setsockopt(SocketOption.RCVTIMEO, 2000)
        sock.connect(worker._decode_listener_thread.address)
        sock.send_multipart(frames)
        reply = msgspec.msgpack.decode(sock.recv())
        assert reply["kind"] == "error"
        assert worker._decode_listener_thread.is_alive()
        assert worker._decode_listener_thread.error is None
        result = worker.get_transfer_results()
        assert not result.finished_recving
        assert not result.failed_recving
        # Use the same connection to verify the listener continues serving.
        sock.send(msgspec.msgpack.encode(completion()))
        assert msgspec.msgpack.decode(sock.recv()) == {"kind": "done_ack", "request_id": "d-request"}
    result = finish(worker)
    assert result.finished_recving == {"d-request"}
    assert not result.failed_recving


def test_done_ack_send_error_reports_request_failure(worker):
    worker.start_load_kv(metadata())

    def fail_send(frames):
        raise Again()

    worker._decode_listener_thread._handle_done_message(
        SimpleNamespace(send_multipart=fail_send), [b"producer", b"", msgspec.msgpack.encode(completion())]
    )
    result = finish(worker)
    assert result.finished_recving == result.failed_recving == {"d-request"}


def test_push_thread_start_failure_is_reported_and_stopped(worker, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Cannot initialize control transport")

    monkeypatch.setattr(push_worker, "zmq_ctx", fail)
    thread = push_worker.MooncakeDecodeListenerThread(worker)
    with pytest.raises(RuntimeError, match="failed to start"):
        thread.start()
    assert isinstance(thread.error, OSError)
    assert not thread.is_alive()
    thread.shutdown()


@pytest.mark.parametrize("same_step", [False, True])
def test_unstarted_metaserver_failure_finishes_receive_without_done(worker, same_step):
    meta = metadata()
    if not same_step:
        worker.start_load_kv(meta)
        meta = MooncakeConnectorMetadata()
    meta.push_failed_requests = {"d-request": True}
    worker.start_load_kv(meta)
    result = worker.get_transfer_results()
    assert result.finished_recving == result.failed_recving == {"d-request"}
    assert not worker._reqs_recv_state
    worker.start_load_kv(meta)
    assert not worker.get_transfer_results().finished_recving


@pytest.mark.parametrize("done_queued", [False, True])
def test_uncertain_metaserver_failure_waits_for_all_writes(worker, done_queued):
    worker.start_load_kv(metadata())
    first = dict(completion(), producer_sizes=[1, 1, 2])
    if done_queued:
        send(worker, first)
    failed = MooncakeConnectorMetadata()
    failed.push_failed_requests = {"d-request": False}
    worker.start_load_kv(failed)
    assert not worker.get_transfer_results().finished_recving
    assert "d-request" in worker._reqs_recv_state
    if not done_queued:
        send(worker, first)
    send(worker, dict(first, producer_rank=[0, 0, 1]))
    result = finish(worker)
    assert result.finished_recving == result.failed_recving == {"d-request"}
