# SPDX-License-Identifier: Apache-2.0
"""D control integration with real sockets; NPU registration alone is mocked."""

import queue
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import msgspec
import pytest
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import push_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import GET_META_MSG
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePushConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_scheduler import MooncakePushConnectorScheduler


@pytest.fixture
def worker(monkeypatch):
    def initialize(self, *args):
        self.engine_id = "d"
        self.pp_rank = self.pcp_rank = self.tp_rank = 0
        self.num_blocks = 100
        self.side_channel_host = "127.0.0.1"
        self.handshake_port = 0
        self.kv_transfer_config = SimpleNamespace(is_kv_producer=False)

    monkeypatch.setattr(MooncakeBaseConnectorWorker, "__init__", initialize)
    monkeypatch.setattr(MooncakeBaseConnectorWorker, "register_kv_caches", lambda *args: None)
    obj = push_worker.MooncakePushConnectorWorker(None, "d", None)
    obj.register_kv_caches({})
    yield obj
    obj.shutdown()
    assert not obj._sending_thread.is_alive()
    assert not obj._recving_thread.is_alive()


@pytest.fixture
def producer():
    messages = queue.Queue()
    stop = threading.Event()
    started = threading.Event()
    address = []

    def serve():
        with zmq.Context() as ctx, ctx.socket(zmq.REP) as sock:
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind("tcp://127.0.0.1:*")
            address.append(sock.getsockopt(zmq.LAST_ENDPOINT).decode())
            started.set()
            while not stop.is_set():
                if sock.poll(50):
                    message = msgspec.msgpack.decode(sock.recv())
                    messages.put(message)
                    sock.send(msgspec.msgpack.encode({**message, "kind": "ready_ack"}))

    thread = threading.Thread(target=serve)
    thread.start()
    assert started.wait(3)
    yield address[0], messages
    stop.set()
    thread.join(3)
    assert not thread.is_alive()


def metadata(transfer_id="t"):
    result = MooncakeConnectorMetadata()
    result.add_new_push_req(
        "r" + transfer_id,
        ([1, 2],),
        32,
        dict(
            transfer_id=transfer_id,
            remote_engine_id="p",
            remote_bootstrap_addr="http://bootstrap",
            num_computed_tokens=0,
        ),
        scheduler_host="127.0.0.1",
        scheduler_port=9000,
    )
    return result


def done(worker, ready, **changes):
    message = {**ready, "kind": "done", "status": "ok", "writes_drained": True, **changes}
    with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.connect(worker._recving_thread.address)
        sock.send(msgspec.msgpack.encode(message))
        return msgspec.msgpack.decode(sock.recv())


def finish(worker):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = worker.get_transfer_results()
        if result.finished_recving:
            return result
        time.sleep(0.01)
    pytest.fail("No terminal receive result")


def test_scheduler_to_ready_to_get_finished(worker, producer, monkeypatch, request):
    address, messages = producer
    monkeypatch.setattr(worker._sending_thread, "_get_remote_worker_addresses", lambda *args: {(0, 0, 0): address})
    scheduler = MooncakePushConnectorScheduler.__new__(MooncakePushConnectorScheduler)
    scheduler.kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    scheduler._metadata_thread = None
    scheduler.bootstrap_server = None
    scheduler.engine_id = "d"
    scheduler.pp_size = scheduler.pcp_size = scheduler.tp_size = scheduler.dcp_size = 1
    scheduler.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    request.addfinalizer(scheduler.shutdown)
    scheduler._reqs_pending_recv = {}
    scheduler.side_channel_host, scheduler.side_channel_port = "127.0.0.1", 9000
    scheduler.side_channel_port = 0
    scheduler.set_xfer_handshake_metadata_from_workers(
        {
            (0, 0, 0): MooncakeTransferMetadata(
                engine_id="d",
                te_rpc_port=9001,
                block_size=16,
                num_blocks=100,
                layer_names=["layer.0"],
                layer_block_sizes=[16],
                group_indices=[0],
                kv_caches_base_addr=[[4096]],
                block_strides=[[128]],
                block_lens=[[128]],
                block_shapes=[[(16, 8)]],
                block_size_scales=[[1]],
                local_ip="127.0.0.1",
                handshake_port=worker.handshake_port,
            )
        }
    )
    scheduler.side_channel_port = int(scheduler._metadata_thread.address.rsplit(":", 1)[1])
    params = dict(
        do_remote_prefill=True,
        transfer_id="t",
        remote_engine_id="p",
        remote_bootstrap_addr="http://bootstrap",
        num_computed_tokens=0,
    )
    request = SimpleNamespace(request_id="rt", kv_transfer_params=params)
    blocks = ([1, 2],)
    scheduler.update_state_after_alloc(request, SimpleNamespace(get_block_ids=lambda: blocks), 32)
    assert scheduler._reqs_pending_recv["rt"][0] is request
    # Cancellation must not discard the pending receive and strand pinned blocks.
    assert scheduler.request_finished(request, blocks) == (False, None)
    connector = MooncakePushConnector.__new__(MooncakePushConnector)
    connector.connector_worker = worker
    connector._connector_metadata = scheduler.build_connector_meta(None)
    assert not scheduler._reqs_pending_recv
    blocks[0].append(99)
    connector.start_load_kv(None)
    connector.start_load_kv(None)  # duplicate submission is idempotent
    ready = messages.get(timeout=3)
    assert ready["block_ids"] == [[1, 2]]
    assert ready["scheduler_address"] == scheduler._metadata_thread.address
    # Simulate P fetching D addresses on a metadata cache miss.
    with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.connect(ready["scheduler_address"])
        sock.send(msgspec.msgpack.encode((GET_META_MSG, ready["consumer_engine_id"])))
        groups = msgspec.msgpack.decode(sock.recv(), type=MooncakeTransferMetadataGroups)
        descriptor = groups.metadata_by_pp_rank[0].metadata_by_pcp_rank[0].metadata_by_tp_rank[0]
        assert descriptor.kv_caches_base_addr == [[4096]]
    assert ready["completion_address"] == worker._recving_thread.address
    assert connector.get_finished(set()) == (set(), set())
    assert done(worker, ready)["kind"] == "done_ack"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        finished = connector.get_finished(set())
        if finished[1]:
            break
        time.sleep(0.01)
    assert finished == (set(), {"rt"})
    done(worker, ready)
    connector.start_load_kv(None)
    assert connector.get_finished(set()) == (set(), set())
    assert messages.empty()
    assert not scheduler.build_connector_meta(None).push_requests


def test_all_sources_required_and_errors_wait_for_drained_writes(worker, producer, monkeypatch):
    address, messages = producer
    monkeypatch.setattr(
        worker._sending_thread, "_get_remote_worker_addresses", lambda *args: {(0, 0, 0): address, (0, 0, 1): address}
    )
    worker.start_load_kv(metadata())
    first, second = messages.get(timeout=3), messages.get(timeout=3)
    assert done(worker, first, writes_drained=False)["kind"] == "error"
    done(worker, first, producer_engine_id="stale")
    done(worker, first, producer_rank=[0, 0, 9])
    assert not worker.get_transfer_results().finished_recving
    done(worker, first, status="error")
    done(worker, first, status="error")
    assert not worker.get_transfer_results().finished_recving
    done(worker, second)
    result = finish(worker)
    assert result.finished_recving == result.failed_recving == {"rt"}


def test_done_listener_progresses_while_discovery_is_blocked(worker, producer, monkeypatch):
    address, messages = producer
    blocked, release = threading.Event(), threading.Event()

    def remote_worker_addresses(client, request):
        if request.transfer_id == "second":
            blocked.set()
            assert release.wait(3)
        return {(0, 0, 0): address}

    monkeypatch.setattr(worker._sending_thread, "_get_remote_worker_addresses", remote_worker_addresses)
    try:
        worker.start_load_kv(metadata())
        ready = messages.get(timeout=3)
        worker.start_load_kv(metadata("second"))
        assert blocked.wait(3)
        done(worker, ready)
        assert finish(worker).finished_recving == {"rt"}
    finally:
        release.set()


def test_discovery_failure_is_safe_to_report(worker, monkeypatch):
    def fail(*args):
        raise ValueError("Incomplete registration")

    monkeypatch.setattr(worker._sending_thread, "_get_remote_worker_addresses", fail)
    worker.start_load_kv(metadata())
    result = finish(worker)
    assert result.finished_recving == result.failed_recving == {"rt"}


def test_uncertain_ready_cannot_release_destination(worker, producer, monkeypatch):
    monkeypatch.setattr(worker._sending_thread, "_get_remote_worker_addresses", lambda *args: {(0, 0, 0): producer[0]})

    def lost_ack(*args):
        raise TimeoutError("Lost ACK after Ready")

    monkeypatch.setattr(worker._sending_thread, "_send_ready", lost_ack)
    worker.start_load_kv(metadata())
    with pytest.raises(RuntimeError, match="outstanding WRITEs"):
        finish(worker)
    assert "t" in worker._reqs_recv_state
    with pytest.raises(RuntimeError, match="outstanding WRITEs"):
        worker.get_finished()


def test_bootstrap_parsing_rejects_partial_topology(worker):
    entry = dict(
        engine_id="p", pp_size=1, pcp_size=1, tp_size=2, worker_addr={"0": {"0": {"0": "tcp://127.0.0.1:1234"}}}
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"0": entry}))
    request = metadata().push_requests["rt"]
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ValueError, match="incomplete"):
            worker._sending_thread._get_remote_worker_addresses(client, request)
        entry["worker_addr"]["0"]["1"] = {"0": "tcp://127.0.0.1:1235"}
        remote_worker_addresses = worker._sending_thread._get_remote_worker_addresses(client, request)
        assert set(remote_worker_addresses) == {(0, 0, 0), (0, 0, 1)}
        assert worker._sending_thread._get_remote_worker_addresses(client, request) is remote_worker_addresses


def test_conflicting_transfer_is_rejected(worker, producer, monkeypatch):
    monkeypatch.setattr(worker._sending_thread, "_get_remote_worker_addresses", lambda *args: {(0, 0, 0): producer[0]})
    meta = metadata()
    worker.start_load_kv(meta)
    meta.push_requests["rt"] = replace(meta.push_requests["rt"], local_block_ids=((3,),))
    with pytest.raises(ValueError, match="Conflicting"):
        worker.start_load_kv(meta)


@pytest.mark.parametrize("sender", [False, True])
def test_push_thread_start_failure_is_reported_and_stopped(worker, monkeypatch, sender):
    def fail(*args, **kwargs):
        raise OSError("Cannot initialize control transport")

    if sender:
        monkeypatch.setattr(push_worker.httpx, "Client", fail)
        thread = push_worker.MooncakeReadySendingThread(worker)
    else:
        monkeypatch.setattr(push_worker, "zmq_ctx", fail)
        thread = push_worker.MooncakeCompletionThread(worker)
    with pytest.raises(RuntimeError, match="failed to start"):
        thread.start()
    assert isinstance(thread.error, OSError)
    assert not thread.is_alive()
    thread.shutdown()


def test_ready_requests_reuse_context_and_close_sockets(worker, producer, monkeypatch):
    address, messages = producer
    monkeypatch.setattr(worker._sending_thread, "_get_remote_worker_addresses", lambda *args: {(0, 0, 0): address})
    make_socket = push_worker.make_zmq_socket
    contexts, sockets = [], []

    def record_socket(**kwargs):
        contexts.append(kwargs["ctx"])
        sock = make_socket(**kwargs)
        sockets.append(sock)
        return sock

    monkeypatch.setattr(push_worker, "make_zmq_socket", record_socket)
    context = worker._sending_thread.zmq_context
    for transfer_id in ("first", "second"):
        worker.start_load_kv(metadata(transfer_id))
        ready = messages.get(timeout=3)
        done(worker, ready)
        assert finish(worker).finished_recving == {"r" + transfer_id}
    worker.shutdown()
    assert len(contexts) == 2
    assert all(item is context for item in contexts)
    assert all(sock.closed for sock in sockets)
    assert context.closed
    assert worker._sending_thread.zmq_context is None


@pytest.mark.parametrize("ack_failed", [False, True])
def test_done_before_ready_ack_waits_for_sender_outcome(worker, producer, monkeypatch, ack_failed):
    address, _ = producer
    monkeypatch.setattr(worker._sending_thread, "_get_remote_worker_addresses", lambda *args: {(0, 0, 0): address})
    ready_messages = queue.Queue()
    release = threading.Event()
    calls = []

    def send_ready(address, message):
        calls.append(message["transfer_id"])
        ready_messages.put(message)
        assert release.wait(3)
        if ack_failed:
            raise TimeoutError("Lost Ready ACK after P completed all WRITEs")

    monkeypatch.setattr(worker._sending_thread, "_send_ready", send_ready)
    worker.start_load_kv(metadata())
    try:
        ready = ready_messages.get(timeout=3)
        done(worker, ready)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            assert not worker.get_transfer_results().finished_recving
            if worker._reqs_recv_state["t"].finished_ranks:
                break
            time.sleep(0.01)
        assert worker._reqs_recv_state["t"].finished_ranks == {(0, 0, 0)}
    finally:
        release.set()
    result = finish(worker)
    assert result.finished_recving == {"rt"}
    assert result.failed_recving == ({"rt"} if ack_failed else set())
    assert calls == ["t"]  # no request-level resend after an ACK failure


def test_done_ack_send_error_reports_request_failure(worker, producer, monkeypatch):
    address, messages = producer
    monkeypatch.setattr(worker._sending_thread, "_get_remote_worker_addresses", lambda *args: {(0, 0, 0): address})
    worker.start_load_kv(metadata())
    ready = messages.get(timeout=3)

    def fail_send(frames):
        raise zmq.Again()

    # The source has certified drained WRITEs, but returning its ACK fails.
    message = {**ready, "kind": "done", "status": "ok", "writes_drained": True}
    worker._recving_thread._handle_frames(
        SimpleNamespace(send_multipart=fail_send), [b"producer", b"", msgspec.msgpack.encode(message)]
    )
    result = finish(worker)
    assert result.finished_recving == result.failed_recving == {"rt"}


def test_partial_ready_failure_stops_the_request_without_releasing_blocks(worker, producer, monkeypatch):
    address, _ = producer
    monkeypatch.setattr(
        worker._sending_thread,
        "_get_remote_worker_addresses",
        lambda *args: {(0, 0, rank): address for rank in range(3)},
    )
    sent_ranks = []

    def send_ready(address, message):
        sent_ranks.append(message["producer_rank"])
        if message["producer_rank"] == [0, 0, 1]:
            raise TimeoutError("Second source ACK failed")

    monkeypatch.setattr(worker._sending_thread, "_send_ready", send_ready)
    worker.start_load_kv(metadata())
    with pytest.raises(RuntimeError, match="outstanding WRITEs"):
        finish(worker)
    assert sent_ranks == [[0, 0, 0], [0, 0, 1]]
    assert worker._reqs_recv_state["t"].failed
    assert "t" not in worker._finished_transfers
    assert worker._stop_event.is_set()
