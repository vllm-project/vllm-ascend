"""Lookup wire identities, failure recovery, and socket ownership."""

from __future__ import annotations

import threading
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from zmq.constants import RCVTIMEO, SNDTIMEO, SocketType
from zmq.sugar.context import Context

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol import rpc
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupCodec,
    LookupRequest,
    LookupResult,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.rpc import LookupClient, LookupServer


@pytest.fixture
def lookup_address():
    # Short paths stay within Unix-domain socket limits on macOS as well as Linux.
    with TemporaryDirectory(prefix="kvlookup-", dir="/tmp") as directory:
        yield f"ipc://{directory}/lookup"


def test_lookup_codec_roundtrip_preserves_groups_and_partial_tail_boundaries() -> None:
    codec = LookupCodec()
    request = LookupRequest(TokenRange(4, 12), (1, 3), (b"a", b"b"))
    result = LookupResult(8, (TailKeyBoundary(1, 12), TailKeyBoundary(3, 8)))
    assert codec.decode_request(codec.encode_request(request)) == request
    assert codec.decode_result(codec.encode_result(result)) == result


def test_lookup_rpc_replies_to_request_failures_and_keeps_serving(lookup_address, caplog) -> None:
    codec = LookupCodec()
    request = LookupRequest(TokenRange(0, 8), (1, 3), (b"a", b"b"))
    expected = LookupResult(8, (TailKeyBoundary(1, 8), TailKeyBoundary(3, 8)))
    query_count = 0

    def lookup(_request):
        nonlocal query_count
        query_count += 1
        if query_count == 1:
            raise RuntimeError("backend query failed")
        if query_count == 2:
            return LookupResult(-1)
        return expected

    server = LookupServer(lookup, lookup_address)
    client = LookupClient(lookup_address)
    try:
        with Context() as context:
            socket = context.socket(SocketType.REQ)
            try:
                socket.setsockopt(RCVTIMEO, 2000)
                socket.connect(lookup_address)
                for frames in ([b"malformed"], codec.encode_request(request), codec.encode_request(request)):
                    socket.send_multipart(frames)
                    assert codec.decode_result(socket.recv()) == LookupResult(0)
                socket.send_multipart(codec.encode_request(request))
                assert codec.decode_result(socket.recv()) == expected
            finally:
                socket.close(linger=0)
        assert client.lookup(request) == expected
        assert caplog.text.count("Lookup request failed") == 3
        client.close()
        client.close()
        with pytest.raises(RuntimeError, match="Lookup client is closed"):
            client.lookup(request)
    finally:
        client.close()
        server.close()
    # Stopping the socket-owning thread also releases its endpoint and Context.
    restarted = LookupServer(lambda _request: expected, lookup_address)
    recovered = LookupClient(lookup_address, timeout_ms=2000)
    try:
        assert recovered.lookup(request) == expected
    finally:
        recovered.close()
        restarted.close()


@pytest.mark.parametrize("failure", ("timeout", "malformed_response", "out_of_range_response"))
def test_lookup_rpc_recreates_connection_without_reusing_a_late_reply(lookup_address, failure, caplog) -> None:
    codec = LookupCodec()
    request = LookupRequest(TokenRange(0, 8), (1,), (b"a", b"b"))
    ready = threading.Event()
    release_first_reply = threading.Event()
    first_reply_sent = threading.Event()
    peer_errors = []

    def serve_peer():
        try:
            with Context() as context:
                socket = context.socket(SocketType.REP)
                try:
                    socket.setsockopt(RCVTIMEO, 3000)
                    socket.bind(lookup_address)
                    ready.set()
                    socket.recv_multipart()
                    if failure == "timeout":
                        assert release_first_reply.wait(timeout=3)
                        first_reply = codec.encode_result(LookupResult(4))
                    elif failure == "malformed_response":
                        first_reply = b"invalid"
                    else:
                        first_reply = codec.encode_result(LookupResult(12))
                    socket.send(first_reply)
                    first_reply_sent.set()
                    socket.recv_multipart()
                    socket.send(codec.encode_result(LookupResult(8, (TailKeyBoundary(1, 8),))))
                finally:
                    socket.close(linger=1000)
        except BaseException as error:
            peer_errors.append(error)
            ready.set()

    peer = threading.Thread(target=serve_peer, daemon=True)
    peer.start()
    client = LookupClient(lookup_address, timeout_ms=1000)
    try:
        assert ready.wait(timeout=3) and not peer_errors
        assert client.lookup(request) == LookupResult(0)
        release_first_reply.set()
        assert first_reply_sent.wait(timeout=3)
        assert client.lookup(request) == LookupResult(8, (TailKeyBoundary(1, 8),))
        assert "Lookup RPC failed" in caplog.text
    finally:
        release_first_reply.set()
        client.close()
        peer.join(timeout=4)
    assert not peer.is_alive() and not peer_errors


@pytest.mark.parametrize("timeout_ms,send_timeout,receive_timeout", ((-1, -1, -1), (0, 0, 0), (1000, 750, 250)))
def test_lookup_rpc_uses_one_timeout_budget_for_send_and_receive(
    monkeypatch, timeout_ms, send_timeout, receive_timeout
) -> None:
    clock = iter((0.0, 0.25, 0.75))
    options = []
    socket = SimpleNamespace(
        setsockopt=lambda option, value: options.append((option, value)),
        send_multipart=lambda *_args, **_kwargs: None,
        recv=lambda: LookupCodec.encode_result(LookupResult(4)),
        close=lambda **_kwargs: None,
    )
    monkeypatch.setattr(rpc, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rpc, "make_zmq_socket", lambda *_args, **_kwargs: socket)
    client = LookupClient("unused", timeout_ms=timeout_ms)
    try:
        assert client.lookup(LookupRequest(TokenRange(0, 4), (0,), (b"a",))) == LookupResult(4)
        assert options == [(SNDTIMEO, send_timeout), (RCVTIMEO, receive_timeout)]
    finally:
        client.close()


def test_lookup_rpc_recovers_when_socket_creation_fails_after_allocating_resources(lookup_address, monkeypatch) -> None:
    server = LookupServer(lambda _request: LookupResult(4), lookup_address)
    make_socket = rpc.make_zmq_socket
    connection_attempts = 0

    def connect(context, address, socket_type, *, bind):
        nonlocal connection_attempts
        connection_attempts += 1
        endpoint = "unsupported://lookup" if connection_attempts == 1 else address
        return make_socket(context, endpoint, socket_type, bind=bind)

    monkeypatch.setattr(rpc, "make_zmq_socket", connect)
    client = LookupClient(lookup_address, timeout_ms=2000)
    request = LookupRequest(TokenRange(0, 4), (0,), (b"a",))
    try:
        assert client.lookup(request) == LookupResult(0)
        assert client.lookup(request) == LookupResult(4)
    finally:
        client.close()
        server.close()


def test_lookup_rpc_close_keeps_active_query_owned_until_it_can_stop(lookup_address) -> None:
    query_started = threading.Event()
    finish_query = threading.Event()

    def lookup(_request):
        query_started.set()
        assert finish_query.wait(timeout=3)
        return LookupResult(4)

    server = LookupServer(lookup, lookup_address)
    try:
        with Context() as context:
            socket = context.socket(SocketType.REQ)
            try:
                socket.setsockopt(RCVTIMEO, 2000)
                socket.connect(lookup_address)
                socket.send_multipart(LookupCodec().encode_request(LookupRequest(TokenRange(0, 4), (0,), (b"a",))))
                assert query_started.wait(timeout=2)
                with pytest.raises(RuntimeError, match="Worker resources must remain open"):
                    server.close(timeout=0)
                finish_query.set()
                server.close(timeout=2)
                assert LookupCodec.decode_result(socket.recv()) == LookupResult(4)
                server.close()
            finally:
                socket.close(linger=0)
    finally:
        finish_query.set()
        server.close()


def test_lookup_rpc_reports_startup_failure_without_waiting_for_a_query() -> None:
    with pytest.raises(RuntimeError, match="Failed to start Lookup server"):
        LookupServer(lambda _request: LookupResult(0), "unsupported://lookup")
