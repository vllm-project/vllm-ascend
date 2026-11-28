# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Bounded-deadline, versioned-protocol and recovery tests for the lookup RPC.

Each failure case asserts three things together -- the value the scheduler sees,
that the client survived, and that a fresh REQ was opened -- because a fail-open
that silently retires the socket is indistinguishable at the scheduler from
"the pool has nothing", which is exactly the failure mode this change removes.
"""

import errno
import json
import logging
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import zmq
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from zmq.constants import POLLIN, RCVTIMEO, REP, SNDTIMEO
from zmq.error import Again, ZMQError
from zmq.sugar.context import Context
from zmq.sugar.poll import Poller

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import lookup_key
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key import (
    DEFAULT_TIMEOUT_MS,
    EXPECTED_LOOKUP_FRAMES,
    SHUTDOWN_JOIN_TIMEOUT_S,
    WARN_EVERY_N,
    LookupKeyClient,
    LookupKeyServer,
    decode_lookup_result,
    validate_timeout_ms,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import StoreLookupResult, TailKeyBoundary

MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key"
HASH_HEX = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"


@pytest.fixture
def sockets(monkeypatch):
    """Every REQ/REP handle this test opens, for as long as the test runs.

    The patch has to outlive construction: recovery happens inside a later call,
    so a context manager around the constructor alone would let the rebuilt
    socket escape the recording.
    """
    created: list = []

    def _make(ctx, path, sock_type, bind=False):
        sock = MagicMock(name=f"sock{len(created)}")
        created.append(sock)
        return sock

    monkeypatch.setattr(lookup_key, "make_zmq_socket", _make)
    monkeypatch.setattr(lookup_key.zmq, "Context", MagicMock(), raising=False)
    return created


def make_config(**extra):
    config = MagicMock()
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_index = 0
    config.kv_transfer_config.kv_connector_extra_config = dict(extra)
    return config


def make_scheduler_cache_config(config):
    groups = [
        KVCacheGroupSpec(
            ["model.layers.0.attn", "model.layers.1.attn"],
            FullAttentionSpec(
                block_size=config.cache_config.block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        )
    ]
    return SimpleNamespace(
        kv_cache_groups=groups,
        hash_block_size=config.cache_config.block_size,
        cache_hit_alignment_tokens=config.cache_config.block_size,
    )


class FakeClock:
    """Monotonic source a case can advance, so budget arithmetic is observable."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance_ms(self, ms):
        self.now += ms / 1000.0


class FakeDecoder:
    """Stands in for MsgpackDecoder so a case controls the decoded shape."""

    def __init__(self, payload=None, raises=None):
        self.payload = payload
        self.raises = raises
        self.calls = []

    def decode(self, frames):
        self.calls.append(tuple(bytes(f) for f in frames))
        if self.raises is not None:
            raise self.raises
        return self.payload


def build_client(sockets, clock=None, **extra):
    return LookupKeyClient(make_config(**extra), clock=clock or FakeClock())


def build_server(sockets):
    """A server whose thread never starts and whose envelopes are recorded."""
    envelopes = []
    encoder = MagicMock()

    def record_envelope(payload):
        envelopes.append(payload)
        return [b"envelope"]

    encoder.encode.side_effect = record_envelope
    with patch(f"{MODULE}.threading.Thread"):
        server = LookupKeyServer(MagicMock(), make_config())
    server.encoder = encoder
    server.envelopes = envelopes  # type: ignore[attr-defined]
    return server


def use_decodes(server, *values):
    server.decoder = FakeDecoder(payload=None)
    server._decode_list = MagicMock(side_effect=list(values))
    return server


def int_frame(value):
    return value.to_bytes(4, byteorder="big")


def lookup_frames(token_len=64, groups=b"group-frame", hbm=8, hashes=b"hash-frame"):
    return [b"lookup", int_frame(token_len), groups, int_frame(hbm), hashes]


def ok_payload(hit, boundaries=()):
    return {"version": 1, "status": "ok", "hit_tokens": hit, "tail_key_boundaries": boundaries}


def err_payload(code):
    return {"version": 1, "status": "error", "code": code, "message": "boom"}


# --------------------------------------------------------------------------------------
# Clause 1: lookup_rpc_timeout_ms is a positive int of milliseconds, rejected at init.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [0, True, "5000", 2**31],
    ids=["zero", "bool", "str", "over-option-cap"],
)
def test_timeout_rejects_every_unusable_shape(value):
    """Both families: not a positive int, and an int too large to be a real deadline."""
    with pytest.raises(ValueError) as excinfo:
        validate_timeout_ms(value)
    assert "lookup_rpc_timeout_ms" in str(excinfo.value)
    if value == 0:
        assert "unbounded" in str(excinfo.value)


def test_timeout_accepts_positive_ints():
    assert validate_timeout_ms(DEFAULT_TIMEOUT_MS) == DEFAULT_TIMEOUT_MS


@pytest.mark.parametrize("value", [0], ids=["zero"])
def test_scheduler_construction_rejects_a_bad_timeout(value):
    """Rejection belongs at engine startup, not on the first lazy client build."""
    from tests.ut.distributed.ascend_store.test_pool_scheduler import make_config as scheduler_config
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

    config = scheduler_config(block_size=16)
    config.kv_transfer_config.kv_connector_extra_config = {"lookup_rpc_timeout_ms": value}
    kv_cache_config = make_scheduler_cache_config(config)
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.resolve_backend"
    ) as mock_backend:
        mock_backend.return_value = ("mock", MagicMock())
        with pytest.raises(ValueError) as excinfo:
            KVPoolScheduler(config, use_layerwise=False, kv_cache_config=kv_cache_config)
    assert "lookup_rpc_timeout_ms" in str(excinfo.value)


@pytest.mark.parametrize("value", [0], ids=["zero"])
def test_construction_fails_before_any_socket_is_opened(sockets, value):
    with pytest.raises(ValueError):
        build_client(sockets, lookup_rpc_timeout_ms=value)
    assert sockets == []


def test_client_defaults_to_five_seconds_on_both_legs(sockets):
    client = build_client(sockets)
    assert client.timeout_ms == DEFAULT_TIMEOUT_MS
    options = [(call.args[0], call.args[1]) for call in client.socket.setsockopt.call_args_list]
    assert (SNDTIMEO, DEFAULT_TIMEOUT_MS) in options
    assert (RCVTIMEO, DEFAULT_TIMEOUT_MS) in options


def test_timeout_above_the_option_capacity_is_rejected_at_construction(sockets):
    """Clamping the option while the Python-side budget stayed huge was an unbounded wait in disguise."""
    with pytest.raises(ValueError) as excinfo:
        build_client(sockets, lookup_rpc_timeout_ms=2**62)
    assert "2147483647" in str(excinfo.value)
    assert sockets == []


def test_the_honoured_budget_never_exceeds_the_socket_option_capacity(sockets):
    client = build_client(sockets, lookup_rpc_timeout_ms=2**31 - 1)
    values = [call.args[1] for call in client.socket.setsockopt.call_args_list]
    assert values and max(values) == 2**31 - 1


# --------------------------------------------------------------------------------------
# Clauses 2 and 3: one wall-clock budget; every fault fails open to 0 and rebuilds.
# --------------------------------------------------------------------------------------

# Connection-level errnos zmq can raise from send/recv. Named here so the disconnect
# leg of the task guide has a test of its own instead of riding the catch-all branch.
DISCONNECT_ERRNOS = ["ENOTCONN", "ETERM", "EHOSTUNREACH", "ENETDOWN"]

RESPONSE_SHAPES = [
    ("empty-response", [], "decode:frame_count"),
    ("legacy-bare-4-byte-int", "legacy", "decode:decode_error"),
    (
        "unknown-version",
        {"version": 2, "status": "ok", "hit_tokens": 8, "tail_key_boundaries": []},
        "decode:decode_error",
    ),
    (
        "unknown-status",
        {"version": 1, "status": "maybe", "hit_tokens": 8, "tail_key_boundaries": []},
        "decode:field_invalid",
    ),
    (
        "bool-hit-count",
        {"version": 1, "status": "ok", "hit_tokens": True, "tail_key_boundaries": []},
        "decode:field_invalid",
    ),
    (
        "over-limit-hit",
        {"version": 1, "status": "ok", "hit_tokens": 65, "tail_key_boundaries": []},
        "decode:field_invalid",
    ),
    ("error-envelope", err_payload("backend_error"), "server:backend_error"),
]


@pytest.mark.parametrize("name, shape, expected_code", RESPONSE_SHAPES, ids=[s[0] for s in RESPONSE_SHAPES])
def test_every_bad_response_shape_fails_open_and_rebuilds(sockets, name, shape, expected_code):
    client = build_client(sockets)
    if shape == "legacy":
        # A pre-upgrade reply: one frame holding a bare 4-byte big-endian int.
        client.decoder = FakeDecoder(payload=int.from_bytes(b"\x00\x00\x00\x10", "big"))
        client.socket.recv_multipart.return_value = [b"\x00\x00\x00\x10"]
    elif isinstance(shape, list) and name in ("empty-response", "two-frames", "zero-length-frame"):
        client.decoder = FakeDecoder(payload=ok_payload(8))
        client.socket.recv_multipart.return_value = shape
    else:
        client.decoder = FakeDecoder(payload=shape)
        client.socket.recv_multipart.return_value = [b"envelope"]

    assert client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8) == StoreLookupResult(0)
    assert not client.closed, "a fail-open must leave a usable client behind"
    assert len(sockets) == 2
    sockets[0].close.assert_called_once_with(linger=0)
    client.ctx.term.assert_not_called()
    assert client.snapshot()[expected_code] == 1


def test_next_lookup_after_a_fault_succeeds(sockets):
    client = build_client(sockets)
    client.decoder = FakeDecoder(payload=Again())
    client.socket.recv_multipart.side_effect = Again()
    assert client.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(0)

    client.socket = sockets[1]
    client.decoder = FakeDecoder(payload=ok_payload(24))
    sockets[1].recv_multipart.return_value = [b"envelope"]
    assert client.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(24)
    assert client.snapshot() == {"recv:timeout": 1}


def test_send_wait_and_validation_share_one_budget_not_one_each(sockets):
    """An implementation giving each leg a fresh budget would still pass three
    timeout legs; only the shared-arithmetic case below catches that."""
    clock = FakeClock()
    client = build_client(sockets, clock=clock, lookup_rpc_timeout_ms=100)
    client.socket.send_multipart.side_effect = lambda *a, **k: clock.advance_ms(60)
    client.socket.recv_multipart.side_effect = lambda *a, **k: (clock.advance_ms(30), [b"envelope"])[1]
    client.decoder = FakeDecoder(payload=ok_payload(16))

    with patch(
        f"{MODULE}.decode_lookup_result",
        side_effect=lambda payload, token_len, group_ids: clock.advance_ms(20) or StoreLookupResult(16),
    ):
        assert client.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(0)

    assert client.snapshot() == {"recv:timeout": 1}
    assert len(sockets) == 2


def test_recv_gets_the_remaining_budget_not_a_fresh_one(sockets):
    clock = FakeClock()
    client = build_client(sockets, clock=clock, lookup_rpc_timeout_ms=1000)
    client.socket.send_multipart.side_effect = lambda *a, **k: clock.advance_ms(400)
    client.decoder = FakeDecoder(payload=ok_payload(0))
    client.socket.recv_multipart.return_value = [b"envelope"]

    assert client.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(0)
    rcvtimeo = [c.args[1] for c in client.socket.setsockopt.call_args_list if c.args[0] == RCVTIMEO]
    assert rcvtimeo[-1] == 600


def test_req_state_from_send_and_from_recv_are_separate_categories_of_one_cause(sockets):
    client = build_client(sockets)
    client.decoder = FakeDecoder(payload=None)
    client.socket.send_multipart.side_effect = ZMQError(int(zmq.constants.EFSM))
    assert client.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(0)
    assert client.snapshot() == {"send:req_state": 1}

    other = build_client(sockets)
    other.decoder = FakeDecoder(payload=None)
    other.socket.recv_multipart.side_effect = ZMQError(int(zmq.constants.EFSM))
    assert other.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(0)
    assert other.snapshot() == {"recv:req_state": 1}
    assert not client.closed and not other.closed
    assert len(sockets) == 4
    assert client.socket is sockets[1], "the failing REQ must be replaced before the call returns"
    assert other.socket is sockets[3]
    sockets[0].close.assert_called_once_with(linger=0)
    sockets[2].close.assert_called_once_with(linger=0)


def test_again_from_send_is_a_timeout(sockets):
    client = build_client(sockets)
    client.decoder = FakeDecoder(payload=None)
    client.socket.send_multipart.side_effect = Again()
    assert client.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(0)
    assert client.snapshot() == {"send:timeout": 1}
    assert client.socket is sockets[1]
    sockets[0].close.assert_called_once_with(linger=0)


def test_lookup_after_close_returns_zero_without_opening_sockets(sockets):
    client = build_client(sockets)
    client.close()
    assert client.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(0)
    assert len(sockets) == 1
    sockets[0].send_multipart.assert_not_called()


def test_client_close_is_idempotent(sockets):
    client = build_client(sockets)
    client.close()
    client.close()
    client.ctx.term.assert_called_once_with()
    sockets[0].close.assert_called_once_with(linger=0)


# --------------------------------------------------------------------------------------
# Task 2: the server refuses malformed requests, with one versioned error each.
# --------------------------------------------------------------------------------------

REQUEST_CASES = [
    ("too-few-frames", lambda: lookup_frames()[: EXPECTED_LOOKUP_FRAMES - 1], [0], [HASH_HEX], "frame_count"),
    ("too-many-frames", lambda: lookup_frames() + [b"extra"], [0], [HASH_HEX], "frame_count"),
    ("short-int-frame", lambda: [b"lookup", b"\x00\x00", b"g", int_frame(0), b"h"], [0], [HASH_HEX], "field_invalid"),
    ("undecodable-payload", lambda: lookup_frames(groups=b"g"), None, None, "decode_error"),
    ("duplicate-group-id", lambda: lookup_frames(groups=b"g"), [0, 0], [HASH_HEX], "field_invalid"),
    ("non-string-hash", lambda: lookup_frames(groups=b"g"), [0], [b"bytes"], "field_invalid"),
    ("oversized-hash", lambda: lookup_frames(groups=b"g"), [0], ["z" * 2000], "field_invalid"),
    ("hbm-above-token-len", lambda: lookup_frames(token_len=10, hbm=11, groups=b"g"), [0], [HASH_HEX], "field_invalid"),
]


@pytest.mark.parametrize(
    "name, frames_fn, groups, hashes, expected_code", REQUEST_CASES, ids=[c[0] for c in REQUEST_CASES]
)
def test_malformed_requests_are_refused_with_a_versioned_error(sockets, name, frames_fn, groups, hashes, expected_code):
    server = build_server(sockets)
    if expected_code == "decode_error":
        server._decode_list = MagicMock(side_effect=lookup_key.ProtocolViolation("decode:decode_error"))
    else:
        use_decodes(server, groups, hashes)

    with pytest.raises(lookup_key.ProtocolViolation) as excinfo:
        server._handle_lookup(frames_fn())

    assert excinfo.value.code == f"decode:{expected_code}"
    assert HASH_HEX not in str(excinfo.value)


def test_valid_request_replies_ok_with_the_backend_count(sockets):
    server = build_server(sockets)
    use_decodes(server, [0], [HASH_HEX])
    server.pool_worker.lookup_scheduler.return_value = StoreLookupResult(30)

    server._handle_lookup(lookup_frames(groups=b"g"))

    assert server.envelopes[-1] == {"version": 1, "status": "ok", "hit_tokens": 30, "tail_key_boundaries": ()}


def test_real_miss_is_ok_zero_while_backend_failure_is_an_error(sockets):
    server = build_server(sockets)
    use_decodes(server, [0], [HASH_HEX], [0], [HASH_HEX])

    server.pool_worker.lookup_scheduler.return_value = StoreLookupResult(0)
    server._handle_lookup(lookup_frames(groups=b"g"))
    assert server.envelopes[-1] == {"version": 1, "status": "ok", "hit_tokens": 0, "tail_key_boundaries": ()}

    server.pool_worker.lookup_scheduler.side_effect = RuntimeError("backend exploded")
    server._handle_lookup(lookup_frames(groups=b"g"))
    assert server.envelopes[-1]["status"] == "error"
    assert server.envelopes[-1]["code"] == "backend_error"
    assert server.snapshot() == {"server:backend_error": 1}


@pytest.mark.parametrize(
    "bad, message",
    [
        ("30", "backend result has the wrong type"),
        (StoreLookupResult(65), "backend lookup result out of range"),
        (StoreLookupResult(32, (TailKeyBoundary(2, 32),)), "backend lookup result out of range"),
    ],
    ids=["wrong-type", "hit-over-limit", "reversed-boundary"],
)
def test_out_of_range_backend_results_are_protocol_errors(sockets, bad, message):
    server = build_server(sockets)
    use_decodes(server, [0], [HASH_HEX])
    server.pool_worker.lookup_scheduler.return_value = bad

    server._handle_lookup(lookup_frames(groups=b"g"))

    assert server.envelopes[-1] == {
        "version": 1,
        "status": "error",
        "code": "field_invalid",
        "message": message,
    }


def test_server_preserves_tail_key_boundaries(sockets):
    server = build_server(sockets)
    use_decodes(server, [0, 1], [HASH_HEX])
    boundaries = (TailKeyBoundary(0, 36), TailKeyBoundary(1, 32))
    server.pool_worker.lookup_scheduler.return_value = StoreLookupResult(32, boundaries)

    server._handle_lookup(lookup_frames(token_len=64, groups=b"g"))

    assert server.envelopes[-1] == {
        "version": 1,
        "status": "ok",
        "hit_tokens": 32,
        "tail_key_boundaries": boundaries,
    }


def test_multi_group_request_reaches_the_backend_unchanged(sockets):
    server = build_server(sockets)
    seen: dict = {}

    def lookup_scheduler(token_len, hashes, groups, use_layerwise, hbm_hit_tokens):
        seen.update(token_len=token_len, hashes=hashes, groups=groups, use_layerwise=use_layerwise, hbm=hbm_hit_tokens)
        return StoreLookupResult(12)

    use_decodes(server, [0, 1, 2], [HASH_HEX, "b" * 16])
    server.pool_worker.lookup_scheduler.side_effect = lookup_scheduler

    server._handle_lookup(lookup_frames(token_len=128, groups=b"g", hbm=16))

    assert seen == {
        "token_len": 128,
        "hashes": [HASH_HEX, "b" * 16],
        "groups": [0, 1, 2],
        "use_layerwise": False,
        "hbm": 16,
    }
    assert server.envelopes[-1]["hit_tokens"] == 12


# --------------------------------------------------------------------------------------
# Clauses 4 and 5: one reply per request, the loop survives, the REP is rebuilt.
# --------------------------------------------------------------------------------------


def poller_for(*batches):
    poller = MagicMock()
    poller.poll.side_effect = list(batches)
    return poller


def test_receive_fault_while_running_rebinds_and_keeps_serving(sockets):
    server = build_server(sockets)
    poller = poller_for([(sockets[0], 1)], [])
    events = []
    poller.unregister.side_effect = lambda sock: events.append("unregister")
    sockets[0].close.side_effect = lambda linger=None: events.append("close")
    sockets[0].recv_multipart.side_effect = ZMQError(errno.ENOTCONN)

    server._serve_iteration(poller)

    assert events == ["unregister", "close"]
    assert server.running
    assert len(sockets) == 2, "a broken REP must be replaced, not abandoned"
    sockets[0].close.assert_called_once_with(linger=0)
    poller.unregister.assert_called_once_with(sockets[0])
    poller.register.assert_called_with(sockets[1], POLLIN)
    sockets[0].send_multipart.assert_not_called()
    assert server.snapshot() == {"server:transport": 1}


def test_spurious_poll_timeout_does_not_rebuild_the_handle(sockets):
    server = build_server(sockets)
    poller = MagicMock()
    poller.poll.return_value = []

    server._serve_iteration(poller)

    assert len(sockets) == 1
    sockets[0].close.assert_not_called()


def test_poll_fault_on_a_dead_handle_rebuilds_the_rep(sockets):
    """Contract 5's unrecoverable-REP case: a poll fault must rebuild, not spin on a dead handle."""
    server = build_server(sockets)
    poller = MagicMock()
    poller.poll.side_effect = ZMQError(errno.ENOTSOCK)

    server._serve_iteration(poller)

    assert server.running
    assert len(sockets) == 2, "a REP the poller can no longer read must be replaced"
    sockets[0].close.assert_called_once_with(linger=0)
    poller.register.assert_called_with(sockets[1], POLLIN)
    sockets[0].send_multipart.assert_not_called()
    assert server.snapshot() == {"server:transport": 1}


def test_malformed_request_is_answered_once_and_the_loop_survives(sockets):
    server = build_server(sockets)
    poller = poller_for([(sockets[0], 1)], [(sockets[0], 1)], [])
    sockets[0].recv_multipart.side_effect = [lookup_frames()[:3], lookup_frames()]
    use_decodes(server, [0], [HASH_HEX], [0], [HASH_HEX])
    server.pool_worker.lookup_scheduler.return_value = StoreLookupResult(24)

    server._serve_iteration(poller)
    server._serve_iteration(poller)

    assert sockets[0].send_multipart.call_count == 2
    assert server.envelopes[0]["version"] == 1
    assert server.envelopes[0]["code"] == "frame_count"
    assert "<redacted>" not in server.envelopes[0]["message"]
    assert server.envelopes[1]["status"] == "ok"
    assert server.running


def test_unknown_message_type_gets_one_error_envelope(sockets):
    server = build_server(sockets)
    poller = poller_for([(sockets[0], 1)])
    sockets[0].recv_multipart.return_value = [b"nonsense"]

    server._serve_iteration(poller)

    assert sockets[0].send_multipart.call_count == 1
    assert server.envelopes[-1] == {
        "version": 1,
        "status": "error",
        "code": "field_invalid",
        "message": "unknown msg_type",
    }


@contextmanager
def capture_lookup_logs(level):
    """Collect what lookup_key actually logs, with no dependence on log propagation.

    vLLM's own dictConfig sets the "vllm" logger to ``propagate: False`` and gives it a
    stream handler, so a root-level catcher (pytest's ``caplog``) sees zero records in a
    real-vLLM run while seeing everything under a stdlib-logger shim. Attaching to the
    logger the module holds is the same assertion in both environments.
    """

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    records: list = []
    log = lookup_key.logger
    handler = _Collector(level)
    previous = log.level
    log.addHandler(handler)
    log.setLevel(level)
    try:
        yield records
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)


def test_unknown_msg_type_is_logged_by_length_and_redacted(sockets):
    """Task 5's log whitelist covers this line too: a hash-shaped tag must not echo."""
    server = build_server(sockets)
    poller = poller_for([(sockets[0], 1)])
    raw_tag = bytes.fromhex(HASH_HEX)
    sockets[0].recv_multipart.return_value = [raw_tag]

    with capture_lookup_logs(logging.WARNING) as records:
        server._serve_iteration(poller)

    text = "\n".join(r.getMessage() for r in records if "unknown msg_type" in r.getMessage())
    assert f"len={len(raw_tag)}" in text
    assert "<32-byte binary frame>" in text
    # Neither the hex spelling nor the escaped-repr spelling of the key may appear.
    assert HASH_HEX not in text
    assert "\\xaa" not in text


def test_ok_debug_line_reports_group_count_not_the_group_list(sockets):
    server = build_server(sockets)
    use_decodes(server, [3, 7], [HASH_HEX])
    server.pool_worker.lookup_scheduler.return_value = StoreLookupResult(12)

    with capture_lookup_logs(logging.DEBUG) as records:
        server._handle_lookup(lookup_frames(groups=b"group-frame"))

    text = "\n".join(r.getMessage() for r in records if "KV pool lookup response" in r.getMessage())
    assert "token_len=64" in text
    assert "group_count=2" in text
    assert "groups=[" not in text
    assert HASH_HEX not in text


def test_reset_keeps_the_bare_wire_format_and_requires_one_frame(sockets):
    server = build_server(sockets)
    assert server._handle_reset([b"reset"]) == [b"\x01"]
    server.pool_worker.reset_store.assert_called_once_with()
    with pytest.raises(lookup_key.ProtocolViolation) as excinfo:
        server._handle_reset([b"reset", b"extra"])
    assert excinfo.value.code == "decode:frame_count"


def test_reset_backend_failure_reports_the_error_byte(sockets):
    server = build_server(sockets)
    server.pool_worker.reset_store.side_effect = RuntimeError("drain failed")
    assert server._handle_reset([b"reset"]) == [b"\x00"]
    assert server.snapshot() == {"reset:backend_error": 1}


def test_server_close_joins_once_with_a_bound(sockets):
    server = build_server(sockets)
    server.thread = MagicMock()
    server.thread.is_alive.return_value = False

    server.close()
    server.close()

    server.thread.join.assert_called_once_with(SHUTDOWN_JOIN_TIMEOUT_S)


# --------------------------------------------------------------------------------------
# Client reset shares the bounded budget.
# --------------------------------------------------------------------------------------


def test_client_reset_timeout_returns_false_and_rebuilds(sockets):
    client = build_client(sockets, lookup_rpc_timeout_ms=500)
    client.socket.recv.side_effect = Again()

    assert client.reset() is False
    assert client.snapshot() == {"reset:timeout": 1}
    assert not client.closed
    assert len(sockets) == 2


def test_client_reset_after_the_budget_is_spent_reports_timeout(sockets):
    clock = FakeClock()
    client = build_client(sockets, clock=clock, lookup_rpc_timeout_ms=500)
    client.socket.send.side_effect = lambda *a, **k: clock.advance_ms(501)

    assert client.reset() is False
    assert client.snapshot() == {"reset:timeout": 1}
    client.socket.recv.assert_not_called()


def test_client_reset_honours_the_ok_byte(sockets):
    client = build_client(sockets)
    client.socket.recv.return_value = b"\x01"
    assert client.reset() is True
    assert len(sockets) == 1
    assert client.snapshot() == {}


def test_client_reset_rejects_the_error_byte(sockets):
    client = build_client(sockets)
    client.socket.recv.return_value = b"\x00"
    assert client.reset() is False
    assert client.snapshot() == {}


def test_reset_on_a_closed_client_is_false_without_touching_the_socket(sockets):
    client = build_client(sockets)
    client.close()
    assert client.reset() is False
    assert len(sockets) == 1


# --------------------------------------------------------------------------------------
# Task 5: counters per category, throttled warnings, no key material in logs.
# --------------------------------------------------------------------------------------


def test_warning_is_throttled_per_category_with_the_first_one_reported(sockets):
    client = build_client(sockets)
    client.decoder = FakeDecoder(payload=None)
    with capture_lookup_logs(logging.WARNING) as records:
        for _ in range(WARN_EVERY_N + 1):
            sockets[-1].recv_multipart.side_effect = Again()
            assert client.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(0)

    lines = [r.getMessage() for r in records if "KV pool lookup RPC failed" in r.getMessage()]
    assert client.snapshot()["recv:timeout"] == WARN_EVERY_N + 1
    assert len(lines) == 2


def test_failure_logs_carry_counts_but_never_key_material(sockets):
    client = build_client(sockets)
    client.decoder = FakeDecoder(payload=None)
    with capture_lookup_logs(logging.WARNING) as records:
        client.socket.recv_multipart.side_effect = Again()
        client.lookup(64, [bytes.fromhex(HASH_HEX)])

    text = "\n".join(r.getMessage() for r in records)
    assert "code=recv:timeout" in text
    assert "token_len=64" in text
    assert "hash_count=1" in text
    assert "group_count=1" in text
    assert "elapsed_ms=" in text
    assert HASH_HEX not in text


def test_decode_debug_log_does_not_leak_key_material(sockets):
    client = build_client(sockets)
    client.decoder = FakeDecoder(raises=RuntimeError(f"key {HASH_HEX} unreadable"))
    client.socket.recv_multipart.return_value = [b"envelope"]
    with capture_lookup_logs(logging.DEBUG) as records:
        assert client.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(0)
    text = "\n".join(logging.Formatter().format(record) for record in records)
    assert HASH_HEX not in text


def test_counter_keys_stay_inside_the_closed_set(sockets):
    closed_set = {
        "send:timeout",
        "send:req_state",
        "send:transport",
        "recv:timeout",
        "recv:req_state",
        "recv:transport",
        "decode:decode_error",
        "decode:frame_count",
        "decode:field_invalid",
        "server:backend_error",
        "server:field_invalid",
        "server:transport",
        "reset:timeout",
        "reset:req_state",
        "reset:transport",
        "reset:backend_error",
        "unexpected:internal_error",
    }
    client = build_client(sockets)
    client.decoder = FakeDecoder(payload=None)
    sockets[-1].recv_multipart.side_effect = Again()
    client.lookup(64, [bytes.fromhex(HASH_HEX)])
    sockets[-1].recv_multipart.side_effect = ZMQError(errno.ENOTCONN)
    client.lookup(64, [bytes.fromhex(HASH_HEX)])
    assert set(client.snapshot()) <= closed_set

    server = build_server(sockets)
    poller = poller_for([(sockets[-1], 1)], [])
    sockets[-1].recv_multipart.side_effect = ZMQError(errno.ENOTCONN)
    server._serve_iteration(poller)
    assert set(server.snapshot()) <= closed_set


def test_server_error_message_carries_no_exception_payload(sockets):
    server = build_server(sockets)
    use_decodes(server, [0], [HASH_HEX])
    server.pool_worker.lookup_scheduler.side_effect = RuntimeError(f"keys {HASH_HEX} unreadable " + "y" * 400)

    with capture_lookup_logs(logging.DEBUG) as records:
        server._handle_lookup(lookup_frames(groups=b"g"))

    assert HASH_HEX not in server.envelopes[-1]["message"]
    assert len(server.envelopes[-1]["message"]) <= 160
    assert HASH_HEX not in "\n".join(logging.Formatter().format(record) for record in records)


# --------------------------------------------------------------------------------------
# Task 3 boundary: envelope validation is one function, tested standalone.
# --------------------------------------------------------------------------------------


def test_valid_ok_envelopes_pass():
    assert decode_lookup_result(ok_payload(64), 64) == StoreLookupResult(64)


def test_valid_tail_key_boundaries_round_trip():
    payload = ok_payload(32, [[0, 36], [1, 32]])
    assert decode_lookup_result(payload, 64, [0, 1]) == StoreLookupResult(
        32,
        (TailKeyBoundary(0, 36), TailKeyBoundary(1, 32)),
    )


@pytest.mark.parametrize(
    "boundaries",
    [
        None,
        [[True, 32]],
        [[0, 65]],
        [[2, 32]],
    ],
    ids=["none", "bool-group", "end-over-limit", "unknown-group"],
)
def test_invalid_tail_key_boundaries_are_protocol_errors(boundaries):
    with pytest.raises(lookup_key.ProtocolViolation) as excinfo:
        decode_lookup_result(ok_payload(32, boundaries), 64, [0, 1])
    assert excinfo.value.code == "decode:field_invalid"


def test_old_bare_int_reply_is_rejected_as_a_protocol_error():
    with pytest.raises(lookup_key.ProtocolViolation) as excinfo:
        decode_lookup_result(int.from_bytes(b"\x00\x00\x00\x10", "big"), 64)
    assert excinfo.value.code == "decode:decode_error"


def test_error_envelope_surfaces_the_server_code():
    with pytest.raises(lookup_key.ProtocolViolation) as excinfo:
        decode_lookup_result(err_payload("backend_error"), 64)
    assert excinfo.value.code == "server:backend_error"


@pytest.mark.parametrize("code", ["", "not_a_code", HASH_HEX, "backend_error\nextra"])
def test_unknown_server_error_codes_are_protocol_errors(code):
    with pytest.raises(lookup_key.ProtocolViolation) as excinfo:
        decode_lookup_result(err_payload(code), 64)
    assert excinfo.value.code == "decode:field_invalid"


def test_unknown_server_code_never_reaches_logs_or_counters(sockets):
    client = build_client(sockets)
    client.decoder = FakeDecoder(payload=err_payload(HASH_HEX))
    client.socket.recv_multipart.return_value = [b"envelope"]
    with capture_lookup_logs(logging.WARNING) as records:
        assert client.lookup(64, [bytes.fromhex(HASH_HEX)]) == StoreLookupResult(0)
    assert client.snapshot() == {"decode:field_invalid": 1}
    assert HASH_HEX not in "\n".join(record.getMessage() for record in records)


# --------------------------------------------------------------------------------------
# Scheduler-level: a failed RPC must look like "no external hit", nothing else.
# --------------------------------------------------------------------------------------


def test_rpc_failure_returns_zero_and_leaves_the_local_path_unloaded(sockets):
    """Scheduler-level view of a failed RPC: (0, False), no load spec, local compute.

    The zero comes from a real client whose recv raises, not from a stubbed
    lookup, so this case also proves recovery ran before the scheduler saw the
    value. The last two lines are the sensitivity control: a scheduler that
    cannot build a load spec at all would pass the "no load spec" assertions
    vacuously.
    """
    from tests.ut.distributed.ascend_store.test_pool_scheduler import make_config as scheduler_config
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

    config = scheduler_config(block_size=16)
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.resolve_backend"
    ) as mock_backend:
        mock_backend.return_value = ("mock", MagicMock())
        scheduler = KVPoolScheduler(
            config,
            use_layerwise=False,
            kv_cache_config=make_scheduler_cache_config(config),
        )

    client = build_client(sockets, lookup_rpc_timeout_ms=250)
    client.socket.recv_multipart.side_effect = Again("timed out")
    scheduler.client = client

    request = MagicMock(prompt_token_ids=list(range(64)), num_tokens=64, request_id="r1", block_hashes=[b"h"] * 4)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert client.snapshot() == {"recv:timeout": 1}
    assert len(sockets) == 2
    assert "r1" not in scheduler.load_specs

    scheduler.update_state_after_alloc(request, MagicMock(), 0)
    assert "r1" not in scheduler.load_specs
    tracked_request, group_block_ids = scheduler._unfinished_requests["r1"]
    assert tracked_request is request
    assert group_block_ids == [[] for _ in scheduler.kv_cache_group_ids]

    client.lookup = lambda *args, **kwargs: StoreLookupResult(32)
    assert scheduler.get_num_new_matched_tokens(request, 0)[0] == 32
    assert scheduler.load_specs["r1"].kvpool_cached_tokens == 32


def test_real_codec_request_is_five_frames_and_decodes(sockets):
    """The frame-count premise is invisible to every mocked case above: only the
    real encoder shows that a list payload occupies exactly one frame."""
    if "serial_utils" not in getattr(lookup_key.MsgpackEncoder, "__module__", ""):
        pytest.skip("the encoder here is the import shim's MagicMock, not vLLM's")
    encoder = lookup_key.MsgpackEncoder()
    frames = [
        b"lookup",
        int_frame(64),
        *encoder.encode([0]),
        int_frame(8),
        *encoder.encode([HASH_HEX]),
    ]
    assert len(frames) == EXPECTED_LOOKUP_FRAMES

    server = build_server(sockets)
    server.pool_worker.lookup_scheduler.return_value = StoreLookupResult(24)
    server._handle_lookup(frames)
    assert server.envelopes[-1] == {"version": 1, "status": "ok", "hit_tokens": 24, "tail_key_boundaries": ()}


# --------------------------------------------------------------------------------------
# Live transport legs. "No reply ever arrived" and "the serving thread exited" are
# properties of a real socket and a real thread, so a mocked handle cannot show them.
# The codec is a stdlib JSON stand-in for vLLM's msgpack one so these legs run wherever
# pyzmq is installed; the wire shape that needs the real codec is a separate case which
# skips when the import shim replaced it.
# --------------------------------------------------------------------------------------


LIVE_TIMEOUT_MS = 250
# The deadline is a wall-clock bound, so the tolerance absorbs scheduler noise on a
# loaded runner. It stays far below the smallest wait a broken build could block for,
# so an unbounded lookup still goes red rather than passing while hanging.
LIVE_TOLERANCE_MS = 1000


class JsonCodec:
    """Encodes each payload as exactly one frame, like vLLM's msgpack codec."""

    def encode(self, obj):
        return [json.dumps(obj).encode()]

    def decode(self, buffers):
        return json.loads(b"".join(buffers))


class BrokenCodec(JsonCodec):
    """Produces frames no decoder can read, standing in for a corrupt request."""

    def encode(self, obj):
        return [b"\xc1\x00\x00"]


def free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("localhost", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


@pytest.fixture
def live_rpc(monkeypatch):
    """A free tcp loopback endpoint, real zmq handles, and guaranteed teardown.

    Windows libzmq has no ipc://, so the shipped path is replaced rather than
    reproduced; nothing under test depends on the transport scheme. Contexts are
    owned here because zmq_ctx_term blocks while a socket is still open and pyzmq
    calls it from Context.__del__ -- a leg that leaked a handle on a failing
    assertion would hang the run instead of going red. destroy(linger=0) closes
    the children first, so it cannot block -- which is exactly why the handle count
    is read BEFORE that call: destroy would mask a handle the code abandoned.
    """
    url = f"tcp://localhost:{free_port()}"
    contexts = []
    handles = []
    real_context = Context

    def _tracked_context(*args):
        ctx = real_context(*args)
        contexts.append(ctx)
        return ctx

    def _factory(ctx, path, sock_type, bind=False):
        sock = ctx.socket(sock_type)
        handles.append(sock)
        if bind:
            sock.bind(path)
        else:
            sock.connect(path)
        return sock

    monkeypatch.setattr(lookup_key, "make_zmq_socket", _factory)
    monkeypatch.setattr(lookup_key, "get_zmq_rpc_path_lookup", lambda cfg: url)
    monkeypatch.setattr(lookup_key.zmq, "Context", _tracked_context)
    yield url
    # Count the leak BEFORE tearing anything down: destroy(linger=0) closes the
    # children, so an abandoned handle would be invisible if counted afterwards.
    leaked = [h for h in handles if not h.closed]
    for ctx in contexts:
        if not ctx.closed:
            ctx.destroy(linger=0)
    assert not leaked, f"recovery or close left {len(leaked)} of {len(handles)} real handles open"


def live_client(timeout_ms):
    client = LookupKeyClient(make_config(lookup_rpc_timeout_ms=timeout_ms))
    # Read the bound back off a real handle: a build that stopped setting RCVTIMEO
    # would otherwise block these legs forever instead of going red.
    assert client.socket.getsockopt(RCVTIMEO) == timeout_ms
    assert client.socket.getsockopt(SNDTIMEO) == timeout_ms
    client.encoder = client.decoder = JsonCodec()
    return client


def silent_peer(url, reply_after_s=None):
    """A bound REP that receives and then stays quiet, or answers too late.

    Returns (arrived, sent, thread): the two events let the caller prove its own
    premise -- that the request really reached the peer, and for the late case that
    a reply was really produced after the deadline instead of never attempted.
    """
    ctx = Context()
    peer = ctx.socket(REP)
    peer.bind(url)
    arrived, sent = threading.Event(), threading.Event()

    def loop():
        poller = Poller()
        poller.register(peer, POLLIN)
        if not poller.poll(timeout=5000):
            return
        peer.recv_multipart()
        arrived.set()
        if reply_after_s is not None:
            time.sleep(reply_after_s)
            peer.send_multipart(
                [json.dumps({"version": 1, "status": "ok", "hit_tokens": 48, "tail_key_boundaries": []}).encode()]
            )
            sent.set()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return arrived, sent, thread


def join_peer(arrived, sent, thread):
    """Let the stub finish before teardown, so no exception is raised under a dead ctx."""
    assert arrived.wait(5.0), "the request has to reach the peer, or nothing was measured"
    thread.join(5.0)
    assert not thread.is_alive(), "the stub peer must finish before the context is destroyed"
    return sent


def measure_ms(fn):
    started = time.monotonic()
    value = fn()
    return value, (time.monotonic() - started) * 1000


def test_absent_server_returns_zero_within_the_deadline(live_rpc):
    client = live_client(LIVE_TIMEOUT_MS)
    try:
        hit, elapsed_ms = measure_ms(lambda: client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8))
    finally:
        close_within(client)
    assert hit == StoreLookupResult(0)
    assert elapsed_ms <= LIVE_TIMEOUT_MS + LIVE_TOLERANCE_MS
    assert client.snapshot() == {"recv:timeout": 1}


def test_silent_server_returns_zero_within_the_deadline(live_rpc):
    arrived, sent, thread = silent_peer(live_rpc)
    client = live_client(LIVE_TIMEOUT_MS)
    try:
        hit, elapsed_ms = measure_ms(lambda: client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8))
    finally:
        close_within(client)
    assert not join_peer(arrived, sent, thread).is_set(), "the premise is a peer that never replies"
    assert hit == StoreLookupResult(0)
    assert elapsed_ms <= LIVE_TIMEOUT_MS + LIVE_TOLERANCE_MS
    assert client.snapshot() == {"recv:timeout": 1}


def test_late_server_reply_is_not_adopted_within_the_deadline(live_rpc):
    arrived, sent, thread = silent_peer(live_rpc, reply_after_s=2 * LIVE_TIMEOUT_MS / 1000)
    client = live_client(LIVE_TIMEOUT_MS)
    try:
        hit, elapsed_ms = measure_ms(lambda: client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8))
    finally:
        close_within(client)
    assert join_peer(arrived, sent, thread).is_set(), "a late reply has to exist, or the leg only measured a silent one"
    assert hit == StoreLookupResult(0), "a reply that lands after the deadline must not be used"
    assert elapsed_ms <= LIVE_TIMEOUT_MS + LIVE_TOLERANCE_MS
    assert client.snapshot() == {"recv:timeout": 1}


def test_the_same_client_returns_a_fresh_value_after_a_late_reply(live_rpc):
    """契约里的后半句在真句柄上量：一次"迟到但确实发出去了"的回复之后，同一个 client
    的下一次 lookup 必须拿到**新**值。既不能是 0（那等于恢复没发生），也不能是迟到的那个
    （那等于旧会话的回复被错认）。

    桩的前提自己也被钉住：后端只第一次超预算，之后立即返回，且每次答案不同。少了这两条，
    这条腿测的就会是"后端一直慢"，那时候返回 0 反而是正确行为。
    """

    class LateOnceWorker:
        def __init__(self):
            self.calls = 0

        def lookup_scheduler(self, token_len, hashes, groups, use_layerwise, hbm_hit_tokens):
            self.calls += 1
            if self.calls == 1:
                time.sleep(2 * LIVE_TIMEOUT_MS / 1000)
            return StoreLookupResult(min(hbm_hit_tokens + 8 + self.calls, token_len))

        def reset_store(self):
            pass

    worker = LateOnceWorker()
    server = LookupKeyServer(worker, make_config())  # type: ignore[arg-type]
    server.decoder = server.encoder = JsonCodec()
    client = live_client(LIVE_TIMEOUT_MS)
    stale = 17  # hbm 8 + 8 + 第 1 次调用
    try:
        late, late_ms = measure_ms(lambda: client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8))
        assert late == StoreLookupResult(0), "the first reply is the late one, so this leg must fail open"
        assert late_ms <= LIVE_TIMEOUT_MS + LIVE_TOLERANCE_MS
        assert client.snapshot() == {"recv:timeout": 1}
        assert worker.calls == 1, "the premise is one slow backend call, not a slow server"

        time.sleep(2 * LIVE_TIMEOUT_MS / 1000)  # let the late reply actually be produced
        assert server.running and server.thread.is_alive(), "the late reply must not end the loop"

        fresh, fresh_ms = measure_ms(lambda: client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8))
        assert fresh == StoreLookupResult(stale + 1), (
            f"recovery must return the new count, got {fresh} (stale would be {stale})"
        )
        assert fresh_ms < LIVE_TIMEOUT_MS, f"a healthy backend must not cost the whole budget: {fresh_ms}ms"
        assert client.snapshot() == {"recv:timeout": 1}, "a recovered call adds no failure category"

        again = client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8)
        assert again == StoreLookupResult(stale + 2), f"the third call must answer afresh, got {again}"
    finally:
        close_within(client)
        server.close()


def test_live_server_answers_a_corrupt_request_then_keeps_serving(live_rpc):
    class Worker:
        def __init__(self):
            self.calls = 0

        def lookup_scheduler(self, token_len, hashes, groups, use_layerwise, hbm_hit_tokens):
            self.calls += 1
            return StoreLookupResult(min(hbm_hit_tokens + 8, token_len))

        def reset_store(self):
            pass

    server = LookupKeyServer(Worker(), make_config())  # type: ignore[arg-type]
    server.decoder = server.encoder = JsonCodec()
    client = live_client(2000)
    try:
        client.encoder = BrokenCodec()
        corrupt, _ = measure_ms(lambda: client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8))
        assert corrupt == StoreLookupResult(0)
        assert client.snapshot() == {"server:decode_error": 1}
        assert server.running and server.thread.is_alive()

        client.encoder = JsonCodec()
        assert client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8) == StoreLookupResult(16)
        assert server.thread.is_alive()
    finally:
        close_within(client)
        server.close()
    assert server.thread.is_alive() is False, "close must retire the serving thread"
    assert server.thread not in threading.enumerate()
    assert server.socket.closed


# --------------------------------------------------------------------------------------
# Task guide item 2's "disconnect", and item 3's "no resource leak": both measured on
# real handles, because the mocked legs cannot see a superseded socket being abandoned.
# --------------------------------------------------------------------------------------


class SimpleWorker:
    """The smallest backend that answers, so a live server can be started and withdrawn."""

    def __init__(self):
        self.calls = 0

    def lookup_scheduler(self, token_len, hashes, groups, use_layerwise, hbm_hit_tokens):
        self.calls += 1
        return StoreLookupResult(min(hbm_hit_tokens + 8, token_len))

    def reset_store(self):
        pass


def close_within(client, seconds=5.0):
    """Run `close()` where a hang cannot take the test session down with it.

    `LookupKeyClient.close` ends in `ctx.term()`, and zmq's term blocks while the
    context still holds an open socket. That makes an abandoned handle a hang rather
    than a leaked fd, so the leak has to be judged by close *returning* in bounded
    time -- a plain assertion on `.closed` would leave CI spinning instead of red.
    """
    thread = threading.Thread(target=client.close, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"close() did not return within {seconds}s: the context still holds an open handle"
    return thread


def test_repeated_timeouts_close_every_superseded_real_handle(live_rpc):
    """Three timeouts are three rebuilds: each retired REQ must be closed by `_recover`
    itself. Checked two ways -- the handle is already closed, and close() still returns,
    which is the shape a leak takes on a real context."""
    client = live_client(LIVE_TIMEOUT_MS)
    seen = [client.socket]
    for _ in range(3):
        assert client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8) == StoreLookupResult(0)
        seen.append(client.socket)
    assert len(seen) == 4, "each timeout must rebuild exactly once, not zero or two"
    assert all(h.closed for h in seen[:-1]), "a rebuilt REQ was left open"
    assert client.snapshot() == {"recv:timeout": 3}
    close_within(client)
    assert all(h.closed for h in seen), "close must retire the handle it currently holds"


def test_a_vanished_server_fails_open_on_every_later_attempt(live_rpc):
    """The measured shape, not an assumed one (see `disconnect_probe`): once the peer is
    gone the first lookup stops at `recv` (timeout), and a socket that is NOT rebuilt
    then throws EFSM from `send` within 0.1 ms. Because this client rebuilds every time,
    every attempt is recorded as `recv:timeout` and returns 0 -- that pairing is the
    point of the leg: EFSM here would mean recovery stopped happening."""
    server = LookupKeyServer(SimpleWorker(), make_config())  # type: ignore[arg-type]
    server.decoder = server.encoder = JsonCodec()
    client = live_client(LIVE_TIMEOUT_MS)
    try:
        assert client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8) == StoreLookupResult(16)
        server.close()
        first, first_ms = measure_ms(lambda: client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8))
        second, second_ms = measure_ms(lambda: client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8))
        assert first == StoreLookupResult(0) and second == StoreLookupResult(0)
        assert client.snapshot() == {"recv:timeout": 2}
        assert first_ms >= 100, f"a vanished peer must spend the budget, not return instantly: {first_ms}"
        assert not client.closed
    finally:
        close_within(client)


@pytest.mark.parametrize("errno_name", DISCONNECT_ERRNOS, ids=DISCONNECT_ERRNOS)
def test_a_disconnect_shaped_errno_is_named_as_transport_not_timeout(sockets, errno_name):
    """The task guide lists 断连 beside timeout / EFSM / malformed / legacy / backend error.
    Those five have their own codes; a connection-level errno must land in
    `<stage>:transport` rather than be disguised as a budget overrun, and it must take the
    same fail-open + linger=0 rebuild as everything else."""
    client = build_client(sockets)
    err = ZMQError(int(getattr(zmq.constants, errno_name)), errno_name)
    client.socket.send_multipart.side_effect = err
    assert client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8) == StoreLookupResult(0)
    assert client.snapshot() == {"send:transport": 1}
    assert not client.closed
    assert len(sockets) == 2
    sockets[0].close.assert_called_once_with(linger=0)


def test_a_failed_rebuild_returns_zero_instead_of_reaching_the_scheduler(sockets, monkeypatch):
    """`_recover` sits inside the fail-open path, so opening the replacement handle can
    itself fail -- EMFILE on a process with no free descriptors is the realistic one.
    That must not turn a bounded lookup into an exception inside the scheduler: the
    contract is that `lookup()` always returns a number, so a client that cannot reopen
    has to keep returning 0 and say which stage broke."""
    client = build_client(sockets)

    def refuse(ctx, path, sock_type, bind=False):
        raise ZMQError(errno.EMFILE, "too many open files")

    client.socket.recv_multipart.side_effect = Again(errno.EAGAIN, "eagain")
    monkeypatch.setattr(lookup_key, "make_zmq_socket", refuse)

    assert client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8) == StoreLookupResult(0)
    assert not client.closed, "a failed reopen is not a reason to retire the client"
    counts = client.snapshot()
    assert counts["recv:timeout"] == 1, counts
    assert any(code.startswith("recover:") for code in counts), counts
    assert client.lookup(64, [bytes.fromhex(HASH_HEX)], hbm_hit_tokens=8) == StoreLookupResult(0)
    assert client.socket is None, "no handle was invented out of a failed reopen"


def test_the_serving_loop_reopens_after_a_failed_rebuild(sockets, monkeypatch):
    """A poll fault rebuilds the REP. If that rebuild cannot open a handle, the loop must
    not settle into polling an empty poller -- the thread would stay alive while the
    server went permanently deaf, which is the exact failure mode this change removes.
    So the next iteration retries the open, and serves again once it succeeds."""
    server = build_server(sockets)

    def open_or_refuse(ctx, path, sock_type, bind=False):
        if state["refuse"]:
            raise ZMQError(errno.EMFILE, "too many open files")
        return real_open(ctx, path, sock_type, bind=bind)

    real_open = lookup_key.make_zmq_socket
    state = {"refuse": True}
    monkeypatch.setattr(lookup_key, "make_zmq_socket", open_or_refuse)
    poller = MagicMock()
    poller.poll.side_effect = ZMQError(errno.EMFILE, "too many open files")

    server._serve_iteration(poller)
    assert server.socket is None, "the failed rebuild must leave no half-open handle"
    assert any(code.startswith("server:") for code in server.snapshot())

    state["refuse"] = False
    poller.poll.side_effect = None
    poller.poll.return_value = []
    server._serve_iteration(poller)
    assert server.socket is not None, "the loop must retry the open, not poll an empty set"
    assert len(sockets) == 2, "exactly one handle opened on the retry"
    poller.register.assert_called_with(sockets[-1], POLLIN)


# --------------------------------------------------------------------------------------
# Review round 2 (zhenwenqi2024): the send leg must spend only the budget the encode
# left behind, and a persistently failing reopen must not spin the serving loop.
# --------------------------------------------------------------------------------------


def test_a_slow_encode_leaves_the_send_only_the_budget_left(live_rpc, monkeypatch):
    """Read SNDTIMEO off the real handle at the instant of sending.

    The option used to be armed once, at socket creation, with the whole configured
    timeout. A slow encode therefore did not shrink it, and "encode then blocked
    send" could wait encode + timeout -- past the bound this parameter exists to
    hold. A mocked setsockopt would only prove the stub recorded a call, so the
    value is read back from the live socket libzmq will actually honour.
    """
    budget_ms = 600
    encode_ms = 200

    class SlowEncoder:
        def encode(self, obj):
            time.sleep(encode_ms / 1000.0)
            return [json.dumps(obj).encode()]

    class SendSpy:
        def __init__(self, inner):
            self.inner = inner
            self.sndtimeo_at_send = None

        def send_multipart(self, frames, copy=True):
            self.sndtimeo_at_send = int(self.inner.getsockopt(SNDTIMEO))
            raise Again("simulated: the peer never read and the send timed out")

        def __getattr__(self, name):
            return getattr(self.inner, name)

    opened = []

    def _factory(ctx, path, sock_type, bind=False):
        real = ctx.socket(sock_type)
        if bind:
            real.bind(path)
        else:
            real.connect(path)
        spy = SendSpy(real)
        opened.append(spy)
        return spy

    monkeypatch.setattr(lookup_key, "make_zmq_socket", _factory)
    client = LookupKeyClient(make_config(lookup_rpc_timeout_ms=budget_ms))
    client.encoder = SlowEncoder()

    started = time.monotonic()
    assert client.lookup(64, [bytes.fromhex(HASH_HEX)], [0], 8) == StoreLookupResult(0)
    waited_ms = (time.monotonic() - started) * 1000

    assert opened[0].sndtimeo_at_send is not None, "the send never reached the socket"
    assert opened[0].sndtimeo_at_send <= budget_ms - encode_ms, (
        f"send was armed with {opened[0].sndtimeo_at_send} ms after {encode_ms} ms of encoding "
        f"against a {budget_ms} ms budget"
    )
    assert waited_ms <= budget_ms + LIVE_TOLERANCE_MS
    assert client.snapshot().get("send:timeout") == 1, "a blocked send must fail open, named"


def test_a_spent_budget_fails_open_before_the_frame_reaches_the_wire(sockets):
    """Nothing is worth sending once the deadline is gone, and sending it would wait again."""
    clock = FakeClock()
    client = build_client(sockets, clock=clock, lookup_rpc_timeout_ms=100)
    real_encode = client.encoder.encode
    # The budget restarts per call, so it has to be spent inside one: the encode is
    # the work that sits between the deadline and the send.
    client.encoder.encode = lambda obj: (clock.advance_ms(100), real_encode(obj))[1]

    assert client.lookup(64, [bytes.fromhex(HASH_HEX)], [0], 8) == StoreLookupResult(0)

    sockets[0].send_multipart.assert_not_called()
    assert client.snapshot().get("send:timeout") == 1


def test_repeated_reopen_failures_log_on_the_throttle_not_once_per_attempt(sockets):
    """The counter is throttled; a second unthrottled warning beside it defeats it."""
    server = build_server(sockets)
    server._open_socket = MagicMock(side_effect=ZMQError(errno.EMFILE, "too many open files"))
    server.socket = None
    poller = MagicMock()

    with capture_lookup_logs(logging.WARNING) as records:
        for _ in range(WARN_EVERY_N * 3):
            server._rebuild_socket(poller)

    assert server.failure_counts.get("server:open_error") == WARN_EVERY_N * 3
    logged = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    # An upper bound would pass on an empty capture -- which is exactly how a broken
    # collector hides. `_note_failure` logs on the first occurrence and then every
    # WARN_EVERY_N, so 3*WARN_EVERY_N failures must produce precisely 4 lines.
    assert len(logged) == 4, (
        f"{len(logged)} warnings for {WARN_EVERY_N * 3} failures"
        " -- expected the first plus every 100th, so the throttle is either off or double-firing"
    )


def test_the_retry_wait_exists_and_close_wakes_it(sockets):
    """The delay has to be there, and it has to be event-driven rather than a sleep."""
    server = build_server(sockets)
    server._open_socket = MagicMock(side_effect=ZMQError(errno.EMFILE, "too many open files"))
    server.socket = None
    poller = MagicMock()

    started = time.monotonic()
    server._serve_iteration(poller)
    armed_ms = (time.monotonic() - started) * 1000
    assert armed_ms >= 20, f"no retry wait at all ({armed_ms:.0f} ms) -- the loop would spin"

    server.close()
    server.running = True
    started = time.monotonic()
    server._serve_iteration(poller)
    woke_ms = (time.monotonic() - started) * 1000
    assert woke_ms < 20, f"close did not wake the retry wait ({woke_ms:.0f} ms)"
