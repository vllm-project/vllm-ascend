#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Failure-path tests for the KV pool lookup RPC.

Both peers must stay responsive when the wire payload is malformed, when the
handler raises, and when ZeroMQ fails mid-exchange: a stuck lookup would block
the scheduler, and a leaked socket would leak IO threads on every retry.
"""

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

# isort: off
from zmq.error import ZMQError

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    LookupKeyServer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    LookupKeyClient,
)

# isort: on

ZERO_HITS = (0).to_bytes(4, "big")

CONNECTOR = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector"
SCHEDULER = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"


def _make_config() -> MagicMock:
    config = MagicMock()
    config.parallel_config.data_parallel_rank = 0
    config.kv_transfer_config.kv_connector_extra_config = {}
    return config


@contextmanager
def _running_server(frames, poller_polls):
    """Start a fake server whose REP loop can be driven from the test body.

    The patches stay active for the whole ``with`` block so the loop body
    (which resolves ``zmq.Poller`` at call time) sees the mocks as well.
    """
    with (
        patch(f"{CONNECTOR}.threading.Thread") as thread_cls,
        patch(f"{CONNECTOR}.zmq.Poller") as poller_cls,
        patch(f"{CONNECTOR}.make_zmq_socket") as make_socket,
        patch(f"{CONNECTOR}.zmq.Context") as context_cls,
    ):
        socket = MagicMock()
        socket.recv_multipart.return_value = frames
        make_socket.return_value = socket

        poller = MagicMock()
        poller.poll.side_effect = poller_polls
        poller_cls.return_value = poller
        thread_cls.return_value = MagicMock()

        server = LookupKeyServer(MagicMock(), _make_config())
        yield server, socket, context_cls, thread_cls


@contextmanager
def _client():
    with (
        patch(f"{SCHEDULER}.make_zmq_socket") as make_socket,
        patch(f"{SCHEDULER}.zmq.Context") as context_cls,
    ):
        socket = MagicMock()
        make_socket.return_value = socket
        client = LookupKeyClient(_make_config())
        client.encoder = MagicMock()
        client.encoder.encode.return_value = [b"frame"]
        yield client, socket, context_cls


class TestLookupKeyServerHardening(unittest.TestCase):
    """The REP loop must answer every request with a 4-byte hit count."""

    def test_truncated_frames_still_reply_zero(self):
        """A payload missing the hash frame must not escape the handler."""
        with _running_server(
            frames=[(64).to_bytes(4, "big")],
            poller_polls=[[(MagicMock(), 1)]],
        ) as (server, socket, context_cls, thread_cls):
            socket.send.side_effect = lambda *_a, **_k: setattr(server, "running", False)

            thread_cls.call_args.kwargs["target"]()

            socket.send.assert_called_once_with(ZERO_HITS)
            socket.close.assert_called_once_with(linger=0)
            context_cls.return_value.term.assert_called_once_with()

    def test_decoder_failure_replies_zero(self):
        """Undecodable groups/hashes are reported as a miss, not an exception."""
        with _running_server(
            frames=[(64).to_bytes(4, "big"), b"groups", (0).to_bytes(4, "big"), b"hashes"],
            poller_polls=[[(MagicMock(), 1)]],
        ) as (server, socket, _context_cls, thread_cls):
            server.decoder = MagicMock()
            server.decoder.decode.side_effect = RuntimeError("bad payload")
            server.pool_worker.lookup_scheduler.side_effect = AssertionError("must not be called")
            socket.send.side_effect = lambda *_a, **_k: setattr(server, "running", False)

            thread_cls.call_args.kwargs["target"]()

            socket.send.assert_called_once_with(ZERO_HITS)

    def test_send_failure_stops_loop_and_releases_socket(self):
        """A broken socket must end the loop instead of spinning forever."""
        with _running_server(
            frames=[(64).to_bytes(4, "big"), b"groups", (0).to_bytes(4, "big"), b"hashes"],
            poller_polls=[[(MagicMock(), 1)]],
        ) as (server, socket, context_cls, thread_cls):
            server.decoder = MagicMock()
            server.decoder.decode.return_value = []
            server.pool_worker.lookup_scheduler.return_value = 0
            socket.send.side_effect = ZMQError("send failed")

            thread_cls.call_args.kwargs["target"]()

            socket.close.assert_called_once_with(linger=0)
            context_cls.return_value.term.assert_called_once_with()

    def test_oversized_payload_is_forwarded_unchanged(self):
        """Many hash frames are legitimate; the handler must not truncate them."""
        hash_frames = [f"hash-{i}".encode() for i in range(512)]
        frames = [(4096).to_bytes(4, "big"), b"groups", (128).to_bytes(4, "big"), *hash_frames]
        with _running_server(
            frames=frames,
            poller_polls=[[(MagicMock(), 1)]],
        ) as (server, socket, _context_cls, thread_cls):
            server.decoder = MagicMock()
            server.decoder.decode.side_effect = [[0], [f"hash-{i}" for i in range(512)]]
            server.pool_worker.lookup_scheduler.return_value = 7
            socket.send.side_effect = lambda *_a, **_k: setattr(server, "running", False)

            thread_cls.call_args.kwargs["target"]()

            socket.send.assert_called_once_with((7).to_bytes(4, "big"))
            args = server.pool_worker.lookup_scheduler.call_args.args
            self.assertEqual(args[0], 4096)
            self.assertEqual(len(args[1]), 512)


class TestLookupKeyClientHardening(unittest.TestCase):
    """The REQ client must degrade to a miss and never leak its context."""

    def test_no_response_times_out_to_zero_and_closes(self):
        """A silent server must not block the scheduler indefinitely."""
        with _client() as (client, socket, context_cls):
            socket.recv.side_effect = ZMQError("timeout")

            hits = client.lookup(16, [MagicMock()], [0], 0)

            self.assertEqual(hits, 0)
            self.assertTrue(client.closed)
            socket.close.assert_called_once_with(linger=0)
            context_cls.return_value.term.assert_called_once_with()

    def test_send_failure_returns_zero_and_closes(self):
        with _client() as (client, socket, _context_cls):
            socket.send_multipart.side_effect = ZMQError("broker gone")

            self.assertEqual(client.lookup(16, [MagicMock()], [0], 0), 0)
            self.assertTrue(client.closed)

    def test_closed_client_short_circuits(self):
        with _client() as (client, socket, _context_cls):
            client.close()
            socket.send_multipart.reset_mock()

            self.assertEqual(client.lookup(16, [MagicMock()], [0], 0), 0)
            socket.send_multipart.assert_not_called()

    def test_malformed_response_is_parsed_leniently(self):
        """An empty reply decodes to 0 instead of raising."""
        with _client() as (client, socket, _context_cls):
            socket.recv.return_value = b""

            self.assertEqual(client.lookup(16, [MagicMock()], [0], 0), 0)

    def test_normal_response_returns_hit_count(self):
        with _client() as (client, socket, _context_cls):
            socket.recv.return_value = (256).to_bytes(4, "big")

            self.assertEqual(client.lookup(16, [MagicMock()], [0], 0), 256)


if __name__ == "__main__":
    unittest.main()
