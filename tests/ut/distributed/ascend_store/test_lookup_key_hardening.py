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


def _make_config() -> MagicMock:
    config = MagicMock()
    config.parallel_config.data_parallel_rank = 0
    config.kv_transfer_config.kv_connector_extra_config = {}
    return config


class TestLookupKeyServerHardening(unittest.TestCase):
    """The REP loop must answer every request with a 4-byte hit count."""

    def _start_server(self, frames, poller_polls):
        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.threading.Thread"
            ) as mock_thread_cls,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.zmq.Poller"
            ) as mock_poller_cls,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.make_zmq_socket"
            ) as mock_make_socket,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.zmq.Context"
            ) as mock_context_cls,
        ):
            socket = MagicMock()
            socket.recv_multipart.return_value = frames
            mock_make_socket.return_value = socket

            poller = MagicMock()
            poller.poll.side_effect = poller_polls
            mock_poller_cls.return_value = poller
            mock_thread_cls.return_value = MagicMock()

            server = LookupKeyServer(MagicMock(), _make_config())
            return server, socket, mock_context_cls, mock_thread_cls

    def test_truncated_frames_still_reply_zero(self):
        """A payload missing the hash frame must not escape the handler."""
        server, socket, context_cls, thread_cls = self._start_server(
            frames=[(64).to_bytes(4, "big")],
            poller_polls=[[(MagicMock(), 1)]],
        )
        target = thread_cls.call_args.kwargs["target"]
        socket.send.side_effect = lambda *_a, **_k: setattr(server, "running", False)

        target()

        socket.send.assert_called_once_with(ZERO_HITS)
        socket.close.assert_called_once_with(linger=0)
        context_cls.return_value.term.assert_called_once_with()

    def test_decoder_failure_replies_zero(self):
        """Undecodable groups/hashes are reported as a miss, not an exception."""
        server, socket, _context_cls, thread_cls = self._start_server(
            frames=[(64).to_bytes(4, "big"), b"groups", (0).to_bytes(4, "big"), b"hashes"],
            poller_polls=[[(MagicMock(), 1)]],
        )
        server.decoder = MagicMock()
        server.decoder.decode.side_effect = RuntimeError("bad payload")
        server.pool_worker.lookup_scheduler.side_effect = AssertionError("must not be called")
        target = thread_cls.call_args.kwargs["target"]
        socket.send.side_effect = lambda *_a, **_k: setattr(server, "running", False)

        target()

        socket.send.assert_called_once_with(ZERO_HITS)

    def test_send_failure_stops_loop_and_releases_socket(self):
        """A broken socket must end the loop instead of spinning forever."""
        server, socket, context_cls, thread_cls = self._start_server(
            frames=[(64).to_bytes(4, "big"), b"groups", (0).to_bytes(4, "big"), b"hashes"],
            poller_polls=[[(MagicMock(), 1)]],
        )
        server.decoder = MagicMock()
        server.decoder.decode.return_value = []
        server.pool_worker.lookup_scheduler.return_value = 0
        socket.send.side_effect = ZMQError("send failed")
        target = thread_cls.call_args.kwargs["target"]

        target()

        socket.close.assert_called_once_with(linger=0)
        context_cls.return_value.term.assert_called_once_with()

    def test_oversized_payload_is_forwarded_unchanged(self):
        """Many hash frames are legitimate; the handler must not truncate them."""
        hash_frames = [f"hash-{i}".encode() for i in range(512)]
        frames = [(4096).to_bytes(4, "big"), b"groups", (128).to_bytes(4, "big"), *hash_frames]
        server, socket, _context_cls, thread_cls = self._start_server(
            frames=frames,
            poller_polls=[[(MagicMock(), 1)]],
        )
        server.decoder = MagicMock()
        server.decoder.decode.side_effect = [[0], [f"hash-{i}" for i in range(512)]]
        server.pool_worker.lookup_scheduler.return_value = 7
        target = thread_cls.call_args.kwargs["target"]
        socket.send.side_effect = lambda *_a, **_k: setattr(server, "running", False)

        target()

        socket.send.assert_called_once_with((7).to_bytes(4, "big"))
        args = server.pool_worker.lookup_scheduler.call_args.args
        self.assertEqual(args[0], 4096)
        self.assertEqual(len(args[1]), 512)


class TestLookupKeyClientHardening(unittest.TestCase):
    """The REQ client must degrade to a miss and never leak its context."""

    def _make_client(self):
        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket"
            ) as mock_make_socket,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq.Context"
            ) as mock_context_cls,
        ):
            socket = MagicMock()
            mock_make_socket.return_value = socket
            client = LookupKeyClient(_make_config())
            client.encoder = MagicMock()
            client.encoder.encode.return_value = [b"frame"]
            return client, socket, mock_context_cls

    def test_no_response_times_out_to_zero_and_closes(self):
        """A silent server must not block the scheduler indefinitely."""
        client, socket, context_cls = self._make_client()
        socket.recv.side_effect = ZMQError("timeout")

        hits = client.lookup(16, [MagicMock()], [0], 0)

        self.assertEqual(hits, 0)
        self.assertTrue(client.closed)
        socket.close.assert_called_once_with(linger=0)
        context_cls.return_value.term.assert_called_once_with()

    def test_send_failure_returns_zero_and_closes(self):
        client, socket, _context_cls = self._make_client()
        socket.send_multipart.side_effect = ZMQError("broker gone")

        hits = client.lookup(16, [MagicMock()], [0], 0)

        self.assertEqual(hits, 0)
        self.assertTrue(client.closed)

    def test_closed_client_short_circuits(self):
        client, socket, _context_cls = self._make_client()
        client.close()
        socket.send_multipart.reset_mock()

        hits = client.lookup(16, [MagicMock()], [0], 0)

        self.assertEqual(hits, 0)
        socket.send_multipart.assert_not_called()

    def test_malformed_response_is_parsed_leniently(self):
        """An empty reply decodes to 0 instead of raising."""
        client, socket, _context_cls = self._make_client()
        socket.recv.return_value = b""

        self.assertEqual(client.lookup(16, [MagicMock()], [0], 0), 0)

    def test_normal_response_returns_hit_count(self):
        client, socket, _context_cls = self._make_client()
        socket.recv.return_value = (256).to_bytes(4, "big")

        self.assertEqual(client.lookup(16, [MagicMock()], [0], 0), 256)


if __name__ == "__main__":
    unittest.main()
