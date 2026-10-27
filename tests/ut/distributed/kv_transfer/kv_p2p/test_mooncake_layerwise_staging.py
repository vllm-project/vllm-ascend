# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the layerwise STAGING ownership lease protocol.

Covers the D-side server branches (ACQUIRE / RELEASE / COPY with lease
expiry and owner checks) and the P-side acquire retry loop, driven over a
real zmq loopback so the wire protocol itself is exercised.
"""

import threading
import time
import types
import unittest
from unittest.mock import patch

import msgspec
import torch
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    KVCacheRecvingLayerThread,
    KVCacheSendingLayerThread,
    MooncakeAgentMetadata,
    STAGING_ACQUIRE_MSG,
    STAGING_BUSY,
    STAGING_COPY_MSG,
    STAGING_RELEASE_MSG,
)

import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector as mlc

_TEST_PORT_BASE = 21783
_port_counter = 0


def _next_port() -> int:
    global _port_counter
    _port_counter += 1
    return _TEST_PORT_BASE + _port_counter


def _make_recv_thread(
    staging_tensor: torch.Tensor | None,
    lease_timeout: float,
) -> KVCacheRecvingLayerThread:
    metadata = MooncakeAgentMetadata(te_rpc_port=1, layer_metadata={})
    ready_event = threading.Event()
    thread = KVCacheRecvingLayerThread(
        tp_rank=0,
        side_channel_port=_next_port(),
        tp_size=1,
        pd_head_ratio=1,
        local_engine_id="test-engine",
        metadata=metadata,
        ready_event=ready_event,
        staging_tensor=staging_tensor,
    )
    with patch.object(mlc, "get_transfer_timeout_value", return_value=lease_timeout), patch.object(
        mlc, "get_ip", return_value="127.0.0.1"
    ):
        # __init__ already consumed get_ip; re-point the host for run().
        thread.side_channel_host = "127.0.0.1"
        thread.start()
        assert ready_event.wait(timeout=5.0), "recv thread did not become ready"
    return thread


def _make_sender(timeout: float) -> KVCacheSendingLayerThread:
    sender = KVCacheSendingLayerThread.__new__(KVCacheSendingLayerThread)
    sender.timeout = timeout
    return sender


class _RecvClient:
    """Raw REQ client speaking the STAGING protocol to the recv thread."""

    def __init__(self, port: int, timeout: float = 3.0):
        self.path = mlc.make_zmq_path("tcp", "127.0.0.1", port)
        self.timeout = timeout
        self.encoder = msgspec.msgpack.Encoder()

    def _roundtrip(self, payload: tuple) -> bytes:
        with mlc.zmq_ctx(zmq.REQ, self.path) as sock:  # type: ignore
            sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
            sock.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
            mlc.ensure_zmq_send(sock, self.encoder.encode(payload), self.path)
            return sock.recv()

    def acquire(self, owner_id: str) -> bytes:
        return self._roundtrip((STAGING_ACQUIRE_MSG, owner_id))

    def release(self, owner_id: str) -> bytes:
        return self._roundtrip((STAGING_RELEASE_MSG, owner_id))

    def copy(self, req_ids: list[str], copy_plan: list[list[int]], owner_id: str) -> bytes:
        return self._roundtrip((STAGING_COPY_MSG, req_ids, copy_plan, owner_id))


class TestStagingOwnershipLease(unittest.TestCase):
    def setUp(self):
        self.staging = torch.zeros(4 * 1024 * 1024, dtype=torch.int8)
        self.staging.fill_(7)
        self.target = torch.zeros(4 * 1024 * 1024, dtype=torch.int8)
        self.port_holder = []

    def _start(self, lease_timeout: float = 600.0) -> _RecvClient:
        thread = _make_recv_thread(self.staging, lease_timeout)
        self.port_holder.append(thread)
        self.addCleanup(thread.join, 0)  # daemon thread, no loop exit; fire-and-forget
        return _RecvClient(thread.side_channel_port)

    def test_acquire_release_roundtrip(self):
        client = self._start()
        self.assertEqual(client.acquire("owner-A"), b"ACK")
        self.assertEqual(client.release("owner-A"), b"ACK")
        # Slot is free again: another owner acquires immediately.
        self.assertEqual(client.acquire("owner-B"), b"ACK")

    def test_second_owner_gets_busy_while_leased(self):
        client = self._start()
        self.assertEqual(client.acquire("owner-A"), b"ACK")
        self.assertEqual(client.acquire("owner-B"), STAGING_BUSY)
        # Re-acquire by the same owner must stay idempotent.
        self.assertEqual(client.acquire("owner-A"), b"ACK")

    def test_release_from_non_owner_is_rejected(self):
        client = self._start()
        self.assertEqual(client.acquire("owner-A"), b"ACK")
        self.assertEqual(client.release("owner-B"), b"FAILED")
        # owner-A still holds the lease afterwards.
        self.assertEqual(client.acquire("owner-B"), STAGING_BUSY)

    def test_stale_owner_expires_after_lease_timeout(self):
        client = self._start(lease_timeout=0.3)
        self.assertEqual(client.acquire("owner-A"), b"ACK")
        time.sleep(0.6)
        # Lease expired: owner-B acquires without BUSY.
        self.assertEqual(client.acquire("owner-B"), b"ACK")

    def test_copy_from_non_owner_is_rejected_and_fails_requests(self):
        client = self._start()
        thread = self.port_holder[-1]
        self.assertEqual(client.acquire("owner-A"), b"ACK")
        self.assertEqual(client.copy(["req-1"], [[0, 16, 0]], "owner-B"), b"FAILED")
        with thread.lock:
            self.assertIn("req-1", thread.failed_requests)
        # Lease is untouched by a rejected copy.
        self.assertEqual(client.acquire("owner-B"), STAGING_BUSY)

    def test_copy_owner_roundtrip_moves_bytes(self):
        client = self._start(lease_timeout=600.0)
        thread = self.port_holder[-1]
        self.assertEqual(client.acquire("owner-A"), b"ACK")

        self.staging[100:116] = torch.arange(100, 116, dtype=torch.int8)
        fake_target = self.target.view(torch.int8)

        def fake_resolve(dst, length):
            return (fake_target, 0)

        with patch.object(mlc, "get_world_group", return_value=types.SimpleNamespace(local_rank=0)), patch.object(
            torch.npu, "set_device", lambda device: None
        ), patch.object(torch.npu, "synchronize", lambda *a, **k: None), patch.object(
            mlc, "get_dsa_pd_mooncake_offloaded_kv_tensor", side_effect=fake_resolve
        ):
            self.assertEqual(client.copy(["req-2"], [[0, 16, 100]], "owner-A"), b"ACK")

        self.assertTrue(torch.equal(self.target[:16], torch.arange(100, 116, dtype=torch.int8)))
        self.assertEqual(self.target[16], 0)
        with thread.lock:
            self.assertNotIn("req-2", thread.failed_requests)

        # A successful copy refreshes the lease: a rival still gets BUSY.
        self.assertEqual(client.acquire("owner-B"), STAGING_BUSY)
        self.assertEqual(client.release("owner-A"), b"ACK")

    def test_copy_failure_clears_owner_and_fails_requests(self):
        client = self._start()
        thread = self.port_holder[-1]
        self.assertEqual(client.acquire("owner-A"), b"ACK")

        with patch.object(mlc, "get_world_group", return_value=types.SimpleNamespace(local_rank=0)), patch.object(
            torch.npu, "set_device", lambda device: None
        ), patch.object(
            mlc, "get_dsa_pd_mooncake_offloaded_kv_tensor", return_value=None
        ):
            self.assertEqual(client.copy(["req-3"], [[0, 16, 0]], "owner-A"), b"FAILED")

        with thread.lock:
            self.assertIn("req-3", thread.failed_requests)
        # Owner was cleared on failure: next acquire wins without BUSY.
        self.assertEqual(client.acquire("owner-B"), b"ACK")


class TestSenderAcquireRetry(unittest.TestCase):
    def setUp(self):
        self.staging = torch.zeros(1024, dtype=torch.int8)
        self.thread = _make_recv_thread(self.staging, 600.0)
        self.addCleanup(self.thread.join, 0)
        self.port = self.thread.side_channel_port
        self.client = _RecvClient(self.port)

    def test_acquire_spins_through_busy_until_release(self):
        sender = _make_sender(timeout=5.0)

        release_timer = threading.Timer(0.5, lambda: self.client.release("holder"))
        with patch.object(mlc, "get_ip", return_value="127.0.0.1"):
            self.assertEqual(self.client.acquire("holder"), b"ACK")
            release_timer.start()
            # owner "waiter" is blocked by "holder" until the timer fires.
            sender._acquire_staging_ownership("127.0.0.1", self.port, "waiter")
            release_timer.join(timeout=2.0)

        # Lease now belongs to the waiter.
        self.assertEqual(self.client.acquire("challenger"), STAGING_BUSY)
        self.assertEqual(self.client.release("waiter"), b"ACK")

    def test_acquire_times_out_when_never_released(self):
        sender = _make_sender(timeout=0.4)
        with patch.object(mlc, "get_ip", return_value="127.0.0.1"):
            self.assertEqual(self.client.acquire("holder"), b"ACK")
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                sender._acquire_staging_ownership("127.0.0.1", self.port, "waiter")
            self.assertLess(time.monotonic() - start, 5.0)
        self.assertEqual(self.client.release("holder"), b"ACK")


if __name__ == "__main__":
    unittest.main()
