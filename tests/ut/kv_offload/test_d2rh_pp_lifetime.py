# SPDX-License-Identifier: Apache-2.0
"""Host staging must remain pinned until every PP H2D pull completes."""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import KVCacheRecvingThread as Base
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import KVCacheTaskTracker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_d2rh_connector import KVCacheRecvingThread as Receiver


def handle_request(self, meta):
    try:
        if meta.get("fail"):
            raise RuntimeError("injected transfer failure")
    finally:
        if self._mark_request_task_done(meta["request_id"], meta["all_task_done"]):
            self.completed.append(meta["request_id"])


def receiver():
    obj = Receiver.__new__(Receiver)
    obj.request_task_counts = defaultdict(int)
    obj.request_task_counts_lock = threading.Lock()
    obj.finished_request_markers = set()
    obj._h2d_remote_request_ids = {}
    obj.freed = []
    obj.completed = []
    obj.block_map = {(0, 1, 0): 7}
    obj.remote_local_block_map = {"D-request": obj.block_map, "P-request": obj.block_map}
    obj.cpu_kvcache_manager = SimpleNamespace(free_block_map=obj.freed.append)
    return obj


def shards():
    return [dict(request_id="D-request", remote_request_id="P-request", all_task_done=last) for last in [False, True]]


class TestH2DLifetime(unittest.TestCase):
    def setUp(self):
        fake = patch.object(Base, "_handle_request", handle_request)
        fake.start()
        self.addCleanup(fake.stop)

    def test_last_submitted_finishes_first_must_keep_host_pinned(self):
        obj = receiver()
        first, last = shards()
        for item in [first, last]:
            obj._mark_request_task_submitted(item)
        obj._handle_request(last)
        self.assertEqual(obj.request_task_counts["D-request"], 1)
        self.assertEqual(obj.freed, [], "Host freed while another PP pull is pending")
        self.assertIn("P-request", obj.remote_local_block_map)
        obj._handle_request(first)
        self.assertEqual(obj.freed, [obj.block_map])
        self.assertEqual(obj.remote_local_block_map, {})
        self.assertEqual(obj.completed, ["D-request"])

    def test_in_order_releases_once(self):
        obj = receiver()
        items = shards()
        for item in items:
            obj._mark_request_task_submitted(item)
        obj._handle_request(items[0])
        self.assertEqual(obj.freed, [])
        obj._handle_request(items[1])
        self.assertEqual(obj.freed, [obj.block_map])

    def test_pp1_release(self):
        obj = receiver()
        item = shards()[-1]
        obj._mark_request_task_submitted(item)
        obj._handle_request(item)
        self.assertEqual(obj.freed, [obj.block_map])

    def test_early_first_completion_before_last_submission(self):
        obj = receiver()
        first, last = shards()
        obj._mark_request_task_submitted(first)
        obj._handle_request(first)
        self.assertEqual(obj.freed, [])
        obj._mark_request_task_submitted(last)
        obj._handle_request(last)
        self.assertEqual(obj.freed, [obj.block_map])

    def test_failed_last_pull_keeps_other_pull_pinned(self):
        obj = receiver()
        first, last = shards()
        last["fail"] = True
        for item in [first, last]:
            obj._mark_request_task_submitted(item)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            obj._handle_request(last)
        self.assertEqual(obj.freed, [])
        obj._handle_request(first)
        self.assertEqual(obj.freed, [obj.block_map])


class TestH2DCompletionOwnership(unittest.TestCase):
    """Exercise the inherited handler without replacing its completion logic."""

    @staticmethod
    def make_receiver():
        obj = receiver()
        obj.task_tracker = KVCacheTaskTracker()
        obj.task_tracker.add_req_to_process("D-request")
        obj.failed_recv_requests = set()
        obj.failed_recv_requests_lock = threading.Lock()
        obj.invalid_block_ids = set()
        obj.pending_reformat = {}
        obj.pending_reformat_lock = threading.Lock()
        obj.proc_not_transfer_request = {}
        obj.proc_not_transfer_request_lock = threading.Lock()
        obj.side_channel_port = obj.local_handshake_port = 47010
        obj.request_queue = MagicMock()
        obj._transfer_kv_cache_all_groups = MagicMock()
        obj._reformat_pending_kv_caches = MagicMock()
        return obj

    def run_completion(self, failure=None, reverse=False):
        obj = self.make_receiver()
        items = shards() if reverse else [shards()[-1]]
        for item in items:
            item.update(
                remote_host="P-host",
                remote_handshake_port=47010,
                remote_port_send_num={47010: {"num": 1, "host": "P-host"}},
                local_block_ids=([7],),
            )
            obj._mark_request_task_submitted(item)
        if failure == "transfer":
            obj._transfer_kv_cache_all_groups.side_effect = RuntimeError("injected transfer failure")
        elif failure == "reformat":
            obj._reformat_pending_kv_caches.side_effect = RuntimeError("injected reformat failure")
        with patch.object(Base, "_send_done_recv_signal") as notify:
            for index, item in enumerate(reversed(items)):
                obj._handle_request(item)
                if index < len(items) - 1:
                    self.assertEqual(obj.freed, [])
                    self.assertEqual(obj.task_tracker.finished_requests, set())
            notify.assert_not_called()
        self.assertEqual(obj.task_tracker.finished_requests, {"D-request"})
        self.assertEqual(obj.freed, [obj.block_map])
        self.assertEqual(obj.remote_local_block_map, {})
        self.assertEqual(obj._h2d_remote_request_ids, {})
        self.assertEqual(obj.request_queue.task_done.call_count, len(items))
        self.assertEqual(obj.invalid_block_ids, {7} if failure else set())

    def test_success_does_not_notify_prefill_again(self):
        self.run_completion()

    def test_transfer_failure_still_completes_local_cleanup(self):
        self.run_completion(failure="transfer")

    def test_reformat_failure_still_completes_local_cleanup(self):
        self.run_completion(failure="reformat")

    def test_out_of_order_shards_only_release_host_after_all_reads(self):
        self.run_completion(reverse=True)

    def test_failed_shard_does_not_release_host_while_another_read_is_pending(self):
        self.run_completion(failure="transfer", reverse=True)

    def test_first_hop_still_sends_prefill_completion(self):
        obj = d2rh.D2RHThread.__new__(d2rh.D2RHThread)
        sock = MagicMock()
        obj._get_remote_socket = MagicMock(return_value=sock)
        obj._return_remote_socket = MagicMock()
        obj._recv_from_socket = MagicMock(return_value=b"ACK")
        obj.encoder = MagicMock()
        with patch.object(d2rh, "ensure_zmq_send") as send:
            obj._send_done_recv_signal("P-request", "P-host", 47010)
        obj.encoder.encode.assert_called_once_with((d2rh.DONE_RECVING_MSG, "P-request", {}))
        send.assert_called_once()
        obj._return_remote_socket.assert_called_once_with(sock, "P-host", 47010)

    def test_non_pulling_cp_peer_still_receives_cleanup(self):
        obj = self.make_receiver()
        ports = {47010: {"num": 1, "host": "P-host"}, 47011: {"num": 0, "host": "P-other"}}
        with patch.object(Base, "_send_done_recv_signal") as notify:
            obj._send_done_signal_to_free_remote_port("P-request", "P-host", ports)
            obj._send_done_recv_signal("P-request", "P-host", 47010, ports)
        notify.assert_called_once_with("P-request", "P-other", 47011, ports)

    def test_single_peer_without_port_counts_does_not_notify_again(self):
        obj = self.make_receiver()
        with patch.object(Base, "_send_done_recv_signal") as notify:
            obj._send_done_recv_signal("P-request", "P-host", 47010, None)
            obj._send_done_recv_signal("P-request", "P-host", 47010, {})
        notify.assert_not_called()
