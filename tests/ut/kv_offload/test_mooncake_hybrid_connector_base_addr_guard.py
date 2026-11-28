"""Regression tests for the Prefill/Decode KV base-address pairing guard.

issue #22659: ``MooncakeHybridConnector`` pairs the local and the peer's KV cache
base-address lists with ``zip``, which silently truncates to the shorter list.
When the two sides export a different number of tensors (for example speculative
decoding enabled on Decode only), the transfer either dropped tail layers or
bound addresses to the wrong KV cache group, without any error.

The guard rejects a count mismatch in either direction before the transfer, and
records the request on the receiving thread so the engine is told the receive
failed instead of serving the request with the KV cache it never got.

The Mooncake transfer engine is stubbed by ``tests/ut/conftest.py`` when no NPU
is present, and every case below injects its own per-test ``engine`` mock, so no
module-level override of ``sys.modules`` is needed here.
"""

import threading
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
    KVCacheRecvingThread,
    MooncakeConnector,
    MooncakeConnectorWorker,
)

LOCAL_ENGINE = "decode"
REMOTE_ENGINE = "prefill"
HANDSHAKE_PORT = 31002
LOCAL_PORT = 32000
ADDR_STEP = 0x1000
REQUEST_ID = "decode-req-0"


def _addrs(count: int, base: int) -> list[int]:
    return [base + index * ADDR_STEP for index in range(count)]


def _make_thread(remote_count: int, local_count: int, group_count: int = 1, prefill_pp_size: int = 1):
    """Build a KVCacheRecvingThread carrying only the state the transfer path reads."""
    thread = object.__new__(KVCacheRecvingThread)
    thread.hma_group_size = group_count
    thread.kv_cache_specs = [MagicMock() for _ in range(group_count)]
    thread.remote_metadata_lock = threading.Lock()
    thread.local_engine_id = LOCAL_ENGINE
    thread.local_handshake_port = LOCAL_PORT
    thread.remote_te_port = {REMOTE_ENGINE: {HANDSHAKE_PORT: 7777}}
    thread.kv_caches_base_addr = {
        LOCAL_ENGINE: {LOCAL_PORT: _addrs(local_count, 0x100000)},
        REMOTE_ENGINE: {HANDSHAKE_PORT: _addrs(remote_count, 0x400000)},
    }
    thread.block_len_per_addr = [16] * local_count
    thread.block_stride_per_addr = [32] * local_count
    thread.addr_group_idx = None
    thread._prefill_pp_size = prefill_pp_size
    thread.tp_rank = 0
    thread.kv_cache_config = types.SimpleNamespace(kv_cache_tensors=[MagicMock() for _ in range(local_count)])
    thread.engine = MagicMock()
    thread.engine.batch_transfer_sync_read.return_value = 0
    thread.failed_recv_requests = set()
    thread.failed_recv_requests_lock = threading.Lock()
    return thread


def _make_plain_thread(remote_count: int, local_count: int):
    """Thread state for the single-group ``_transfer_kv_cache`` path."""
    thread = _make_thread(remote_count, local_count)
    thread.pp_layer_indices = {0: (0, 1)}
    thread.kv_caches = {"layer0": [MagicMock(), MagicMock()]}
    thread.vllm_config = types.SimpleNamespace(speculative_config=None)
    thread.use_hybrid = False
    return thread


def _req_meta(group_count: int = 1, offset: int = 0) -> dict[str, Any]:
    return {
        "request_id": REQUEST_ID,
        "remote_request_id": "prefill-req-0",
        "remote_block_ids": [[1, 2] for _ in range(group_count)],
        "local_block_ids": [[3, 4] for _ in range(group_count)],
        "remote_engine_id": REMOTE_ENGINE,
        "remote_host": "192.0.2.1",
        "remote_handshake_port": HANDSHAKE_PORT,
        "offset": offset,
        "tp_num_need_pulls": 1,
        "all_task_done": True,
    }


def _transfer_count(thread) -> int:
    """Number of (src, dst, length) triples handed to the transfer engine."""
    assert thread.engine.batch_transfer_sync_read.call_count == 1
    src_list, dst_list, length_list = thread.engine.batch_transfer_sync_read.call_args.args[1:]
    assert len(src_list) == len(dst_list) == len(length_list)
    return len(src_list)


class TestBaseAddressCountGuard(unittest.TestCase):
    def test_short_peer_list_is_rejected_before_transfer(self):
        # Two peer tensors against three local ones: zip() would drop a layer.
        thread = _make_thread(remote_count=2, local_count=3)

        with self.assertRaisesRegex(ValueError, "base address count mismatch"):
            thread._transfer_kv_cache_all_groups(_req_meta())

        thread.engine.batch_transfer_sync_read.assert_not_called()

    def test_equal_lists_pair_by_position(self):
        thread = _make_thread(remote_count=2, local_count=2, group_count=2)

        thread._transfer_kv_cache_all_groups(_req_meta(group_count=2))

        # Two groups x two base addresses, unchanged from the pre-fix behaviour.
        self.assertEqual(_transfer_count(thread), 4)
        self.assertEqual(thread.get_and_clear_failed_recv_requests(), set())

    def test_extra_peer_entries_are_rejected(self):
        # The peer exporting more tensors is not harmless either: the pairing is
        # positional and the handshake carries no tensor identity, so the surplus
        # entries cannot be verified to belong after the local ones.
        thread = _make_thread(remote_count=3, local_count=2, group_count=2)

        with self.assertRaisesRegex(ValueError, "base address count mismatch"):
            thread._transfer_kv_cache_all_groups(_req_meta(group_count=2))

        thread.engine.batch_transfer_sync_read.assert_not_called()

    def test_pp_guard_compares_the_per_stage_slice(self):
        # Prefill runs PP=2 while Decode holds all eight tensors: each stage
        # exports four, and the guard must compare against the sliced list.
        for offset in (0, 1):
            with self.subTest(offset=offset):
                thread = _make_thread(remote_count=4, local_count=8, prefill_pp_size=2)

                thread._transfer_kv_cache_all_groups(_req_meta(offset=offset))

                self.assertEqual(_transfer_count(thread), 4)

    def test_pp_slice_longer_than_peer_is_rejected(self):
        # Ten local tensors with PP=2 mean five per stage; the peer stage
        # exports four, so the slice would pair five local entries with four
        # remote ones.
        thread = _make_thread(remote_count=4, local_count=10, prefill_pp_size=2)

        with self.assertRaisesRegex(ValueError, "base address count mismatch"):
            thread._transfer_kv_cache_all_groups(_req_meta(offset=1))

    def test_failed_transfer_is_reported_to_the_scheduler(self):
        thread = _make_thread(remote_count=2, local_count=3)

        with self.assertRaises(ValueError):
            thread._transfer_kv_cache_all_groups(_req_meta())

        # The engine must learn that this request was not received, otherwise
        # _handle_request() would still mark it done and Decode would read the
        # KV cache that was never written.
        self.assertEqual(thread.get_and_clear_failed_recv_requests(), {REQUEST_ID})
        self.assertEqual(thread.get_and_clear_failed_recv_requests(), set())

    def test_non_hybrid_mismatch_is_rejected_and_reported(self):
        thread = _make_plain_thread(remote_count=1, local_count=2)

        with self.assertRaisesRegex(ValueError, "base address count mismatch"):
            thread._transfer_kv_cache(_req_meta())

        thread.engine.batch_transfer_sync_read.assert_not_called()
        self.assertEqual(thread.get_and_clear_failed_recv_requests(), {REQUEST_ID})

    def test_non_hybrid_equal_lengths_still_transfer(self):
        thread = _make_plain_thread(remote_count=2, local_count=2)
        ascend_config = types.SimpleNamespace(enable_kv_nz=False, enable_transpose_kv_cache_by_block=False)

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.get_ascend_config",
            return_value=ascend_config,
        ):
            thread._transfer_kv_cache(_req_meta())

        self.assertGreater(_transfer_count(thread), 0)
        self.assertEqual(thread.get_and_clear_failed_recv_requests(), set())


class TestFailedRecvReporting(unittest.TestCase):
    def test_producer_worker_never_reports_receive_failures(self):
        worker = object.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_producer"
        worker.kv_recv_thread = None

        self.assertEqual(worker.get_and_clear_failed_recv_requests(), set())

    def test_connector_merges_failed_requests_into_transfer_results(self):
        connector = object.__new__(MooncakeConnector)
        connector.connector_worker = MagicMock()
        connector.connector_worker.get_finished.return_value = (set(), {REQUEST_ID})
        connector.connector_worker.get_and_clear_failed_recv_requests.return_value = {REQUEST_ID}

        results = connector.get_transfer_results(set())

        self.assertEqual(results.finished_recving, {REQUEST_ID})
        self.assertEqual(results.failed_recving, {REQUEST_ID})


if __name__ == "__main__":
    unittest.main()
