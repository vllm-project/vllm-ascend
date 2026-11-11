# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Synchronous backend failures must reach the connector's failure channel."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class TestSyncLoadFailureReporting(unittest.TestCase):
    def _load_sync_result(self, block_ids_by_group, result, num_groups=None):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.load_async = False
        worker.tp_mismatch = False
        worker.tp_rank = 0
        worker.backend_name = "mooncake"
        worker.cache_transfer_granularity = 16
        worker.num_kv_cache_groups = num_groups or len(block_ids_by_group)
        worker.grouped_block_size = [16] * worker.num_kv_cache_groups
        worker.group_uses_align_state = [False] * worker.num_kv_cache_groups
        worker._invalid_block_ids = set()
        worker._failed_requests = set()
        worker._invalid_block_ids_lock = threading.Lock()
        worker._save_disabled = threading.Event()
        worker.kv_send_thread = worker.kv_recv_thread = None
        worker._record_kv_connector_operation = MagicMock()
        worker.token_database = MagicMock()
        block_ids = [group[0] for group in block_ids_by_group]
        worker.token_database.process_token_key_strings_with_block_ids.side_effect = [
            [(0, 16, f"key-{block_id}", "h0", block_id)] for block_id in block_ids
        ]
        worker.token_database.prepare_value.side_effect = [
            ([1000 + block_id * 160], [160], block_id) for block_id in block_ids
        ]
        worker.m_store = MagicMock()
        worker.m_store.get.return_value = result
        request = ReqMeta(
            req_id="load-failure",
            token_len_chunk=16,
            block_ids_by_group=block_ids_by_group,
            kv_cache_group_ids=list(range(len(block_ids_by_group))),
            block_hashes=["h0"],
            load_spec=LoadSpec(0, 16, can_load=True, token_len=16),
        )
        request.can_save = True
        worker.start_load_kv(SimpleNamespace(requests=[request]))
        worker.m_store.get.assert_called_once()
        return worker, request

    def test_hybrid_partial_failure_reports_request(self):
        for result in ([0, -704], [-704, 0], [-704, -704]):
            with self.subTest(result=result):
                worker, request = self._load_sync_result([[11], [21]], result)
                self.assertEqual(worker.get_failed_requests(), {request.req_id})
                self.assertEqual(worker.get_failed_requests(), set())
                self.assertEqual(worker.get_block_ids_with_load_errors(), set())
                self.assertFalse(request.can_save)
                self.assertTrue(worker._save_disabled.is_set())

    def test_hybrid_missing_result_reports_request(self):
        worker, request = self._load_sync_result([[11], [21]], None)
        self.assertEqual(worker.get_failed_requests(), {request.req_id})
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())
        self.assertFalse(request.can_save)

    def test_hybrid_success_does_not_report_failure(self):
        worker, request = self._load_sync_result([[11], [21]], [0, 0])
        self.assertEqual(worker.get_failed_requests(), set())
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())
        self.assertTrue(request.can_save)
        self.assertFalse(worker._save_disabled.is_set())

    def test_one_loaded_group_in_hybrid_layout_still_reports_request(self):
        worker, request = self._load_sync_result([[11]], [-704], num_groups=2)
        self.assertEqual(worker.get_failed_requests(), {request.req_id})
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())
        self.assertFalse(request.can_save)

    def test_single_group_block_error_reporting_is_preserved(self):
        for result, expected in (([-704], {11}), (None, {11}), ([0], set())):
            with self.subTest(result=result):
                worker, request = self._load_sync_result([[11]], result)
                self.assertEqual(worker.get_block_ids_with_load_errors(), expected)
                self.assertEqual(worker.get_failed_requests(), set())
                self.assertTrue(request.can_save)
