# SPDX-License-Identifier: Apache-2.0
"""Request accounting retires after saves finish or their generation is released."""

import unittest
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class TestStateRequestRetirement(unittest.TestCase):
    def make_worker(self, state_reuse=True):
        # No thread or device work: use real sender accounting and worker entrypoints.
        sender = KVCacheStoreSendingThread(
            SimpleNamespace(),
            SimpleNamespace(group_block_len=[[16]]),
            block_size=16,
            tp_rank=0,
            enable_kv_event=True,
            state_store_layouts={} if state_reuse else None,
        )
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.enable_state_store_reuse = state_reuse
        worker.kv_send_thread = sender
        worker.kv_recv_thread = None
        worker._retired_store_req_ids = set()
        worker.tp_rank = 0
        return worker, sender

    def test_finished_request_cleans_progress_and_retry_tokens_after_sender_finishes(self):
        for state_reuse in (False, True):
            with self.subTest(state_reuse=state_reuse):
                worker, sender = self.make_worker(state_reuse)
                request = ReqMeta(req_id="r", store_job_id=1, save_start_token=16, token_ids=[21, 22])
                sender.add_stored_request(request)
                sender.record_saved_offset(request, 16)
                sender._remember_retry_token_ids(request)
                metadata = AscendConnectorMetadata(set())

                worker.get_transfer_results({"r"}, metadata)
                self.assertEqual(sender._saved_offsets, {"r": 16})
                self.assertEqual(sender._retry_token_ids, {"r": (16, [21, 22])})
                self.assertEqual(worker._retired_store_req_ids, {"r"})

                sender.finish_store_job(request)
                # The scheduler reports a finished request once, not on every poll.
                worker.get_transfer_results(set(), metadata)
                self.assertEqual(sender._saved_offsets, {})
                self.assertEqual(sender._retry_token_ids, {})
                self.assertEqual(worker._retired_store_req_ids, set())

    def test_released_state_request_cannot_be_repopulated_by_its_old_job(self):
        worker, sender = self.make_worker()
        old = ReqMeta(req_id="r", store_job_id=1, save_start_token=16, token_ids=[21, 22])
        sender.add_stored_request(old)
        sender.record_saved_offset(old, 16)
        sender._remember_retry_token_ids(old)
        worker.get_transfer_results({"r"}, AscendConnectorMetadata(set()))

        worker.handle_released_saves({"r"})
        sender.record_saved_offset(old, 64)
        sender._remember_retry_token_ids(old)
        self.assertEqual(sender.stored_requests, {})
        self.assertEqual(sender._saved_offsets, {})
        self.assertEqual(sender._retry_token_ids, {})
        self.assertEqual(worker._retired_store_req_ids, set())

        # Reusing the request ID must not authorize the retired job's writes.
        new = ReqMeta(req_id="r", store_job_id=2, save_start_token=8, token_ids=[31])
        sender.add_stored_request(new)
        sender.record_saved_offset(new, 8)
        sender._remember_retry_token_ids(new)
        sender.record_saved_offset(old, 64)
        sender._remember_retry_token_ids(old)
        sender.finish_store_job(old)
        self.assertEqual(sender.stored_requests, {"r": {2}})
        self.assertEqual(sender._saved_offsets, {"r": 8})
        self.assertEqual(sender._retry_token_ids, {"r": (8, [31])})
