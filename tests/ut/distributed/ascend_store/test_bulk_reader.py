# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import unittest
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.executor import PoolReadRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.bulk_reader import BulkReader


class TestBulkReader(unittest.TestCase):
    def test_whole_request_completion_reports_attempt_and_blocks(self):
        executor = MagicMock()
        reader = BulkReader(executor)
        try:
            plan = [PoolReadRange("key", 8, 0, 1000, 8)]
            handle = reader.submit("request", "attempt", plan, frozenset({1, 2}))
            handle.future.result(timeout=5)
            results = reader.poll_results()
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].success)
            self.assertEqual(results[0].copied_bytes, 8)
            self.assertEqual(results[0].attempt_id, "attempt")
            self.assertEqual(results[0].block_ids, frozenset({1, 2}))
            executor.restore.assert_called_once_with(plan)
            self.assertEqual(reader.poll_results(), [])
        finally:
            reader.close()

    def test_cancelled_native_io_must_drain_before_completion_or_retry(self):
        entered, release = threading.Event(), threading.Event()
        executor = MagicMock()

        def restore(_):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test copy timed out")

        executor.restore.side_effect = restore
        reader = BulkReader(executor)
        try:
            handle = reader.submit("request", "attempt", [], frozenset({3}))
            self.assertTrue(entered.wait(timeout=5))
            reader.cancel("request")
            self.assertEqual(reader.poll_results(), [])
            with self.assertRaisesRegex(ValueError, "drain"):
                reader.submit("request", "retry", [], frozenset({3}))
            release.set()
            handle.future.result(timeout=5)
            result = reader.poll_results()[0]
            self.assertFalse(result.success)
            self.assertEqual(result.copied_bytes, 0)
            self.assertTrue(result.cancelled)
            reader.submit("request", "retry", [], frozenset({4})).future.result(timeout=5)
        finally:
            release.set()
            reader.close()

    def test_failed_copy_reports_failure_instead_of_partial_success(self):
        executor = MagicMock()
        executor.restore.side_effect = RuntimeError("second object failed")
        reader = BulkReader(executor)
        try:
            handle = reader.submit("request", "attempt", [], frozenset({7}))
            with self.assertRaisesRegex(RuntimeError, "second object"):
                handle.future.result(timeout=5)
            result = reader.poll_results()[0]
            self.assertFalse(result.success)
            self.assertEqual(result.block_ids, frozenset({7}))
            self.assertIn("second object", result.error)
            self.assertEqual(result.copied_bytes, 0)
        finally:
            reader.close()
