# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Unit tests for KV transfer failure handling in ascend_store.

This module tests the record_failed_blocks function which handles KV transfer
failures by recording which blocks failed to load during the transfer process.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import torch

if not hasattr(torch, "npu"):
    torch.npu = types.SimpleNamespace(Event=type("Event", (), {}))  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import record_failed_blocks


class TestRecordFailedBlocks(unittest.TestCase):
    """Test cases for the record_failed_blocks function.

    The record_failed_blocks function takes a list of block IDs and their corresponding
    return codes from a KV transfer operation, and returns a set of block IDs that failed
    (i.e., those with non-zero return codes).
    """

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_all_blocks_succeed(self, mock_logger: MagicMock):
        """Test when all blocks are transferred successfully (all return codes are 0)."""
        block_ids: list[int] = [1, 2, 3, 4, 5]
        ret_codes: list[int] = [0, 0, 0, 0, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, set())
        self.assertEqual(len(result), 0)
        mock_logger.error.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_all_blocks_fail(self, mock_logger: MagicMock):
        """Test when all blocks fail to transfer (all return codes are non-zero)."""
        block_ids: list[int] = [1, 2, 3, 4, 5]
        ret_codes: list[int] = [1, 2, 3, 4, 5]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {1, 2, 3, 4, 5})
        self.assertEqual(len(result), 5)
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_partial_blocks_fail(self, mock_logger: MagicMock):
        """Test when some blocks fail and some succeed."""
        block_ids: list[int] = [1, 2, 3, 4, 5]
        ret_codes: list[int] = [0, 1, 0, 2, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {2, 4})
        self.assertEqual(len(result), 2)
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_empty_lists(self, mock_logger: MagicMock):
        """Test with empty block_ids and ret_codes."""
        block_ids: list[int] = []
        ret_codes: list[int] = []

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, set())
        mock_logger.error.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_single_block_succeed(self, mock_logger: MagicMock):
        """Test with a single block that succeeds."""
        block_ids: list[int] = [42]
        ret_codes: list[int] = [0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, set())
        mock_logger.error.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_single_block_fail(self, mock_logger: MagicMock):
        """Test with a single block that fails."""
        block_ids: list[int] = [42]
        ret_codes: list[int] = [1]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {42})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_negative_return_codes(self, mock_logger: MagicMock):
        """Test with negative return codes (error conditions)."""
        block_ids: list[int] = [1, 2, 3]
        ret_codes: list[int] = [0, -1, -2]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {2, 3})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_large_block_ids(self, mock_logger: MagicMock):
        """Test with large block ID values."""
        block_ids: list[int] = [1000000, 2000000, 3000000]
        ret_codes: list[int] = [0, 1, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {2000000})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_mixed_error_codes(self, mock_logger: MagicMock):
        """Test with various non-zero error codes."""
        block_ids: list[int] = [10, 20, 30, 40, 50]
        ret_codes: list[int] = [0, -1, 100, 0, 999]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {20, 30, 50})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_logs_failed_blocks(self, mock_logger: MagicMock):
        """Test that failed blocks are logged."""
        block_ids: list[int] = [1, 2, 3]
        ret_codes: list[int] = [0, 1, 2]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {2, 3})
        mock_logger.error.assert_called_once()
        log_msg, failed_blocks = mock_logger.error.call_args[0]
        self.assertIn("Failed to load blocks", log_msg)
        self.assertEqual(failed_blocks, {2, 3})

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_no_log_when_all_succeed(self, mock_logger: MagicMock):
        """Test that no error is logged when all blocks succeed."""
        block_ids: list[int] = [1, 2, 3]
        ret_codes: list[int] = [0, 0, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, set())
        mock_logger.error.assert_not_called()


class TestRecordFailedBlocksEdgeCases(unittest.TestCase):
    """Additional edge case tests for record_failed_blocks."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_duplicate_block_ids_all_fail(self, mock_logger: MagicMock):
        """Test with duplicate block IDs that all fail."""
        # Note: This tests the behavior with duplicates
        # The set will deduplicate, but all should be marked as failed
        block_ids: list[int] = [1, 1, 2, 2]
        ret_codes: list[int] = [1, 1, 2, 2]

        result = record_failed_blocks(block_ids, ret_codes)

        # Set deduplicates, so we get unique failed block IDs
        self.assertEqual(result, {1, 2})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_zero_block_id_with_failure(self, mock_logger: MagicMock):
        """Test with block ID 0 failing."""
        block_ids: list[int] = [0, 1, 2]
        ret_codes: list[int] = [1, 0, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {0})
        mock_logger.error.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger")
    def test_consecutive_failures(self, mock_logger: MagicMock):
        """Test with consecutive block failures."""
        block_ids: list[int] = [100, 101, 102, 103, 104]
        ret_codes: list[int] = [1, 1, 1, 0, 0]

        result = record_failed_blocks(block_ids, ret_codes)

        self.assertEqual(result, {100, 101, 102})
        mock_logger.error.assert_called_once()


class TestAscendStoreConnector(unittest.TestCase):
    """Regression tests for connector-level load failure reporting."""

    def test_get_block_ids_with_load_errors_forwards_to_worker(self):
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.connector_worker = MagicMock()
        connector.connector_worker.get_block_ids_with_load_errors.return_value = {3, 7}

        result = connector.get_block_ids_with_load_errors()

        self.assertEqual(result, {3, 7})
        connector.connector_worker.get_block_ids_with_load_errors.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
