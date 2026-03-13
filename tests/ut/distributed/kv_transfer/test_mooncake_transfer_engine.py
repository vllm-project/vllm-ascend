# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE


class TestGlobalTEReset(unittest.TestCase):
    def setUp(self):
        self.global_te = GlobalTE()

    def test_reset_noop_when_engine_is_none(self):
        self.global_te.is_register_buffer = True
        self.global_te.reset()
        self.assertIsNone(self.global_te.transfer_engine)
        self.assertFalse(self.global_te.is_register_buffer)

    def test_reset_clears_engine_and_register_flag(self):
        old_engine = MagicMock()
        self.global_te.transfer_engine = old_engine
        self.global_te.is_register_buffer = True

        self.global_te.reset()

        self.assertIsNone(self.global_te.transfer_engine)
        self.assertFalse(self.global_te.is_register_buffer)

    def test_reset_calls_first_available_closer(self):
        old_engine = MagicMock()
        old_engine.close = MagicMock()
        old_engine.shutdown = MagicMock()
        self.global_te.transfer_engine = old_engine

        self.global_te.reset()

        old_engine.close.assert_called_once()
        old_engine.shutdown.assert_not_called()

    def test_reset_suppresses_closer_exceptions(self):
        old_engine = MagicMock()
        old_engine.close = MagicMock(side_effect=RuntimeError("close failed"))
        self.global_te.transfer_engine = old_engine

        self.global_te.reset()

        self.assertIsNone(self.global_te.transfer_engine)
        old_engine.close.assert_called_once()
