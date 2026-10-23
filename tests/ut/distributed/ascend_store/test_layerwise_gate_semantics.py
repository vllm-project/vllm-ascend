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
"""Layerwise attention-gate semantics.

Loads are gated on the compute stream reaching attention. If the thread is
asked to stop, the gate wait must return ``False`` so the caller drops its
transfer tasks instead of submitting work behind a stream that is shutting
down. Conversely an already-open gate must never be delayed by the shutdown
poll.
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    AttentionComputeStartGate,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVTransferThread,
)

# isort: on

POLL_TARGET = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer._TRANSFER_SHUTDOWN_POLL_S"


class TestLayerwiseGateWait(unittest.TestCase):
    def _make_thread(self) -> KVTransferThread:
        thread = KVTransferThread.__new__(KVTransferThread)
        thread._stopping = threading.Event()
        return thread

    def test_open_gate_resolves_immediately(self):
        thread = self._make_thread()
        gate = threading.Event()
        gate.set()

        start = time.monotonic()
        self.assertTrue(thread._wait_for_event(gate))
        self.assertLess(time.monotonic() - start, 0.5)

    def test_open_gate_still_aborts_when_stopping(self):
        """A gate that opens during shutdown must not resume the transfer."""
        thread = self._make_thread()
        thread._stopping.set()
        gate = threading.Event()
        gate.set()

        self.assertFalse(thread._wait_for_event(gate))

    def test_closed_gate_aborts_when_stopping_already_set(self):
        thread = self._make_thread()
        thread._stopping.set()

        start = time.monotonic()
        self.assertFalse(thread._wait_for_event(threading.Event()))
        self.assertLess(time.monotonic() - start, 0.5)

    @patch(POLL_TARGET, 0.02)
    def test_closed_gate_aborts_when_stopping_arrives_late(self):
        """Shutdown must interrupt an in-flight gate wait, not hang on it."""
        thread = self._make_thread()
        timer = threading.Timer(0.1, thread._stopping.set)
        timer.start()
        try:
            start = time.monotonic()
            self.assertFalse(thread._wait_for_event(threading.Event()))
            self.assertLess(time.monotonic() - start, 2.0)
        finally:
            timer.cancel()

    @patch(POLL_TARGET, 0.01)
    def test_failure_checker_runs_on_every_timeout(self):
        """Each poll tick must give the caller a chance to observe failures."""
        thread = self._make_thread()
        checker = MagicMock()

        def _stop_after_three() -> None:
            if checker.call_count >= 3:
                thread._stopping.set()

        checker.side_effect = _stop_after_three

        self.assertFalse(thread._wait_for_event(threading.Event(), failure_checker=checker))
        self.assertGreaterEqual(checker.call_count, 3)

    def test_failure_checker_skipped_when_gate_is_open(self):
        thread = self._make_thread()
        gate = threading.Event()
        gate.set()
        checker = MagicMock()

        self.assertTrue(thread._wait_for_event(gate, failure_checker=checker))
        checker.assert_not_called()

    @patch(POLL_TARGET, 0.02)
    def test_attention_gate_times_out_closed_then_opens(self):
        """A gate that is never recorded keeps reporting 'not ready'."""
        gate = AttentionComputeStartGate()

        self.assertFalse(gate.wait(timeout=0.02))

        gate._event = MagicMock()
        self.assertTrue(gate.wait(timeout=0.02))


if __name__ == "__main__":
    unittest.main()
