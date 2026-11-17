# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for host publication order and the device boundary it protects.

Load the actual fence module in isolation so these concurrency tests need no
NPU runtime and do not modify the process-wide connector's current gate.
"""

import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class TestAttentionEnqueueGate(unittest.TestCase):
    def setUp(self):
        self.trace = []
        self.device_boundary = threading.Event()
        self.synchronizing = threading.Event()
        trace = self.trace
        ready = self.device_boundary
        synchronizing = self.synchronizing

        class Event:
            def record(self, stream):
                trace.append(("record", stream))

            def synchronize(self):
                synchronizing.set()
                if not ready.wait(timeout=2):
                    raise TimeoutError("Test did not release the device boundary")
                trace.append("boundary_complete")

        source = Path(__file__).resolve().parents[4] / (
            "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py"
        )
        spec = importlib.util.spec_from_file_location("isolated_attention_fence", source)
        self.module = importlib.util.module_from_spec(spec)
        torch = SimpleNamespace(npu=SimpleNamespace(Event=Event, current_stream=lambda: "compute"))
        with patch.dict(sys.modules, {"torch": torch}):
            spec.loader.exec_module(self.module)
        self.gate = self.module.reset_attention_compute_start_gate()
        publish = self.gate.publish

        def trace_publication(event):
            if self.gate._event is None:
                self.trace.append("publish")
            publish(event)

        self.gate.publish = trace_publication
        self.gate.on_finish = lambda: self.trace.append("finish")

    def test_deferred_publication_uses_pre_attention_event(self):
        with self.module.attention_transfer_window(defer_transfer_start=True):
            self.assertEqual(self.trace, [("record", "compute")])
            self.assertIsNone(self.gate._event)
            self.trace.append("enqueue_attention")
        self.assertEqual(self.trace, [("record", "compute"), "enqueue_attention", "publish", "finish"])
        # Publishing did not synchronize the stream or wait for attention.
        self.assertFalse(self.synchronizing.is_set())

    def test_default_policy_remains_immediate(self):
        with self.module.attention_transfer_window():
            self.assertEqual(self.trace, [("record", "compute"), "publish"])
            self.trace.append("enqueue_attention")
        self.assertEqual(self.trace[-2:], ["enqueue_attention", "finish"])

    def test_backend_save_ready_callback_keeps_pre_attention_boundary(self):
        self.gate.on_start = lambda: self.trace.append("save_ready")
        with self.module.attention_transfer_window(defer_transfer_start=True):
            self.assertEqual(self.trace, [("record", "compute"), "publish", "save_ready"])
            self.trace.append("enqueue_attention")
        self.assertEqual(self.trace[-2:], ["enqueue_attention", "finish"])

    def test_waiter_needs_both_publication_and_device_boundary(self):
        finished = threading.Event()
        result = []

        def wait_for_transfer():
            result.append(self.gate.wait(timeout=2))
            finished.set()

        waiter = threading.Thread(target=wait_for_transfer)
        waiter.start()
        try:
            with self.module.attention_transfer_window(defer_transfer_start=True):
                self.assertFalse(self.synchronizing.is_set())
                self.assertFalse(finished.is_set())
            self.assertTrue(self.synchronizing.wait(timeout=2))
            self.assertFalse(finished.is_set())
            self.device_boundary.set()
            self.assertTrue(finished.wait(timeout=2))
            self.assertEqual(result, [True])
        finally:
            self.device_boundary.set()
            self.gate.cancel()
            waiter.join(timeout=3)
        self.assertFalse(waiter.is_alive())

    def test_submission_error_propagates_without_stranding_gate(self):
        with (
            self.assertRaisesRegex(RuntimeError, "operator submission"),
            self.module.attention_transfer_window(defer_transfer_start=True),
        ):
            raise RuntimeError("operator submission")
        self.assertIsNotNone(self.gate._event)
        self.assertEqual(self.trace[-2:], ["publish", "finish"])

    def test_first_publication_wins_and_callback_runs_once(self):
        first = Mock()
        later = Mock()
        self.gate.publish(first)
        self.gate.publish(later)
        self.assertTrue(self.gate.wait())
        first.synchronize.assert_called_once_with()
        later.synchronize.assert_not_called()
        self.assertEqual(self.trace, ["publish"])

    def test_timeout_without_publication_does_not_touch_device(self):
        self.assertFalse(self.gate.wait(timeout=0.001))
        self.assertFalse(self.synchronizing.is_set())

    def test_cancel_does_not_invoke_connector_callbacks(self):
        self.gate.on_start = Mock()
        callback = self.gate.on_start
        self.gate.cancel()
        self.assertEqual(self.trace, [("record", "compute"), "publish"])
        callback.assert_not_called()
        self.assertIsNotNone(self.gate._event)
        self.assertIsNone(self.gate.on_finish)

    def test_no_connector_gate_does_not_create_device_event(self):
        with (
            patch.object(self.module, "_attention_compute_start_gate", None),
            self.module.attention_transfer_window(defer_transfer_start=True),
        ):
            self.trace.append("enqueue_attention")
        self.assertEqual(self.trace, ["enqueue_attention"])


if __name__ == "__main__":
    unittest.main()
