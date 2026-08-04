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

import threading
import unittest
from unittest.mock import MagicMock, patch

# memcache_comm_fence imports torch and regex; both are real deps on the NPU
# box. torch.npu is stubbed per-test below so the gate logic runs without a
# device.
import vllm_ascend.memcache_comm_fence as fence


def _make_npu_event():
    event = MagicMock()
    event.record = MagicMock()
    event.synchronize = MagicMock()
    return event


class TestPerLayerAttentionGate(unittest.TestCase):
    def setUp(self):
        fence.reset_attention_compute_start_gates(3)

    def _record(self, layer_name):
        with patch.object(fence.torch.npu, "Event", side_effect=lambda: _make_npu_event()):
            fence.record_attention_compute_start(layer_name)

    def test_gate_resolves_layer_index_from_name(self):
        gate = fence.get_attention_compute_start_gate(2)
        self.assertIsNotNone(gate)
        self._record("model.layers.2.self_attn")
        self.assertIsNotNone(gate._event)

    def test_unrecorded_layer_gate_stays_closed(self):
        gate0 = fence.get_attention_compute_start_gate(0)
        gate1 = fence.get_attention_compute_start_gate(1)
        self._record("model.layers.1.self_attn")
        self.assertIsNone(gate0._event)
        self.assertIsNotNone(gate1._event)

    def test_out_of_range_layer_record_is_ignored(self):
        self._record("model.layers.99.self_attn")  # no gate; must not raise
        self._record("mtp.0.self_attn")  # unparseable -> ignored
        for i in range(3):
            self.assertIsNone(fence.get_attention_compute_start_gate(i)._event)

    def test_get_gate_out_of_range_returns_none(self):
        self.assertIsNone(fence.get_attention_compute_start_gate(7))
        self.assertIsNone(fence.get_attention_compute_start_gate(-1))

    def test_empty_name_uses_sequential_cursor(self):
        with patch.object(fence.torch.npu, "Event", side_effect=lambda: _make_npu_event()):
            fence.record_attention_compute_start("")
            fence.record_attention_compute_start("")
        self.assertIsNotNone(fence.get_attention_compute_start_gate(0)._event)
        self.assertIsNotNone(fence.get_attention_compute_start_gate(1)._event)
        self.assertIsNone(fence.get_attention_compute_start_gate(2)._event)

    def test_reset_recreates_fresh_gates(self):
        self._record("model.layers.0.self_attn")
        fence.reset_attention_compute_start_gates(3)
        for i in range(3):
            self.assertIsNone(fence.get_attention_compute_start_gate(i)._event)

    def test_wait_blocks_until_own_layer_recorded(self):
        # A load bound to layer 2 must not be released by layer 0's record.
        gate2 = fence.get_attention_compute_start_gate(2)
        released = threading.Event()

        def waiter():
            with patch.object(fence.torch.npu, "Event", side_effect=lambda: _make_npu_event()):
                gate2.wait(timeout=5)
            released.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        self._record("model.layers.0.self_attn")  # different layer
        self.assertFalse(released.wait(timeout=0.3))
        self._record("model.layers.2.self_attn")  # the bound layer
        self.assertTrue(released.wait(timeout=5))
        t.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
