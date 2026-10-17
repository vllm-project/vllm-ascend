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
from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode
from vllm.sequence import IntermediateTensors

from tests.ut.base import TestBase
from vllm_ascend.compilation import graph_io
from vllm_ascend.compilation.graph_io import GraphIOBinder

MODE_NONE = CUDAGraphMode.NONE


def _metadata(slot_values, *, dtype=torch.int32):
    return {"layer.0.attn": SimpleNamespace(slot_mapping=torch.tensor(slot_values, dtype=dtype))}


class TestBindForwardInputs(TestBase):
    def setUp(self):
        super().setUp()
        # A stand-in for the runner's persistent intermediate buffer.
        self.carry_buffer = IntermediateTensors({"hidden_states": torch.zeros(8, 4)})
        self.synced: list[tuple[int, IntermediateTensors]] = []

        def _sync(num_tokens, tensors):
            self.synced.append((num_tokens, tensors))
            return self.carry_buffer

        self.binder = GraphIOBinder(intermediate_sync=_sync)

    def _capture(self, model_inputs):
        with self.binder.capture_context():
            return self.binder.bind_forward_inputs(model_inputs, CUDAGraphMode.FULL)

    def test_capture_records_anchors_and_passes_through(self):
        ids = torch.arange(8)
        positions = torch.arange(8)
        inputs = {"input_ids": ids, "positions": positions, "inputs_embeds": None}
        captured = self._capture(inputs)

        self.assertIs(captured, inputs)
        self.assertIs(self.binder._anchors["input.input_ids"], ids)
        self.assertIs(self.binder._anchors["input.positions"], positions)

    def test_replay_copies_into_anchors_and_returns_views(self):
        ids = torch.zeros(8, dtype=torch.int64)
        positions = torch.zeros(8)
        self._capture({"input_ids": ids, "positions": positions})

        batch_ids = torch.arange(3) + 100
        batch_positions = torch.arange(3) + 7
        bound = self.binder.bind_forward_inputs(
            {"input_ids": batch_ids, "positions": batch_positions},
            CUDAGraphMode.FULL,
        )

        # The graph reads the anchors, so the returned tensors must be views of
        # them, trimmed to the batch length.
        self.assertEqual(bound["input_ids"].data_ptr(), ids.data_ptr())
        self.assertEqual(bound["input_ids"].numel(), 3)
        self.assertEqual(ids[:3].tolist(), batch_ids.tolist())
        self.assertEqual(bound["positions"].data_ptr(), positions.data_ptr())
        self.assertEqual(positions[:3].tolist(), batch_positions.tolist())
        # Padding rows stay untouched.
        self.assertEqual(ids[3:].tolist(), [0] * 5)

    def test_same_pointer_is_not_copied(self):
        ids = torch.zeros(4, dtype=torch.int64)
        self._capture({"input_ids": ids})

        bound = self.binder.bind_forward_inputs({"input_ids": ids}, CUDAGraphMode.FULL)
        self.assertIs(bound["input_ids"], ids)

    def test_none_mode_leaves_inputs_untouched(self):
        ids = torch.zeros(8, dtype=torch.int64)
        self._capture({"input_ids": ids})

        batch_ids = torch.arange(2) + 5
        inputs = {"input_ids": batch_ids}
        bound = self.binder.bind_forward_inputs(inputs, MODE_NONE)
        self.assertIs(bound, inputs)
        self.assertEqual(batch_ids.tolist(), [5, 6])  # anchor untouched

    def test_mrope_positions_align_on_last_dim(self):
        positions = torch.zeros(3, 8)
        self._capture({"positions": positions})

        batch_positions = torch.full((3, 2), 9)
        bound = self.binder.bind_forward_inputs({"positions": batch_positions}, CUDAGraphMode.FULL)

        self.assertEqual(bound["positions"].shape, (3, 2))
        self.assertEqual(bound["positions"].data_ptr(), positions.data_ptr())
        self.assertEqual(positions[:, :2].tolist(), [[9, 9]] * 3)

    def test_inputs_embeds_align_on_token_dim(self):
        embeds = torch.zeros(8, 4)
        self._capture({"inputs_embeds": embeds})

        batch_embeds = torch.ones(2, 4)
        bound = self.binder.bind_forward_inputs({"inputs_embeds": batch_embeds}, CUDAGraphMode.FULL)

        self.assertEqual(bound["inputs_embeds"].data_ptr(), embeds.data_ptr())
        self.assertEqual(bound["inputs_embeds"].shape, (2, 4))
        self.assertEqual(embeds[:2].tolist(), [[1.0] * 4] * 2)

    def test_wider_capture_extends_anchor(self):
        buffer = torch.zeros(16, dtype=torch.int64)
        self._capture({"input_ids": buffer[:4]})
        self._capture({"input_ids": buffer})
        self.assertIs(self.binder._anchors["input.input_ids"], buffer)

    def test_moved_anchor_is_not_rebound(self):
        first = torch.zeros(4)
        self._capture({"positions": first})
        self._capture({"positions": torch.zeros(4)})
        self.assertIs(self.binder._anchors["input.positions"], first)

    def test_unbindable_critical_input_raises(self):
        self._capture({"input_ids": torch.zeros(4, 8, dtype=torch.int64)})
        with self.assertRaises(RuntimeError):
            self.binder.bind_forward_inputs(
                {"input_ids": torch.zeros(2, 3, dtype=torch.int64)},
                CUDAGraphMode.FULL,
            )

    def test_unbindable_kwarg_warns_and_passes_through(self):
        self._capture({"lookback": torch.zeros(4, 2)})
        batch = torch.zeros(2, 3)
        bound = self.binder.bind_forward_inputs({"lookback": batch}, CUDAGraphMode.FULL)
        self.assertIs(bound["lookback"], batch)


class TestIntermediateBinding(TestBase):
    def setUp(self):
        super().setUp()
        self.carry_buffer = IntermediateTensors({"hidden_states": torch.zeros(8, 4)})
        self.binder = GraphIOBinder(intermediate_sync=lambda num_tokens, tensors: self.carry_buffer)

    def test_capture_records_intermediate_anchors(self):
        carry = IntermediateTensors({"hidden_states": torch.zeros(8, 4)})
        with self.binder.capture_context():
            self.binder.bind_forward_inputs({"intermediate_tensors": carry}, CUDAGraphMode.FULL)
        self.assertIn("intermediate.hidden_states", self.binder._anchors)

    def test_replay_binds_intermediates(self):
        carry = torch.zeros(8, 4)
        with self.binder.capture_context():
            self.binder.bind_forward_inputs(
                {"intermediate_tensors": IntermediateTensors({"hidden_states": carry})},
                CUDAGraphMode.FULL,
            )

        batch_carry = torch.ones(2, 4)
        bound = self.binder.bind_forward_inputs(
            {"intermediate_tensors": IntermediateTensors({"hidden_states": batch_carry})},
            CUDAGraphMode.FULL,
        )
        self.assertEqual(bound["intermediate_tensors"]["hidden_states"].data_ptr(), carry.data_ptr())
        self.assertEqual(carry[:2].tolist(), [[1.0] * 4] * 2)

    def test_pin_intermediates_uses_runner_buffer(self):
        carry = IntermediateTensors({"hidden_states": torch.ones(4, 4)})
        result = self.binder.pin_intermediates(4, carry)
        self.assertIs(result, self.carry_buffer)

    def test_snapshot_is_a_private_copy(self):
        carry = IntermediateTensors({"hidden_states": torch.ones(4, 4)})
        copy = GraphIOBinder.snapshot(carry)
        self.assertNotEqual(copy["hidden_states"].data_ptr(), carry["hidden_states"].data_ptr())
        carry["hidden_states"].zero_()
        self.assertEqual(copy["hidden_states"].sum().item(), 16.0)


class TestPinMetadata(TestBase):
    def setUp(self):
        super().setUp()
        self.binder = GraphIOBinder()

    def test_records_and_pins_for_full(self):
        captured = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        self.binder.record_metadata_anchors({"layer.0.attn": SimpleNamespace(slot_mapping=captured)})

        batch = torch.tensor([9, 8, 7], dtype=torch.int32).clone()
        self.binder.pin_metadata({"layer.0.attn": SimpleNamespace(slot_mapping=batch)}, CUDAGraphMode.FULL)

        self.assertEqual(captured[:3].tolist(), [9, 8, 7])
        self.assertEqual(captured[3].item(), 4)

    def test_piecewise_does_not_touch_metadata(self):
        captured = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        self.binder.record_metadata_anchors({"layer.0.attn": SimpleNamespace(slot_mapping=captured)})

        batch = torch.tensor([9, 8, 7], dtype=torch.int32)
        self.binder.pin_metadata({"layer.0.attn": SimpleNamespace(slot_mapping=batch)}, CUDAGraphMode.PIECEWISE)
        self.assertEqual(captured.tolist(), [1, 2, 3, 4])

    def test_aliased_clones_are_copied_once(self):
        captured = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        metadata = {
            "layer.0.attn": SimpleNamespace(slot_mapping=captured),
            "layer.1.attn": SimpleNamespace(slot_mapping=captured),
        }
        self.binder.record_metadata_anchors(metadata)

        shared = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
        self.binder.pin_metadata(
            {
                "layer.0.attn": SimpleNamespace(slot_mapping=shared),
                "layer.1.attn": SimpleNamespace(slot_mapping=shared),
            },
            CUDAGraphMode.FULL,
        )
        self.assertEqual(captured.tolist(), [5, 6, 7, 8])

    def test_unknown_layer_is_ignored(self):
        captured = torch.tensor([1, 2, 3], dtype=torch.int32)
        self.binder.record_metadata_anchors({"layer.0.attn": SimpleNamespace(slot_mapping=captured)})

        other = torch.tensor([7], dtype=torch.int32)
        self.binder.pin_metadata({"layer.9.attn": SimpleNamespace(slot_mapping=other)}, CUDAGraphMode.FULL)
        self.assertEqual(captured.tolist(), [1, 2, 3])


class TestDirectReadVerification(TestBase):
    def setUp(self):
        super().setUp()
        self.anchor = torch.zeros(8, dtype=torch.int32)
        self.binder = GraphIOBinder()
        self.binder.record_metadata_anchors({"layer.0.attn": SimpleNamespace(slot_mapping=self.anchor)})

    def test_view_of_anchor_is_accepted(self):
        with self.binder.capture_context():
            graph_io.note_graph_direct_read("slot_mapping", self.anchor[:4])

    def test_foreign_tensor_warns_without_strict(self):
        with self.binder.capture_context():
            graph_io.note_graph_direct_read("slot_mapping", torch.zeros(4, dtype=torch.int32))
        self.assertEqual(len(self.binder._warned), 1)

    def test_foreign_tensor_raises_with_strict(self):
        self.binder._strict = True
        with self.binder.capture_context(), self.assertRaises(RuntimeError):
            graph_io.note_graph_direct_read("slot_mapping", torch.zeros(4, dtype=torch.int32))

    def test_capture_flag_only_set_inside_context(self):
        self.assertFalse(graph_io.graph_io_capture_active())
        with self.binder.capture_context():
            self.assertTrue(graph_io.graph_io_capture_active())
        self.assertFalse(graph_io.graph_io_capture_active())
