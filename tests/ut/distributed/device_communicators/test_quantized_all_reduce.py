#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Unit tests for the quantized all-reduce wrapper (mocked, no NPU needed).

Covers the same-sign maximum-value overflow regression reported in review:
with ws=4 and all ranks holding x == amax, the headroom scale
(global_amax * ws / 127) makes round(1/scale) == 32 and an unclamped 4-rank
sum is 128, which overflows int8 (HCCL wraps modulo 256).

The wrapper must therefore use the exact-bound scale
global_amax / (127 // ws) instead: |x/scale| <= max_q by construction, so
rounding can never exceed max_q and ws * max_q <= 127 holds for any input.
This needs no clamp kernel at all (int8 clamp_/clamp/hardtanh are ~250x
slower than minimum/maximum on Ascend).

The wrapper is also built for non-TP groups (ep/mc2 = DP x PCP x TP). Those can
exceed the supported group size, so an oversized group must fall back to bf16
instead of raising: the review reported a 16-rank EP/MC2 group aborting start-up
while a supported TP=4 was configured.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.distributed.device_communicators.quantized_all_reduce import (
    QuantizedAllReduce,
)


def _make(
    world_size: int = 4,
    min_numel: int = 8,
    device_type: AscendDeviceType = AscendDeviceType.A2,
) -> QuantizedAllReduce:
    """Build a wrapper; raises on unvalidated device families only."""
    with patch(
        "vllm_ascend.distributed.device_communicators.quantized_all_reduce.get_ascend_device_type",
        return_value=device_type,
    ):
        group = MagicMock()
        with patch(
            "vllm_ascend.distributed.device_communicators.quantized_all_reduce.dist.get_world_size",
            return_value=world_size,
        ):
            return QuantizedAllReduce(group, min_numel=min_numel)


class TestQuantizedAllReduce(unittest.TestCase):
    def test_min_numel_from_config(self):
        wrapper = _make(min_numel=12345)
        self.assertEqual(wrapper.min_numel, 12345)

    def test_accepted_on_validated_device_families(self):
        for dt in (AscendDeviceType.A2, AscendDeviceType.A3):
            wrapper = _make(world_size=4, device_type=dt)
            self.assertEqual(wrapper.max_q, 31)

    def test_raises_on_unvalidated_device_families(self):
        for dt in (AscendDeviceType._310P, AscendDeviceType.A5):
            with self.assertRaisesRegex(RuntimeError, "only validated on"):
                _make(world_size=4, device_type=dt)

    def test_supported_flag_only_up_to_max_group_size(self):
        for ws in (2, 4, 8):
            wrapper = _make(world_size=ws)
            self.assertTrue(wrapper.supported)
            self.assertEqual(wrapper.max_q, 127 // ws)
        # EP/MC2 groups are DP x PCP x TP: larger than any valid TP degree.
        for ws in (9, 16, 32):
            wrapper = _make(world_size=ws)
            self.assertFalse(wrapper.supported)
            self.assertEqual(wrapper.max_q, 0)

    def test_oversized_group_falls_back_to_bf16_without_raising(self):
        # 16 ranks = the reported TP=4/DP=4 EP/MC2 group. Must not raise.
        wrapper = _make(world_size=16, min_numel=1)
        self.assertFalse(wrapper.supported)
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        with (
            patch("torch.distributed.all_reduce") as mock_ar,
            patch("torch_npu.npu_quantize", create=True) as mock_q,
            patch("torch_npu.npu_anti_quant", create=True) as mock_aq,
        ):
            out = wrapper.all_reduce(x)
            # exactly one bf16 all-reduce, no amax sync and no quantization
            mock_ar.assert_called_once()
            self.assertEqual(mock_ar.call_args.args[0].dtype, torch.bfloat16)
            mock_q.assert_not_called()
            mock_aq.assert_not_called()
            self.assertTrue(out is x)

    def test_oversized_group_warns_with_its_group_size(self):
        wrapper = _make(world_size=16, min_numel=1)
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        with (
            patch("vllm_ascend.distributed.device_communicators.quantized_all_reduce.logger") as mock_logger,
            patch("torch.distributed.all_reduce"),
            patch("torch_npu.npu_quantize", create=True),
            patch("torch_npu.npu_anti_quant", create=True),
        ):
            wrapper.all_reduce(x)
        mock_logger.warning_once.assert_called_once()
        self.assertEqual(mock_logger.warning_once.call_args.args[1], 16)

    def test_max_q_integer_safe_for_common_tp_sizes(self):
        for ws in (2, 4, 8):
            self.assertLessEqual(ws * (127 // ws), 127)

    def test_small_message_skips_quantization(self):
        wrapper = _make(min_numel=8)
        x = torch.randn(2, 3, dtype=torch.bfloat16)  # numel=6 < 8
        with (
            patch("torch.distributed.all_reduce") as mock_ar,
            patch("torch_npu.npu_quantize", create=True) as mock_q,
            patch("torch_npu.npu_anti_quant", create=True) as mock_aq,
        ):
            out = wrapper.all_reduce(x)
            # small messages fall back to a plain bf16 all-reduce
            mock_ar.assert_called_once()
            mock_q.assert_not_called()
            mock_aq.assert_not_called()
            self.assertTrue(out is x)

    def test_degenerate_shapes_skip_quantization(self):
        """1-D tensors and tiny last dims fall back to bf16 (no quantize)."""
        wrapper = _make(min_numel=8)
        cases = [
            torch.randn(64, dtype=torch.bfloat16),  # 1-D, numel=64 >= 8
            torch.randn(64, 32, dtype=torch.bfloat16),  # last dim 32 < 64
        ]
        for x in cases:
            with (
                self.subTest(shape=tuple(x.shape)),
                patch("torch.distributed.all_reduce") as mock_ar,
                patch("torch_npu.npu_quantize", create=True) as mock_q,
            ):
                out = wrapper.all_reduce(x)
                mock_ar.assert_called_once()
                mock_q.assert_not_called()
                self.assertTrue(out is x)

    def test_large_message_uses_quantized_path(self):
        wrapper = _make(min_numel=8)
        x = torch.randn(4, 64, dtype=torch.bfloat16)  # numel=256 >= 8, C=64

        q_fake = torch.zeros(4, 64, dtype=torch.int8)
        deq_fake = torch.randn_like(x)
        with (
            patch("torch.distributed.all_reduce") as mock_ar,
            patch("torch_npu.npu_quantize", create=True, return_value=q_fake) as mock_q,
            patch("torch_npu.npu_anti_quant", create=True, return_value=deq_fake) as mock_aq,
        ):
            out = wrapper.all_reduce(x)
            # two all-reduces happen: per-channel amax sync + int8 payload
            self.assertEqual(mock_ar.call_count, 2)
            mock_q.assert_called_once()
            mock_aq.assert_called_once()
            self.assertTrue(torch.equal(out, deq_fake))
            # out-of-place return (GroupCoordinator contract): input untouched
            self.assertFalse(torch.equal(x, deq_fake))
            # payload all-reduce must be int8
            second_call = mock_ar.call_args_list[1]
            self.assertEqual(second_call.args[0].dtype, torch.int8)

    def test_exact_bound_scale_never_wraps(self):
        """Same-sign worst case: every element equals the per-channel amax.

        With scale = amax / max_q and max_q = 127 // ws, x == amax quantizes
        to exactly max_q on every rank, so the reduced sum is ws * max_q:
        ws=2 -> 126, ws=4 -> 124, ws=8 -> 120. All <= 127, no clamp needed.
        """
        for ws in (2, 4, 8):
            wrapper = _make(world_size=ws)
            amax = torch.ones(8, dtype=torch.float32)
            scale = (amax / float(wrapper.max_q)).clamp_min(1e-8)
            x = torch.ones(ws, 8, dtype=torch.bfloat16) * amax[0]
            q = (x.float() / scale).round().to(torch.int8)
            self.assertLessEqual(q.abs().max().item(), wrapper.max_q)
            total = q.sum(dim=0).abs().max().item()
            self.assertLessEqual(total, 127)

    def test_wrapper_uses_exact_bound_scale_and_does_not_clamp(self):
        """The wrapper must use scale = amax / max_q and skip the clamp."""
        wrapper = _make(min_numel=8)
        x = torch.full((4, 64), 1.0, dtype=torch.bfloat16)  # x == amax, C=64
        captured: dict[str, list[torch.Tensor]] = {}

        def fake_ar(t, group=None, op=None):
            captured.setdefault("payload", []).append(t.clone())

        q_fake = torch.full((4, 64), wrapper.max_q, dtype=torch.int8)
        with (
            patch("torch.distributed.all_reduce", side_effect=fake_ar),
            patch("torch_npu.npu_quantize", create=True, return_value=q_fake) as mock_q,
            patch(
                "torch_npu.npu_anti_quant",
                create=True,
                return_value=torch.randn_like(x),
            ) as mock_aq,
        ):
            wrapper.all_reduce(x)
        # scale handed to npu_quantize must be global_amax / max_q = 1 / 31
        scale = mock_q.call_args.args[1]
        self.assertTrue(torch.allclose(scale, torch.full_like(scale, 1.0 / wrapper.max_q)))
        # [0] = amax fp32 sync, [1] = int8 payload; payload is exactly what
        # npu_quantize produced, i.e. nothing clamped it afterwards
        payload = captured["payload"][1]
        self.assertTrue(torch.equal(payload, q_fake))
        mock_aq.assert_called_once()


if __name__ == "__main__":
    unittest.main()
