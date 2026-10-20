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

Includes the same-sign maximum-value overflow regression reported in review:
with ws=4 and all ranks holding x == amax, round(1/scale) == 32 and an
unclamped 4-rank sum is 128, which overflows int8 (HCCL wraps modulo 256).
The clamp to +/-floor(127/ws) makes the worst-case sum ws*max_q <= 127.
"""

import os
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
    """Build a wrapper; raises RuntimeError on unvalidated configs."""
    with (
        patch.dict(
            os.environ, {"VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL": str(min_numel)}
        ),
        patch(
            "vllm_ascend.distributed.device_communicators.quantized_all_reduce.get_ascend_device_type",
            return_value=device_type,
        ),
    ):
        group = MagicMock()
        with patch(
            "vllm_ascend.distributed.device_communicators.quantized_all_reduce.dist.get_world_size",
            return_value=world_size,
        ):
            return QuantizedAllReduce(group)


class TestQuantizedAllReduce(unittest.TestCase):
    def test_min_numel_from_env(self):
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

    def test_raises_above_max_tp_size(self):
        # ws=16 -> max_q=7 (3 effective int8 levels): refuse instead of
        # silently degrading accuracy.
        with self.assertRaisesRegex(RuntimeError, "supports TP size up to"):
            _make(world_size=16, device_type=AscendDeviceType.A2)
        wrapper = _make(world_size=8, device_type=AscendDeviceType.A2)
        self.assertEqual(wrapper.max_q, 15)

    def test_max_q_integer_safe_for_common_tp_sizes(self):
        for ws in (2, 4, 8):
            self.assertLessEqual(ws * (127 // ws), 127)

    def test_small_message_skips_quantization(self):
        wrapper = _make(min_numel=8)
        x = torch.randn(2, 3, dtype=torch.bfloat16)  # numel=6 < 8
        with patch("torch.distributed.all_reduce") as mock_ar, patch(
            "torch_npu.npu_quantize"
        ) as mock_q, patch("torch_npu.npu_anti_quant") as mock_aq:
            out = wrapper.all_reduce(x)
            mock_q.assert_not_called()
            mock_aq.assert_not_called()
            self.assertTrue(out is x)

    def test_large_message_uses_quantized_path(self):
        wrapper = _make(min_numel=8)
        x = torch.randn(4, 8, dtype=torch.bfloat16)  # numel=32 >= 8

        q_fake = torch.zeros(4, 8, dtype=torch.int8)
        deq_fake = torch.randn_like(x)
        with patch("torch.distributed.all_reduce") as mock_ar, patch(
            "torch_npu.npu_quantize", return_value=q_fake
        ) as mock_q, patch(
            "torch_npu.npu_anti_quant", return_value=deq_fake
        ) as mock_aq:
            out = wrapper.all_reduce(x)
            # two all-reduces happen: per-channel amax sync + int8 payload
            self.assertEqual(mock_ar.call_count, 2)
            mock_q.assert_called_once()
            mock_aq.assert_called_once()
            self.assertTrue(torch.equal(out, deq_fake))
            # result is written in place to keep the mutation contract
            self.assertTrue(torch.equal(x, deq_fake))
            # payload all-reduce must be int8
            second_call = mock_ar.call_args_list[1]
            self.assertEqual(second_call.args[0].dtype, torch.int8)

    def test_clamp_bounds_quantized_values(self):
        """Same-sign worst case: every element equals the per-channel amax.

        With ws=4, scale = amax * 4/127 so x/scale == 31.75 and npu_quantize
        rounds to 32. Without the clamp the 4-rank sum is 128 which overflows
        int8. The clamp must bound every rank at max_q=31 (sum=124 <= 127).
        """
        for ws in (2, 4, 8):
            wrapper = _make(world_size=ws)
            amax = torch.ones(8, dtype=torch.float32)
            scale = amax * (ws / 127.0)
            x = torch.ones(ws, 8, dtype=torch.bfloat16) * amax[0]
            q = (x.float() / scale).round().to(torch.int8)
            # rounding always exceeds the integer-safe bound (ws=2: 63.5->64,
            # ws=4: 31.75->32, ws=8: 15.875->16), hence the clamp is required
            self.assertGreater(q.abs().max().item(), wrapper.max_q)
            q.clamp_(-wrapper.max_q, wrapper.max_q)
            self.assertLessEqual(q.abs().max().item(), wrapper.max_q)
            # x has 4 rows simulating 4 ranks; the reduced sum is per position
            total = q.sum(dim=0).abs().max().item()
            self.assertLessEqual(total, 127)

    def test_clamp_applied_in_wrapper_path(self):
        """The wrapper must clamp the int8 payload before the all-reduce."""
        wrapper = _make(min_numel=8)
        x = torch.full((4, 8), 1.0, dtype=torch.bfloat16)  # x == amax
        captured = {}

        def fake_ar(t, group=None, op=None):
            captured.setdefault("payload", []).append(t.clone())

        with patch("torch.distributed.all_reduce", side_effect=fake_ar), patch(
            "torch_npu.npu_quantize",
            return_value=torch.full((4, 8), 32, dtype=torch.int8),
        ), patch(
            "torch_npu.npu_anti_quant",
            return_value=torch.randn_like(x),
        ) as mock_aq:
            wrapper.all_reduce(x)
        payload = captured["payload"][1]  # [0]=amax fp32, [1]=int8 payload
        self.assertEqual(payload.abs().max().item(), wrapper.max_q)
        mock_aq.assert_called_once()


if __name__ == "__main__":
    unittest.main()
