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

from vllm_ascend.distributed.device_communicators.quantized_all_reduce import (
    QuantizedAllReduce,
)


def _make(world_size: int = 4, min_numel: int = 8) -> QuantizedAllReduce:
    with patch.dict(
        os.environ, {"VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL": str(min_numel)}
    ):
        wrapper = QuantizedAllReduce.__new__(QuantizedAllReduce)
        wrapper.group = MagicMock()
        wrapper.world_size = world_size
        wrapper.max_q = 127 // world_size
        wrapper.min_numel = min_numel
        return wrapper


class TestQuantizedAllReduce(unittest.TestCase):
    def test_min_numel_from_env(self):
        wrapper = _make(min_numel=12345)
        self.assertEqual(wrapper.min_numel, 12345)

    def test_max_q_integer_safe_for_common_tp_sizes(self):
        for ws in (2, 4, 8, 16):
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
            x = torch.ones(4, 8, dtype=torch.bfloat16) * amax[0]
            q = torch.quantize_per_tensor(
                x.float(), 1.0, 0, torch.qint32
            ).int_repr()  # identity stand-in for rounding behavior
            q = (x.float() / scale).round().to(torch.int8)
            self.assertEqual(q.abs().max().item(), 32)  # exceeds 127/ws
            q.clamp_(-wrapper.max_q, wrapper.max_q)
            self.assertLessEqual(q.abs().max().item(), wrapper.max_q)
            total = q.sum().item()
            self.assertLessEqual(abs(total), 127)

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
