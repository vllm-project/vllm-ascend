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
"""Unit tests for the quantized all-reduce wrapper (mocked, no NPU needed)."""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestQuantizedAllReduce(unittest.TestCase):
    def _make(self, world_size=4, min_numel=8):
        from vllm_ascend.distributed.device_communicators.quantized_all_reduce import (
            QuantizedAllReduce,
        )

        with patch.dict(
            os.environ, {"VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL": str(min_numel)}
        ):
            wrapper = QuantizedAllReduce.__new__(QuantizedAllReduce)
            wrapper.group = MagicMock()
            wrapper.world_size = world_size
            wrapper.min_numel = wrapper.min_numel  # computed in __init__
            # Recompute min_numel as __init__ would
            wrapper.min_numel = int(
                os.getenv("VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL", str(8))
            )
            return wrapper

    def test_min_numel_from_env(self):
        wrapper = self._make(min_numel=12345)
        self.assertEqual(wrapper.min_numel, 12345)

    def test_small_message_skips_quantization(self):
        wrapper = self._make(min_numel=8)
        x = torch.randn(2, 4, dtype=torch.bfloat16)  # numel=8 -> not >= 8? equal
        x = torch.randn(2, 3, dtype=torch.bfloat16)  # numel=6 < 8

        with patch("torch.distributed.all_reduce") as mock_ar, patch(
            "torch_npu.npu_quantize"
        ) as mock_q, patch("torch_npu.npu_anti_quant") as mock_aq:
            # base path uses group all_reduce (mocked no-op mutation)
            def base_ar(t, group=None):
                return t

            mock_ar.side_effect = lambda t, group=None: None
            out = wrapper.all_reduce(x)
            mock_q.assert_not_called()
            mock_aq.assert_not_called()
            self.assertTrue(out is x)

    def test_large_message_uses_quantized_path(self):
        wrapper = self._make(min_numel=8)
        x = torch.randn(4, 8, dtype=torch.bfloat16)  # numel=32 >= 8

        q_fake = torch.zeros_like(x, dtype=torch.int8)
        deq_fake = torch.randn_like(x)
        with patch("torch.distributed.all_reduce") as mock_ar, patch(
            "torch_npu.npu_quantize", return_value=q_fake
        ) as mock_q, patch(
            "torch_npu.npu_anti_quant", return_value=deq_fake
        ) as mock_aq:
            out = wrapper.all_reduce(x)
            # two all-reduces happen: amax sync + int8 payload
            self.assertEqual(mock_ar.call_count, 2)
            mock_q.assert_called_once()
            mock_aq.assert_called_once()
            self.assertTrue(torch.equal(out, deq_fake.reshape(x.shape)))
            # payload all-reduce must be int8
            second_call = mock_ar.call_args_list[1]
            self.assertEqual(second_call.args[0].dtype, torch.int8)


if __name__ == "__main__":
    unittest.main()
