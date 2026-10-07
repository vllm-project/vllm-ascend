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
#

import unittest
from unittest import mock
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear


def _cpu_unquantized_apply(layer, x, bias=None):
    """Stand in for AscendUnquantizedLinearMethod.apply on CPU.

    Production apply dispatches vllm::unquantized_gemm (PrivateUse1 / NPU only).
    """
    return F.linear(x, layer.weight, bias)


class TestAscendGateLinear(TestBase):
    def setUp(self):
        super().setUp()

        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 1
        self.mock_group.rank_in_group = 0

        self.patches = [
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
            patch(
                "vllm_ascend.ops.linear_op.get_tp_group",
                return_value=self.mock_group,
            ),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

        super().tearDown()

    def test_forward_keeps_router_logits_fp32(self):
        gate = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            out_dtype=torch.float32,
            prefix="test.gate",
        )

        self.assertEqual(gate.weight.dtype, torch.float32)
        self.assertEqual(gate.out_dtype, torch.float32)

        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        with patch.object(gate.quant_method, "apply", side_effect=_cpu_unquantized_apply):
            output, output_bias = gate(hidden_states)

        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.shape, (2, 4))
        self.assertIsNone(output_bias)


if __name__ == "__main__":
    unittest.main()
