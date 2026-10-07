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
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod


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

    def test_v031_gate_keywords_and_tensor_return(self):
        gate = AscendGateLinear(16, 4, quant_config=None, return_bias=False)
        gate.set_out_dtype(torch.bfloat16)
        with patch.object(gate.quant_method, "apply", side_effect=_cpu_unquantized_apply):
            output = gate(torch.randn(2, 16, dtype=torch.bfloat16))
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(output.dtype, torch.float32)

    def test_skip_bias_add_returns_unapplied_bias(self):
        gate = AscendGateLinear(16, 4, bias=True, skip_bias_add=True, quant_config=None)
        with torch.no_grad():
            gate.weight.fill_(1)
            gate.bias.fill_(3)
        with patch.object(gate.quant_method, "apply", side_effect=_cpu_unquantized_apply):
            output, bias = gate(torch.ones(2, 16, dtype=torch.bfloat16))
        torch.testing.assert_close(output, torch.full((2, 4), 16.0))
        self.assertIs(bias, gate.bias)

    def test_quant_config_selects_gate_method(self):
        quant_config = mock.MagicMock()
        quant_config.get_quant_method.return_value = AscendUnquantizedLinearMethod()
        gate = AscendGateLinear(16, 4, quant_config=quant_config, prefix="model.gate")
        quant_config.get_quant_method.assert_called_once_with(gate, prefix="model.gate")
        self.assertIs(gate.quant_config, quant_config)

    def test_routing_type_and_scaling_with_v031(self):
        router = AscendGroupedTopKRouter(
            top_k=2,
            global_num_experts=4,
            num_expert_group=None,
            topk_group=None,
            routed_scaling_factor=2.5,
        )
        self.assertEqual(router.routing_method_type, RoutingMethodType.RenormalizeNaive)
        weights, ids = router._compute_routing(torch.zeros(1, 16), torch.tensor([[0.0, 1.0, 2.0, 3.0]]), torch.int32)
        torch.testing.assert_close(weights.sum(-1), torch.tensor([2.5]))
        self.assertEqual(ids.tolist(), [[3, 2]])


if __name__ == "__main__":
    unittest.main()
