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

    def test_forward_bf16_router_gemm_fast_path(self):
        """bf16 weight + bf16 x + fp32 out -> tier 4 torch.mm(out_dtype=fp32)."""
        gate = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            out_dtype=torch.float32,
            prefix="test.gate",
        )
        gate.weight.data = gate.weight.data.to(torch.bfloat16)
        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)

        with patch("vllm_ascend.ops.fused_moe.gate_linear.torch.mm") as mm:
            mm.return_value = torch.randn(2, 4, dtype=torch.float32)
            output, output_bias = gate(hidden_states)

        self.assertIs(output, mm.return_value)
        self.assertIsNone(output_bias)
        mm.assert_called_once()
        args, kwargs = mm.call_args
        self.assertIs(args[0], hidden_states)
        self.assertEqual(args[1].data_ptr(), gate.weight.data_ptr())
        self.assertEqual(kwargs["out_dtype"], torch.float32)

    def test_none_out_dtype_defaults_to_fp32(self):
        """out_dtype=None (upstream dsv2/glm5next wiring) defaults to fp32:
        bf16 weights take tier 4 directly; fp32 weights cast x instead of
        raising a dtype mismatch."""
        gate = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            out_dtype=None,
            prefix="test.gate",
        )
        self.assertEqual(gate.out_dtype, torch.float32)
        gate.weight.data = gate.weight.data.to(torch.bfloat16)
        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)

        with patch("vllm_ascend.ops.fused_moe.gate_linear.torch.mm") as mm:
            mm.return_value = torch.randn(2, 4, dtype=torch.float32)
            output, output_bias = gate(hidden_states)
        mm.assert_called_once()
        self.assertIs(output, mm.return_value)
        self.assertIsNone(output_bias)

        # fp32 weight + bf16 x: tier 5 casts x to the weight dtype.
        gate2 = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            out_dtype=None,
            prefix="test.gate2",
        )
        with patch.object(gate2.quant_method, "apply", side_effect=_cpu_unquantized_apply):
            output2, _ = gate2(hidden_states)
        self.assertEqual(output2.dtype, torch.float32)

    def test_params_follow_model_dtype_unless_forced_fp32(self):
        """Without force_fp32_compute, params follow the model (default)
        dtype, mirroring upstream; with it, the weight is stored fp32."""
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            gate = AscendGateLinear(
                input_size=16,
                output_size=4,
                bias=False,
                out_dtype=torch.float32,
                prefix="test.gate",
            )
            self.assertEqual(gate.weight.dtype, torch.bfloat16)

            forced = AscendGateLinear(
                input_size=16,
                output_size=4,
                bias=False,
                out_dtype=torch.float32,
                force_fp32_compute=True,
                prefix="test.gate.forced",
            )
            self.assertEqual(forced.weight.dtype, torch.float32)
        finally:
            torch.set_default_dtype(default_dtype)

    def test_forward_fallback_casts_output_to_out_dtype(self):
        """Tier 5: x is cast to the weight dtype and the output to out_dtype."""
        gate = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            out_dtype=torch.bfloat16,
            prefix="test.gate",
        )
        # Weight stays fp32 (test default dtype); bf16 x -> fp32 compute,
        # then the output is cast to the requested bf16 out_dtype.
        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        with patch.object(gate.quant_method, "apply", side_effect=_cpu_unquantized_apply):
            output, output_bias = gate(hidden_states)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (2, 4))
        self.assertIsNone(output_bias)


if __name__ == "__main__":
    unittest.main()
