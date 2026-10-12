# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU regression for W8A8 GMM1 weight-scale reuse."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("tokens", [1, 32])
@pytest.mark.parametrize("enable_eplb", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
def test_w8a8_gmm1_bf16_scale_reuse(tokens, enable_eplb, use_graph):
    torch.manual_seed(17)
    experts, hidden, intermediate = 4, 128, 64
    scale = (torch.rand(experts, intermediate, device="npu") * 0.02 + 0.01).bfloat16()
    layer = SimpleNamespace(
        w13_weight=torch_npu.npu_format_cast(
            torch.randint(-8, 9, (experts, hidden, intermediate), dtype=torch.int8, device="npu"), 29
        ),
        w13_weight_scale=scale,
        w13_weight_scale_fp32=scale.float(),
        w2_weight=None,
        w2_weight_scale=torch.ones(experts, hidden, device="npu", dtype=torch.bfloat16),
    )
    method = AscendW8A8DynamicFusedMoEMethod.__new__(AscendW8A8DynamicFusedMoEMethod)
    method.use_expert_weight_list = False
    method.enable_eplb = enable_eplb
    x = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
    groups = torch.tensor([tokens // experts] * (experts - 1) + [tokens - 3 * (tokens // experts)], device="npu")

    def run():
        return method.apply_gmm1(
            MoEMlpComputeInput(
                hidden_states=x.clone(),
                group_list=groups,
                group_list_type=1,
                dynamic_scale=None,
                topk_scales=None,
                weights=MoEWeights(w1=None, w2=None),
                quant=MoEQuantParams(quant_type=QuantType.W8A8),
                fusion=False,
                layer=layer,
            )
        )

    def reference():
        quantized, per_token = torch_npu.npu_dynamic_quant(x)
        return torch_npu.npu_grouped_matmul(
            x=[quantized],
            weight=[layer.w13_weight],
            scale=[layer.w13_weight_scale_fp32.bfloat16()],
            per_token_scale=[per_token],
            split_item=2,
            group_type=0,
            group_list=groups,
            group_list_type=1,
            output_dtype=torch.bfloat16,
        )[0]

    for _ in range(3):
        run()
    torch.npu.synchronize()
    if use_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()

    for _ in range(2):
        if use_graph:
            graph.replay()
        else:
            actual = run()
        torch.testing.assert_close(actual.cpu(), reference().cpu(), rtol=0, atol=0)
        # EPLB migrates the FP32 copy alone; ordinary reload updates both.
        layer.w13_weight_scale_fp32.mul_(2)
        if not enable_eplb:
            layer.w13_weight_scale.mul_(2)
