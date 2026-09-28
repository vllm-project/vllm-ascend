"""Grouped W4A8 expert projections with only packed low-rank factors resident."""

import torch
import torch_npu

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoELowRankLinear
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput


def factor_matmul(x, weight, scale, bias, groups, group_type, dynamic_scale=None):
    if group_type == 0:
        groups = torch.cat((groups[:1], groups[1:] - groups[:-1]))
        group_type = 1
    if dynamic_scale is None:
        quantized, dynamic_scale = torch_npu.npu_dynamic_quant(x)
    else:
        # A8W4 GMM may modify INT8 activations in place. Gate and up must not
        # consume the same dispatch buffer.
        quantized = x.clone()
    return torch_npu.npu_grouped_matmul(
        x=[quantized],
        weight=[weight],
        scale=[scale],
        bias=[bias],
        per_token_scale=[dynamic_scale],
        group_list=groups,
        split_item=2,
        group_type=0,
        group_list_type=group_type,
        output_dtype=torch.bfloat16,
    )[0]


def low_rank_linear(x, factors: MoELowRankLinear, groups, group_type=1, dynamic_scale=None):
    latent = factor_matmul(x, factors.right, factors.right_scale, factors.right_bias, groups, group_type, dynamic_scale)
    return factor_matmul(latent, factors.left, factors.left_scale, factors.left_bias, groups, group_type)


def low_rank_apply_mlp(inputs: MoEMlpComputeInput) -> tuple[torch.Tensor, torch.npu.Event]:
    activation = getattr(inputs.activation, "value", inputs.activation)
    if activation != "silu" or inputs.dynamic_eplb:
        raise ValueError("Low-rank MoE currently supports SiLU and static expert placement")
    if inputs.lora_context is not None or getattr(inputs.layer, "_ascend_moe_lora_context", None) is not None:
        raise ValueError("Low-rank MoE does not support LoRA adapters")
    assert inputs.weights.low_rank is not None
    gate_factors, up_factors, down_factors = inputs.weights.low_rank
    gate = low_rank_linear(
        inputs.hidden_states, gate_factors, inputs.group_list, inputs.group_list_type, inputs.dynamic_scale
    )
    up = low_rank_linear(
        inputs.hidden_states, up_factors, inputs.group_list, inputs.group_list_type, inputs.dynamic_scale
    )
    if inputs.swiglu_limit > 0:
        gate = gate.clamp(max=inputs.swiglu_limit)
        up = up.clamp(min=-inputs.swiglu_limit, max=inputs.swiglu_limit)
    hidden = torch.nn.functional.silu(gate) * up
    if inputs.topk_scales is not None:
        hidden = hidden * inputs.topk_scales
    before_gmm2_evt = torch.npu.current_stream().record_event()
    output = low_rank_linear(hidden, down_factors, inputs.group_list, inputs.group_list_type)
    return output, before_gmm2_evt
