# SPDX-License-Identifier: Apache-2.0
"""Home GEMM1 overlaps remote weight prefetch; Slot GEMM1 joins before use."""
from __future__ import annotations

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import dispose_tensor


def _one_tensor(value, name):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Home GEMM requires one {name} tensor")
        return value[0]
    return value


def _bf16_activation(gate_up, activation, limit):
    if activation == MoEActivation.SWIGLUOAI:
        return AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up)
    if activation == MoEActivation.SWIGLUSTEP:
        return AscendSwigluStepAndMul.swiglustep_forward(gate_up, limit=7.0)
    if activation in (MoEActivation.GELU, MoEActivation.GELU_TANH):
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.gelu(
            gate, approximate="tanh" if activation == MoEActivation.GELU_TANH else "none"
        ) * up
    if limit > 0:
        gate, up = gate_up.chunk(2, dim=-1)
        gate.clamp_(max=limit)
        up.clamp_(min=-limit, max=limit)
    return torch_npu.npu_swiglu(gate_up)


def home_gemm_apply_mlp(
    *, mlp_compute_input: MoEMlpComputeInput, home_experts: int,
    intermediate_size: int, range_kernel, prefetch_ready=None,
):
    """Only GEMM1 is split. Both launches write one original-order output.

    No token count is read on the host. Input quantization is reused and its
    storage stays alive until both GEMM1 launches have been enqueued.
    """
    mci = mlp_compute_input
    if mci.activation not in (
        MoEActivation.SILU, MoEActivation.SWIGLUOAI, MoEActivation.SWIGLUSTEP,
        MoEActivation.GELU, MoEActivation.GELU_TANH,
    ):
        raise NotImplementedError(f"Home GEMM activation is not adapted: {mci.activation}")
    if mci.group_list_type != 1:
        raise ValueError("Home GEMM requires per-expert counts, not prefix sums")
    if mci.lora_context is not None:
        raise NotImplementedError("Home GEMM range does not support MoE LoRA")
    weights = mci.weights
    if any(getattr(weights, name) is not None for name in
           ("w1_bias", "w2_bias", "w1_scale_bias", "w2_scale_bias", "w1_offset", "w2_offset")):
        raise NotImplementedError("Home GEMM range does not support bias/antiquant offsets")
    x = mci.hidden_states
    counts = mci.group_list
    if counts.dtype != torch.int64:
        counts = counts.to(torch.int64)
    groups = counts.numel()
    if not 0 < home_experts <= groups:
        raise ValueError("Invalid Home expert boundary")
    w1 = _one_tensor(weights.w1, "w1")
    w2 = _one_tensor(weights.w2, "w2")
    if w1.shape[0] != groups or w2.shape[0] != groups:
        raise ValueError("Home/Slot weight groups must match dispatch counts")
    quantized = mci.quant.is_quant
    if quantized and mci.quant.quant_type != QuantType.W4A8MXFP:
        raise NotImplementedError(f"Home GEMM backend is not adapted: {mci.quant.quant_type}")
    if quantized and mci.activation != MoEActivation.SILU:
        raise NotImplementedError("Home MXFP4 GEMM currently requires SiLU")
    if not quantized and x.dtype != torch.bfloat16:
        raise NotImplementedError("Unquantized Home GEMM currently requires BF16")

    gate_up = torch.empty((x.shape[0], 2 * intermediate_size), dtype=torch.bfloat16, device=x.device)
    main = torch_npu.npu.current_stream()
    if quantized:
        if mci.quant.mxfp is None:
            raise ValueError("Missing MXFP quantization parameters")
        if mci.dynamic_scale is None:
            quant_x, x_scale = DeviceOperator.npu_dynamic_quant(
                hidden_states=x, dynamic_scale=None, act_quant_type=torch.float8_e4m3fn,
                use_mxfp_quant=True,
            )
            dispose_tensor(x)
        else:
            quant_x = x
            x_scale = DeviceOperator.maybe_normalize_mxfp_scale_layout(mci.dynamic_scale)
        w1_scale = _one_tensor(weights.w1_scale, "w1_scale")
        w2_scale = _one_tensor(weights.w2_scale, "w2_scale")
        range_kernel.mxfp4(quant_x, w1, w1_scale, x_scale, counts, gate_up, 0, home_experts)
        if prefetch_ready is not None:
            main.wait_event(prefetch_ready)
        range_kernel.mxfp4(quant_x, w1, w1_scale, x_scale, counts, gate_up, home_experts, groups)
        dispose_tensor(quant_x)
        hidden, hidden_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
            gate_up, topk_weight=None, group_index=None, dst_type=torch.float8_e4m3fn,
            quant_mode=2, clamp_value=mci.swiglu_limit,
        )
        hidden_scale = DeviceOperator.maybe_normalize_mxfp_scale_layout(hidden_scale)
        dispose_tensor(gate_up)
        mxfp = mci.quant.mxfp
        before_gmm2_evt = main.record_event()
        rows = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden, weight=w2, weight_scale=w2_scale,
            per_token_scale=hidden_scale, group_list=counts, group_list_type=1,
            input_dtype=torch.bfloat16, act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=mxfp.weight_quant_type, scale_type=mxfp.scale_dtype,
            per_token_scale_type=mxfp.per_token_scale_dtype, use_bf16=mxfp.use_bf16,
            use_mxfp_quant=True, bias=None, fallback_output_dtype=torch.bfloat16,
            mxfp_quant_dtype=QuantType.W4A8MXFP,
        )
        return rows, before_gmm2_evt

    if mci.need_trans:
        w1, w2 = w1.transpose(1, 2), w2.transpose(1, 2)
    range_kernel.bf16(x, w1, counts, gate_up, 0, home_experts)
    if prefetch_ready is not None:
        main.wait_event(prefetch_ready)
    range_kernel.bf16(x, w1, counts, gate_up, home_experts, groups)
    hidden = _bf16_activation(gate_up, mci.activation, mci.swiglu_limit)
    if mci.topk_scales is not None:
        hidden *= mci.topk_scales
    before_gmm2_evt = main.record_event()
    rows = torch_npu.npu_grouped_matmul(
        x=[hidden], weight=[w2], split_item=2, group_list_type=1,
        group_type=0, group_list=counts,
    )[0]
    return rows, before_gmm2_evt
