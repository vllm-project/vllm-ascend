# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Extensible quantized MoE LoRA execution.

The public entry point is intentionally independent of a concrete quantization
scheme. Each supported scheme registers an implementation keyed by
``QuantType``. Implementations preserve floating-point activation boundaries
for LoRA while retaining the quantized base expert matmuls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.fused_moe import moe_lora_apply_w2, moe_lora_apply_w13
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType

QuantMoELoRAApply = Callable[[MoEMlpComputeInput], tuple[torch.Tensor, torch.npu.Event | None]]


@dataclass(frozen=True)
class QuantMoELoRAImpl:
    apply: QuantMoELoRAApply
    requires_unquantized_dispatch: bool


_QUANT_MOE_LORA_IMPLS: dict[QuantType, QuantMoELoRAImpl] = {}


def register_quant_moe_lora_impl(
    quant_type: QuantType,
    *,
    requires_unquantized_dispatch: bool = True,
):
    """Register the LoRA execution implementation for a quantized MoE type."""

    def decorator(apply: QuantMoELoRAApply) -> QuantMoELoRAApply:
        if quant_type in _QUANT_MOE_LORA_IMPLS:
            raise ValueError(f"Quantized MoE LoRA implementation already registered for {quant_type}.")
        _QUANT_MOE_LORA_IMPLS[quant_type] = QuantMoELoRAImpl(
            apply=apply,
            requires_unquantized_dispatch=requires_unquantized_dispatch,
        )
        return apply

    return decorator


def apply_quant_moe_lora(
    *,
    mlp_compute_input: MoEMlpComputeInput,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Dispatch a quantized MoE LoRA request to its quant implementation."""
    quant_type = mlp_compute_input.quant.quant_type
    impl = _get_quant_moe_lora_impl(quant_type)
    return impl.apply(mlp_compute_input)


def quant_moe_lora_requires_unquantized_dispatch(quant_type: QuantType) -> bool:
    """Return the dispatch policy declared by a quantized MoE LoRA impl."""
    return _get_quant_moe_lora_impl(quant_type).requires_unquantized_dispatch


def _get_quant_moe_lora_impl(quant_type: QuantType) -> QuantMoELoRAImpl:
    impl = _QUANT_MOE_LORA_IMPLS.get(quant_type)
    if impl is None:
        supported = ", ".join(item.name for item in _QUANT_MOE_LORA_IMPLS)
        raise NotImplementedError(
            "Ascend quantized MoE LoRA has no implementation registered for "
            f"{quant_type.name}. Registered quant types: {supported or 'none'}."
        )
    return impl


def _apply_moe_activation(
    gate_up_out: torch.Tensor,
    activation: str | None,
    swiglu_limit: float,
) -> torch.Tensor:
    if activation == MoEActivation.SWIGLUOAI:
        return AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up_out)
    if activation == MoEActivation.SWIGLUSTEP:
        return AscendSwigluStepAndMul.swiglustep_forward(
            gate_up_out,
            limit=swiglu_limit or 7.0,
        )
    if activation in (MoEActivation.GELU, MoEActivation.GELU_TANH):
        gate, up = gate_up_out.chunk(2, dim=-1)
        approximate = "tanh" if activation == MoEActivation.GELU_TANH else "none"
        return torch.nn.functional.gelu(gate, approximate=approximate) * up

    if swiglu_limit > 0:
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        gate_up_out = torch.cat((gate, up), dim=-1)
    return torch_npu.npu_swiglu(gate_up_out)


@register_quant_moe_lora_impl(QuantType.W8A8)
def _apply_dynamic_int8_moe_lora(
    mlp_compute_input: MoEMlpComputeInput,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Dynamic INT8 base experts with floating-point LoRA boundaries."""
    if _EXTRA_CTX.moe_comm_type != MoECommType.ALLGATHER:
        raise NotImplementedError(
            "Ascend quantized MoE LoRA currently supports only the "
            "AllGather TP path; EP, AlltoAll, MC2, and FusedMC2 are unsupported."
        )
    if mlp_compute_input.dynamic_eplb:
        raise NotImplementedError("Ascend quantized MoE LoRA does not support dynamic EPLB.")

    hidden_states = mlp_compute_input.hidden_states
    if mlp_compute_input.dynamic_scale is not None or hidden_states.dtype == torch.int8:
        raise AssertionError(
            "Quantized MoE LoRA requires BF16/FP16 routed activations. "
            "Dispatch-side quantization must be disabled for LoRA batches."
        )
    if mlp_compute_input.expanded_row_idx is None or mlp_compute_input.topk_ids is None:
        raise AssertionError("Quantized MoE LoRA requires AllGather routing metadata (expanded_row_idx and topk_ids).")

    weights = mlp_compute_input.weights
    if weights.w1_scale_bias is not None or weights.w2_scale_bias is not None:
        raise NotImplementedError("Quantized MoE LoRA does not support fused scale-bias.")
    if weights.w1_offset is not None or weights.w2_offset is not None:
        raise NotImplementedError("Quantized MoE LoRA does not support antiquant offsets.")
    if weights.w1_scale is None or weights.w2_scale is None:
        raise AssertionError("Quantized MoE LoRA requires w1 and w2 weight scales.")

    w1_list = weights.w1 if isinstance(weights.w1, list) else [weights.w1]
    w2_list = weights.w2 if isinstance(weights.w2, list) else [weights.w2]
    w1_scale_list = weights.w1_scale if isinstance(weights.w1_scale, list) else [weights.w1_scale]
    w2_scale_list = weights.w2_scale if isinstance(weights.w2_scale, list) else [weights.w2_scale]
    if not all(len(values) == 1 for values in (w1_list, w2_list, w1_scale_list, w2_scale_list)):
        raise NotImplementedError("Quantized MoE LoRA does not support per-expert tensor lists used by dynamic EPLB.")

    input_dtype = hidden_states.dtype
    quantized_input, input_scale = DeviceOperator.npu_dynamic_quant(
        hidden_states=hidden_states,
        dynamic_scale=None,
        act_quant_type=torch.int8,
        use_mxfp_quant=False,
    )
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[quantized_input],
        weight=w1_list,
        scale=[w1_scale_list[0].to(w2_scale_list[0].dtype)],
        per_token_scale=[input_scale],
        split_item=2,
        group_type=0,
        group_list=mlp_compute_input.group_list,
        group_list_type=mlp_compute_input.group_list_type,
        output_dtype=input_dtype,
    )[0]
    lora_routing = moe_lora_apply_w13(
        mlp_compute_input.lora_context,
        gate_up_out=gate_up_out,
        hidden_states=hidden_states,
        expanded_row_idx=mlp_compute_input.expanded_row_idx,
        topk_ids=mlp_compute_input.topk_ids,
    )

    activated = _apply_moe_activation(
        gate_up_out,
        mlp_compute_input.activation,
        mlp_compute_input.swiglu_limit,
    )
    quantized_activated, activated_scale = DeviceOperator.npu_dynamic_quant(
        hidden_states=activated,
        dynamic_scale=None,
        act_quant_type=torch.int8,
        use_mxfp_quant=False,
    )
    before_gmm2_evt = torch.npu.current_stream().record_event()
    down_out = DeviceOperator.npu_grouped_matmul_gmm2(
        hidden_states=quantized_activated,
        weight=w2_list,
        weight_scale=w2_scale_list,
        per_token_scale=activated_scale,
        group_list=mlp_compute_input.group_list,
        group_list_type=mlp_compute_input.group_list_type,
        input_dtype=input_dtype,
        act_quant_type=torch.int8,
        weight_quant_type=None,
        scale_type=None,
        per_token_scale_type=None,
        use_bf16=input_dtype == torch.bfloat16,
        use_mxfp_quant=False,
        bias=None,
        fallback_output_dtype=w2_scale_list[0].dtype,
        mxfp_quant_dtype=None,
    )
    moe_lora_apply_w2(
        mlp_compute_input.lora_context,
        down_out=down_out,
        silu_out=activated,
        lora_routing=lora_routing,
    )
    return down_out, before_gmm2_evt


__all__ = [
    "apply_quant_moe_lora",
    "quant_moe_lora_requires_unquantized_dispatch",
    "register_quant_moe_lora_impl",
]
