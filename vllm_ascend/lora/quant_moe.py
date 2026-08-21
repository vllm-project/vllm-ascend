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
"""Extensible quantized MoE LoRA execution for Ascend.

Each quantization scheme registers the dispatch policy and MLP implementation
needed to preserve floating-point LoRA boundaries around quantized base expert
matmuls. The token dispatcher and the common MoE MLP entry therefore do not
need scheme-specific dtype checks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.fused_moe import (
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
    reset_lora_indices,
)
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType

QuantMoELoRAApply = Callable[[MoEMlpComputeInput], tuple[torch.Tensor, torch.npu.Event | None]]
QuantMoELoRAActivationValidator = Callable[[torch.Tensor, torch.Tensor | None], None]


@dataclass(frozen=True)
class QuantMoELoRAImpl:
    apply: QuantMoELoRAApply
    validate_activation_input: QuantMoELoRAActivationValidator | None


@dataclass(frozen=True)
class _CompositeLoraGMMRouting:
    permutation: torch.Tensor
    group_list: torch.Tensor
    enabled: torch.Tensor


@dataclass(frozen=True)
class _SingleLoraGMMRouting:
    slot: torch.Tensor
    enabled: torch.Tensor


_QUANT_MOE_LORA_IMPLS: dict[QuantType, QuantMoELoRAImpl] = {}

MOE_LORA_GMM_MAX_LORAS = 3
MOE_LORA_GMM_RANK = 16
MOE_LORA_GMM_TOP_K = 6
MOE_LORA_GMM_MIN_ROWS_PER_GROUP = 8


def register_quant_moe_lora_impl(
    quant_type: QuantType,
    *,
    validate_activation_input: QuantMoELoRAActivationValidator | None = None,
):
    """Register one quantized MoE LoRA implementation."""

    def decorator(apply: QuantMoELoRAApply) -> QuantMoELoRAApply:
        if quant_type in _QUANT_MOE_LORA_IMPLS:
            raise ValueError(f"Quantized MoE LoRA implementation already registered for {quant_type}.")
        _QUANT_MOE_LORA_IMPLS[quant_type] = QuantMoELoRAImpl(
            apply=apply,
            validate_activation_input=validate_activation_input,
        )
        return apply

    return decorator


def _get_quant_moe_lora_impl(quant_type: QuantType) -> QuantMoELoRAImpl:
    impl = _QUANT_MOE_LORA_IMPLS.get(quant_type)
    if impl is None:
        supported = ", ".join(item.name for item in _QUANT_MOE_LORA_IMPLS)
        raise NotImplementedError(
            "Ascend quantized MoE LoRA has no implementation registered for "
            f"{quant_type.name}. Registered quant types: {supported or 'none'}."
        )
    return impl


def quant_apply_mlp_with_moe_lora(
    *,
    mlp_compute_input: MoEMlpComputeInput,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Dispatch an active quantized MoE LoRA batch to its backend."""
    return _get_quant_moe_lora_impl(mlp_compute_input.quant.quant_type).apply(mlp_compute_input)


def validate_quant_moe_lora_activation_input(
    *,
    quant_type: QuantType,
    hidden_states: torch.Tensor,
    dynamic_scale: torch.Tensor | None,
) -> None:
    """Validate activations before quantized MoE LoRA prepare/dispatch."""
    impl = _get_quant_moe_lora_impl(quant_type)
    if impl.validate_activation_input is not None:
        impl.validate_activation_input(hidden_states, dynamic_scale)


def _apply_moe_activation(
    gate_up_out: torch.Tensor,
    activation: str | None,
    swiglu_limit: float,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> torch.Tensor:
    """Match the activation semantics of the common unquantized MoE path."""
    act_name = getattr(activation, "value", activation)
    if activation == MoEActivation.SWIGLUOAI:
        return AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up_out)
    if act_name == "swigluoai_uninterleave":
        return torch_npu.npu_clipped_swiglu(
            gate_up_out,
            interleaved=False,
            alpha=swiglu_alpha,
            limit=swiglu_limit,
            bias=swiglu_beta,
        )
    if activation == MoEActivation.SWIGLUSTEP:
        return AscendSwigluStepAndMul.swiglustep_forward(gate_up_out, limit=swiglu_limit or 7.0)
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


def _validate_dynamic_int8_activations(
    hidden_states: torch.Tensor,
    dynamic_scale: torch.Tensor | None,
) -> None:
    if dynamic_scale is not None or hidden_states.dtype == torch.int8:
        raise NotImplementedError("Dynamic INT8 MoE LoRA requires unquantized activations before expert routing.")


def _can_use_single_lora_gmm(
    lora_context,
    *,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
) -> bool:
    """Return whether one active adapter can use expert-grouped GMM."""
    punica_wrapper = getattr(lora_context, "punica_wrapper", None)
    if punica_wrapper is None:
        return False
    if (
        getattr(punica_wrapper, "num_active_moe_loras", 0) != 1
        or lora_context.fully_sharded
        or group_list_type != 1
        or lora_context.top_k != MOE_LORA_GMM_TOP_K
        or hidden_states.dtype != torch.bfloat16
    ):
        return False

    w13_a = lora_context.w13_lora_a_stacked
    w13_b = lora_context.w13_lora_b_stacked
    w2_a = lora_context.w2_lora_a_stacked
    w2_b = lora_context.w2_lora_b_stacked
    weights = (*w13_a, *w13_b, *w2_a, *w2_b)
    num_experts = group_list.numel()
    return not (
        not weights
        or any(weight.shape[0] != MOE_LORA_GMM_MAX_LORAS for weight in weights)
        or any(weight.shape[1] != num_experts for weight in weights)
        or any(weight.dtype != hidden_states.dtype for weight in weights)
        or any(weight.shape[-2] != MOE_LORA_GMM_RANK for weight in (*w13_a, *w2_a))
        or any(weight.shape[-1] != MOE_LORA_GMM_RANK for weight in (*w13_b, *w2_b))
        or hidden_states.shape[0] < MOE_LORA_GMM_MIN_ROWS_PER_GROUP * num_experts
    )


def _can_use_composite_lora_gmm(
    lora_context,
    *,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
) -> bool:
    """Return whether mixed requests can use ``(slot, expert)`` GMM."""
    punica_wrapper = getattr(lora_context, "punica_wrapper", None)
    if (
        punica_wrapper is None
        or getattr(punica_wrapper, "no_lora", True)
        or getattr(punica_wrapper, "num_active_moe_loras", 0) < 2
        or lora_context.fully_sharded
        or group_list_type != 1
        or lora_context.top_k != MOE_LORA_GMM_TOP_K
        or hidden_states.dtype != torch.bfloat16
    ):
        return False

    w13_a = lora_context.w13_lora_a_stacked
    w13_b = lora_context.w13_lora_b_stacked
    w2_a = lora_context.w2_lora_a_stacked
    w2_b = lora_context.w2_lora_b_stacked
    weights = (*w13_a, *w13_b, *w2_a, *w2_b)
    num_experts = group_list.numel()
    num_composite_groups = MOE_LORA_GMM_MAX_LORAS * num_experts
    return not (
        not weights
        or any(weight.shape[0] != MOE_LORA_GMM_MAX_LORAS for weight in weights)
        or any(weight.shape[1] != num_experts for weight in weights)
        or any(weight.dtype != hidden_states.dtype for weight in weights)
        or any(weight.shape[-2] != MOE_LORA_GMM_RANK for weight in (*w13_a, *w2_a))
        or any(weight.shape[-1] != MOE_LORA_GMM_RANK for weight in (*w13_b, *w2_b))
        or hidden_states.shape[0] < MOE_LORA_GMM_MIN_ROWS_PER_GROUP * num_composite_groups
    )


def _build_single_lora_gmm_routing(
    lora_context,
    *,
    expanded_row_idx: torch.Tensor,
    topk_ids: torch.Tensor,
) -> _SingleLoraGMMRouting:
    """Select one adapter and align its active/base mask to dispatched rows."""
    token_lora_indices = lora_context.punica_wrapper.token_lora_indices
    token_lora_slots = token_lora_indices[: topk_ids.shape[0]]
    max_loras = lora_context.w13_lora_a_stacked[0].shape[0]
    safe_token_lora_slots = token_lora_slots.clamp(min=0, max=max_loras - 1)
    token_enabled = (token_lora_slots >= 0) & lora_context.adapter_enabled[safe_token_lora_slots].bool()
    pair_enabled = token_enabled.unsqueeze(-1).expand_as(topk_ids).reshape(-1)

    dispatched_rows = expanded_row_idx.abs().reshape(-1).to(torch.long)
    dispatched_enabled = torch.empty_like(pair_enabled)
    dispatched_enabled.index_copy_(0, dispatched_rows, pair_enabled)
    return _SingleLoraGMMRouting(
        slot=safe_token_lora_slots.max().reshape(1),
        enabled=dispatched_enabled,
    )


def _build_composite_lora_gmm_routing(
    lora_context,
    *,
    expanded_row_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> _CompositeLoraGMMRouting:
    """Group dispatched rows by ``(LoRA slot, expert)`` without host sync."""
    token_lora_indices = lora_context.punica_wrapper.token_lora_indices
    token_lora_slots = token_lora_indices[: topk_ids.shape[0]]

    max_loras = lora_context.w13_lora_a_stacked[0].shape[0]
    safe_token_lora_slots = token_lora_slots.clamp(min=0, max=max_loras - 1)
    adapter_enabled = lora_context.adapter_enabled
    token_enabled = (token_lora_slots >= 0) & adapter_enabled[safe_token_lora_slots].bool()
    pair_enabled = token_enabled.unsqueeze(-1).expand_as(topk_ids).reshape(-1)

    # Inactive/base rows still need a static GMM group. Route them through an
    # enabled slot with zeroed inputs so unloaded weight buffers are not read.
    fallback_slot = torch.argmax(adapter_enabled[:max_loras])
    token_group_slots = torch.where(token_enabled, safe_token_lora_slots, fallback_slot)
    pair_group_ids = (token_group_slots.unsqueeze(-1) * num_experts + topk_ids.to(torch.long)).reshape(-1)

    dispatched_rows = expanded_row_idx.abs().reshape(-1).to(torch.long)
    dispatched_group_ids = torch.empty_like(pair_group_ids)
    dispatched_group_ids.index_copy_(0, dispatched_rows, pair_group_ids)
    dispatched_enabled = torch.empty_like(pair_enabled)
    dispatched_enabled.index_copy_(0, dispatched_rows, pair_enabled)

    _, permutation = torch.sort(dispatched_group_ids)
    grouped_enabled = dispatched_enabled.index_select(0, permutation)
    num_groups = max_loras * num_experts
    composite_group_list = torch.zeros(
        num_groups,
        dtype=torch.int64,
        device=dispatched_group_ids.device,
    )
    composite_group_list.scatter_add_(
        0,
        dispatched_group_ids,
        torch.ones_like(dispatched_group_ids, dtype=torch.int64),
    )
    return _CompositeLoraGMMRouting(
        permutation=permutation,
        group_list=composite_group_list,
        enabled=grouped_enabled,
    )


def _grouped_lora_matmul(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    """Apply one BF16 expert-grouped LoRA projection.

    LoRA factors are stored for BGMV as ``[E, N, K]``.  GMM consumes
    ``[E, K, N]``; the transpose is a view, matching the existing unquantized
    MoE GMM path and avoiding a full weight copy on every forward.
    """
    return torch_npu.npu_grouped_matmul(
        x=[inputs],
        weight=[weight.transpose(-1, -2)],
        split_item=2,
        group_type=0,
        group_list=group_list,
        group_list_type=1,
    )[0]


def _add_single_lora_gmm(
    output: torch.Tensor,
    inputs: torch.Tensor,
    lora_a_stacked: tuple[torch.Tensor, ...],
    lora_b_stacked: tuple[torch.Tensor, ...],
    *,
    routing: _SingleLoraGMMRouting,
    group_list: torch.Tensor,
) -> None:
    """Add one adapter with two grouped matmuls per output slice.

    Slot selection and the optional base-token mask remain graph tensors.
    """
    masked_inputs = inputs * routing.enabled.unsqueeze(-1).to(inputs.dtype)
    output_offset = 0
    for lora_a, lora_b in zip(lora_a_stacked, lora_b_stacked, strict=True):
        selected_lora_a = torch.index_select(lora_a, 0, routing.slot).squeeze(0)
        selected_lora_b = torch.index_select(lora_b, 0, routing.slot).squeeze(0)
        shrink = _grouped_lora_matmul(masked_inputs, selected_lora_a, group_list)
        delta = _grouped_lora_matmul(shrink, selected_lora_b, group_list)
        delta *= routing.enabled.unsqueeze(-1).to(delta.dtype)
        output_size = delta.shape[-1]
        output.narrow(-1, output_offset, output_size).add_(delta)
        output_offset += output_size


def _add_composite_lora_gmm(
    output: torch.Tensor,
    inputs: torch.Tensor,
    lora_a_stacked: tuple[torch.Tensor, ...],
    lora_b_stacked: tuple[torch.Tensor, ...],
    *,
    routing: _CompositeLoraGMMRouting,
) -> None:
    """Add LoRA deltas after grouping rows by dynamic slot and expert."""
    grouped_inputs = inputs.index_select(0, routing.permutation)
    grouped_inputs = grouped_inputs * routing.enabled.unsqueeze(-1).to(inputs.dtype)

    output_offset = 0
    for lora_a, lora_b in zip(lora_a_stacked, lora_b_stacked, strict=True):
        lora_a_flat = lora_a.flatten(0, 1)
        lora_b_flat = lora_b.flatten(0, 1)
        shrink = _grouped_lora_matmul(grouped_inputs, lora_a_flat, routing.group_list)
        delta = _grouped_lora_matmul(shrink, lora_b_flat, routing.group_list)
        delta *= routing.enabled.unsqueeze(-1).to(delta.dtype)
        output_size = delta.shape[-1]
        output.narrow(-1, output_offset, output_size).index_add_(
            0,
            routing.permutation,
            delta,
        )
        output_offset += output_size


@register_quant_moe_lora_impl(
    QuantType.W8A8,
    validate_activation_input=_validate_dynamic_int8_activations,
)
def _apply_dynamic_int8_moe_lora(
    mlp_compute_input: MoEMlpComputeInput,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Run INT8 base experts and inject LoRA at BF16/FP16 boundaries."""
    comm_type = _EXTRA_CTX.moe_comm_type
    if comm_type not in {MoECommType.ALLGATHER, MoECommType.ALLTOALL}:
        raise NotImplementedError(
            "Ascend quantized MoE LoRA currently supports the AllGather TP and AlltoAll EP paths; "
            "MC2 and FusedMC2 are unsupported."
        )
    lora_context = mlp_compute_input.lora_context
    if mlp_compute_input.dynamic_eplb:
        raise NotImplementedError("Ascend quantized MoE LoRA does not support dynamic EPLB.")

    hidden_states = mlp_compute_input.hidden_states
    if mlp_compute_input.dynamic_scale is not None or hidden_states.dtype == torch.int8:
        raise AssertionError(
            "Quantized MoE LoRA requires BF16/FP16 routed activations. "
            "Dispatch-side quantization must be disabled for LoRA batches."
        )
    if comm_type == MoECommType.ALLGATHER and (
        mlp_compute_input.expanded_row_idx is None or mlp_compute_input.topk_ids is None
    ):
        raise AssertionError("Quantized MoE LoRA requires AllGather routing metadata (expanded_row_idx and topk_ids).")
    if hidden_states.shape[0] == 0:
        # An EP rank may receive no routed tokens. Keep participating in the
        # surrounding AlltoAll collectives, but avoid empty-tensor NPU kernels.
        return hidden_states, None

    weights = mlp_compute_input.weights
    if weights.w1_scale_bias is not None or weights.w2_scale_bias is not None:
        raise NotImplementedError("Quantized MoE LoRA does not support fused scale-bias.")
    if weights.w1_offset is not None or weights.w2_offset is not None:
        raise NotImplementedError("Quantized MoE LoRA does not support antiquant offsets.")
    if weights.w1_scale is None or weights.w2_scale is None:
        raise AssertionError("Quantized MoE LoRA requires w1 and w2 weight scales.")

    w1 = weights.w1 if isinstance(weights.w1, list) else [weights.w1]
    w2 = weights.w2 if isinstance(weights.w2, list) else [weights.w2]
    w1_scale = weights.w1_scale if isinstance(weights.w1_scale, list) else [weights.w1_scale]
    w2_scale = weights.w2_scale if isinstance(weights.w2_scale, list) else [weights.w2_scale]
    if not all(len(values) == 1 for values in (w1, w2, w1_scale, w2_scale)):
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
        weight=w1,
        scale=[w1_scale[0].to(w2_scale[0].dtype)],
        per_token_scale=[input_scale],
        split_item=2,
        group_type=0,
        group_list=mlp_compute_input.group_list,
        group_list_type=mlp_compute_input.group_list_type,
        output_dtype=input_dtype,
    )[0]

    use_single_lora_gmm = False
    use_composite_lora_gmm = False
    if comm_type == MoECommType.ALLGATHER:
        use_single_lora_gmm = _can_use_single_lora_gmm(
            lora_context,
            hidden_states=hidden_states,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
        )
        if not use_single_lora_gmm:
            use_composite_lora_gmm = _can_use_composite_lora_gmm(
                lora_context,
                hidden_states=hidden_states,
                group_list=mlp_compute_input.group_list,
                group_list_type=mlp_compute_input.group_list_type,
            )

    lora_routing = None
    single_lora_routing = None
    composite_lora_routing = None
    if use_single_lora_gmm:
        single_lora_routing = _build_single_lora_gmm_routing(
            lora_context,
            expanded_row_idx=mlp_compute_input.expanded_row_idx,
            topk_ids=mlp_compute_input.topk_ids,
        )
        _add_single_lora_gmm(
            gate_up_out,
            hidden_states,
            lora_context.w13_lora_a_stacked,
            lora_context.w13_lora_b_stacked,
            routing=single_lora_routing,
            group_list=mlp_compute_input.group_list,
        )
    elif use_composite_lora_gmm:
        composite_lora_routing = _build_composite_lora_gmm_routing(
            lora_context,
            expanded_row_idx=mlp_compute_input.expanded_row_idx,
            topk_ids=mlp_compute_input.topk_ids,
            num_experts=mlp_compute_input.group_list.numel(),
        )
        _add_composite_lora_gmm(
            gate_up_out,
            hidden_states,
            lora_context.w13_lora_a_stacked,
            lora_context.w13_lora_b_stacked,
            routing=composite_lora_routing,
        )
    elif comm_type == MoECommType.ALLGATHER:
        lora_routing = _recover_moe_lora_routing_allgather(
            lora_context,
            mlp_compute_input.expanded_row_idx,
            mlp_compute_input.topk_ids,
        )
    else:
        lora_routing = _recover_moe_lora_routing_all2all(
            lora_context,
            group_list=mlp_compute_input.group_list,
        )
    if lora_routing is not None:
        moe_lora_apply_w13(
            lora_context,
            gate_up_out=gate_up_out,
            hidden_states=hidden_states,
            lora_routing=lora_routing,
        )

    activated = _apply_moe_activation(
        gate_up_out,
        mlp_compute_input.activation,
        mlp_compute_input.swiglu_limit,
        mlp_compute_input.swiglu_alpha,
        mlp_compute_input.swiglu_beta,
    )
    if mlp_compute_input.topk_scales is not None:
        activated *= mlp_compute_input.topk_scales

    quantized_activated, activated_scale = DeviceOperator.npu_dynamic_quant(
        hidden_states=activated,
        dynamic_scale=None,
        act_quant_type=torch.int8,
        use_mxfp_quant=False,
    )
    down_out = DeviceOperator.npu_grouped_matmul_gmm2(
        hidden_states=quantized_activated,
        weight=w2,
        weight_scale=w2_scale,
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
        fallback_output_dtype=w2_scale[0].dtype,
        mxfp_quant_dtype=None,
    )
    if use_single_lora_gmm:
        # activated already includes topk_scales, so the W2 LoRA delta must
        # not multiply routed weights a second time.
        _add_single_lora_gmm(
            down_out,
            activated,
            lora_context.w2_lora_a_stacked,
            lora_context.w2_lora_b_stacked,
            routing=single_lora_routing,
            group_list=mlp_compute_input.group_list,
        )
        reset_lora_indices(lora_context)
    elif use_composite_lora_gmm:
        _add_composite_lora_gmm(
            down_out,
            activated,
            lora_context.w2_lora_a_stacked,
            lora_context.w2_lora_b_stacked,
            routing=composite_lora_routing,
        )
        reset_lora_indices(lora_context)
    else:
        moe_lora_apply_w2(
            lora_context,
            down_out=down_out,
            silu_out=activated,
            lora_routing=lora_routing,
        )
    return down_out, None


__all__ = [
    "quant_apply_mlp_with_moe_lora",
    "register_quant_moe_lora_impl",
    "validate_quant_moe_lora_activation_input",
]
