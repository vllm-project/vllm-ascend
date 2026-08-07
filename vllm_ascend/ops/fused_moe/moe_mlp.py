# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from __future__ import annotations

import torch
import torch_npu

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.moe_mlp_activation import (
    MoEMLPActivationKind,
    apply_quantized_activation,
    apply_unquantized_activation,
    resolve_mlp_activation,
)
from vllm_ascend.ops.fused_moe.moe_mlp_group_list import convert_group_list
from vllm_ascend.ops.fused_moe.moe_mlp_plan import GateUpKernel, MoEMLPPlan
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import dispose_tensor


def cumsum_group_list(
    group_list: torch.Tensor, src_list_type: int, dst_list_type: int, active_num: int = 0, expert_num: int = 0
) -> torch.Tensor:
    """Compatibility wrapper for callers migrating to convert_group_list."""
    return convert_group_list(group_list, src_list_type, dst_list_type, active_num, expert_num)


def _require_single_tensor(tensor_or_list: list[torch.Tensor] | torch.Tensor, *, name: str) -> torch.Tensor:
    if isinstance(tensor_or_list, list):
        if len(tensor_or_list) != 1:
            raise ValueError(f"{name} must be a tensor or a single-element list, but got {len(tensor_or_list)}.")
        return tensor_or_list[0]
    return tensor_or_list


def _prepare_dequant_swiglu_weight_scale(
    w1_scale: list[torch.Tensor] | torch.Tensor,
    is_swigluoai_uninterleave: bool,
) -> torch.Tensor:
    if not is_swigluoai_uninterleave:
        weight_scale = w1_scale[0]
    elif isinstance(w1_scale, list):
        if len(w1_scale) == 1:
            weight_scale = w1_scale[0]
        else:
            weight_scale = torch.stack([scale.reshape(-1) for scale in w1_scale], dim=0)
    else:
        weight_scale = w1_scale
    if weight_scale.dtype != torch.float32:
        weight_scale = weight_scale.to(torch.float32)
    if is_swigluoai_uninterleave and weight_scale.dim() == 1:
        weight_scale = weight_scale.reshape(1, -1)
    return weight_scale


def _prepare_swigluoai_grouped_matmul_scales(
    weight_scale: list[torch.Tensor] | torch.Tensor, output_dtype: torch.dtype
) -> list[torch.Tensor]:
    scales = weight_scale if isinstance(weight_scale, list) else [weight_scale]
    return [scale.to(output_dtype) if scale.dtype != output_dtype else scale for scale in scales]


def weight_only_apply_mlp(
    hidden_states: torch.Tensor,
    w1: list[torch.Tensor] | torch.Tensor,
    w1_scale: list[torch.Tensor] | torch.Tensor,
    w1_offset: torch.Tensor,
    w2: list[torch.Tensor] | torch.Tensor,
    w2_scale: list[torch.Tensor] | torch.Tensor,
    w2_offset: torch.Tensor,
    group_list: torch.Tensor,
    output_dtype: torch.dtype,
    activation: MoEMLPActivationKind,
    group_list_type: int = 1,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Execute the W4A16 antiquant GMM1 -> activation -> GMM2 path."""
    w1 = _require_single_tensor(w1, name="w1")
    w1_scale = _require_single_tensor(w1_scale, name="w1_scale")
    w2 = _require_single_tensor(w2, name="w2")
    w2_scale = _require_single_tensor(w2_scale, name="w2_scale")

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        antiquant_scale=[w1_scale],
        antiquant_offset=[w1_offset],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]
    dispose_tensor(hidden_states)
    gate_up_out = apply_unquantized_activation(
        gate_up_out,
        activation,
        hidden_size=w1.shape[-1],
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )

    before_gmm2_evt = torch.npu.current_stream().record_event()
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        antiquant_scale=[w2_scale],
        antiquant_offset=[w2_offset],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]
    return hidden_states, before_gmm2_evt


def quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: list[torch.Tensor] | torch.Tensor,
    w1_scale: list[torch.Tensor] | torch.Tensor,
    w2: list[torch.Tensor] | torch.Tensor,
    w2_scale: list[torch.Tensor] | torch.Tensor,
    output_dtype: torch.dtype,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor = None,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    use_mxfp_quant: bool = False,
    mxfp_quant_dtype: QuantType | None = None,
    act_quant_type: torch.dtype = torch.float8_e4m3fn,
    weight_quant_type: torch.dtype | None = None,
    scale_type: torch.dtype | None = None,
    per_token_scale_type: torch.dtype | None = None,
    use_bf16: bool = True,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    kernel_plan: MoEMLPPlan | None = None,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    if kernel_plan is None:
        raise ValueError("kernel_plan is required for quantized MoE MLP execution.")
    gate_up_kernel = kernel_plan.gate_up_kernel
    if gate_up_kernel in (GateUpKernel.UNQUANTIZED, GateUpKernel.ANTIQUANT):
        raise ValueError(f"{gate_up_kernel} must use its dedicated MLP executor.")
    is_w4a8_per_channel = kernel_plan.key.quant_type == QuantType.W4A8
    if is_w4a8_per_channel and (w1_scale_bias is None or w2_scale_bias is None):
        raise ValueError("W4A8 per-channel MLP requires both w1_scale_bias and w2_scale_bias.")
    is_swigluoai_uninterleave = kernel_plan.key.activation == MoEMLPActivationKind.SWIGLUOAI_UNINTERLEAVE

    if use_mxfp_quant:
        if w1_scale_bias is not None or w2_scale_bias is not None:
            raise NotImplementedError("MXFP path does not support scale_bias yet.")

    if mxfp_quant_dtype == QuantType.W4A16MXFP:
        quantized_hidden_states = None
        pertoken_scale = None
    elif dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=hidden_states,
            dynamic_scale=None,
            act_quant_type=act_quant_type,
            use_mxfp_quant=use_mxfp_quant,
        )
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        pertoken_scale = (
            DeviceOperator.maybe_normalize_mxfp_scale_layout(dynamic_scale) if use_mxfp_quant else dynamic_scale
        )
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    if is_w4a8_per_channel:
        if group_list_type == 0:
            group_list = convert_group_list(group_list, group_list_type, 1)
            group_list_type = 1
        bias1 = w1_scale_bias
        bias2 = w2_scale_bias

    if gate_up_kernel == GateUpKernel.CUSTOM_NZ:
        hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
            x=hidden_states,
            weight=w1,
            weight_scale=w1_scale,
            x_scale=pertoken_scale,
            group_list=cumsum_group_list(group_list, group_list_type, 0),
            swiglu_limit=swiglu_limit,
        )
    elif gate_up_kernel == GateUpKernel.FUSED:
        hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
            x=hidden_states,
            weight=_require_single_tensor(w1, name="w1"),
            group_list=cumsum_group_list(group_list, group_list_type, 0),
            weight_scale=_require_single_tensor(w1_scale, name="w1_scale"),
            x_scale=pertoken_scale,
            bias=None,
            use_mxfp_quant=use_mxfp_quant,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            swiglu_limit=swiglu_limit,
            mxfp_quant_dtype=mxfp_quant_dtype,
        )
        if quantized_hidden_states is not None:
            dispose_tensor(quantized_hidden_states)
    elif gate_up_kernel == GateUpKernel.DECOMPOSED:
        gmm1_output_dtype = (
            torch.bfloat16
            if is_w4a8_per_channel or use_mxfp_quant or kernel_plan.key.activation == MoEMLPActivationKind.SWIGLUSTEP
            else output_dtype
        )
        scale = [w1_scale[0].to(w2_scale[0].dtype)] if isinstance(w1_scale, list) else [w1_scale]
        if is_swigluoai_uninterleave:
            scale = _prepare_swigluoai_grouped_matmul_scales(w1_scale, gmm1_output_dtype)
        gmm1_kwargs = {
            "x": [hidden_states],
            "weight": w1 if isinstance(w1, list) else [w1],
            "scale": scale,
            "bias": bias1,
            "per_token_scale": [pertoken_scale],
            "split_item": 2,
            "group_type": 0,
            "group_list": group_list,
            "group_list_type": group_list_type,
            "output_dtype": gmm1_output_dtype,
        }
        if use_mxfp_quant:
            gmm1_kwargs.update(
                {
                    "scale_dtype": torch_npu.float8_e8m0fnu,
                    "per_token_scale_dtype": torch_npu.float8_e8m0fnu,
                }
            )
        hidden_states = torch_npu.npu_grouped_matmul(**gmm1_kwargs)[0]
        if quantized_hidden_states is not None:
            dispose_tensor(quantized_hidden_states)
        hidden_states, swiglu_out_scale = apply_quantized_activation(
            hidden_states,
            kernel_plan.key.activation,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            act_quant_type=act_quant_type,
            use_mxfp_quant=use_mxfp_quant,
            group_list=group_list,
            group_list_type=group_list_type,
        )
    elif gate_up_kernel == GateUpKernel.DEQUANT_SWIGLU:
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w1,
            split_item=3,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.int32,
        )[0]
        if quantized_hidden_states is not None:
            dispose_tensor(quantized_hidden_states)
        dequant_swiglu_kwargs = {
            "x": hidden_states,
            "weight_scale": _prepare_dequant_swiglu_weight_scale(w1_scale, is_swigluoai_uninterleave),
            "activation_scale": pertoken_scale,
            "bias": None,
            "quant_scale": None,
            "quant_offset": None,
            "group_index": cumsum_group_list(group_list, group_list_type, 1),
            "activate_left": True,
            "quant_mode": 1,
        }
        if is_swigluoai_uninterleave:
            dequant_swiglu_kwargs.update(
                {
                    "swiglu_mode": 1,
                    "clamp_limit": swiglu_limit,
                    "glu_alpha": swiglu_alpha,
                    "glu_bias": swiglu_beta,
                }
            )
        hidden_states, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(**dequant_swiglu_kwargs)
    elif gate_up_kernel == GateUpKernel.W4A8_PER_CHANNEL:
        hidden_states, swiglu_out_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
            x=hidden_states,
            weight=w1,
            weight_scale=w1_scale if isinstance(w1_scale, list) else [w1_scale],
            x_scale=pertoken_scale,
            group_list=group_list,
            weight_assist_matrix=bias1,
            dequant_mode=0,
            group_list_type=group_list_type,
            swiglu_limit=swiglu_limit,
        )
    else:
        raise ValueError(f"Unsupported quantized gate/up kernel: {gate_up_kernel}")

    before_gmm2_evt = torch.npu.current_stream().record_event()
    # gmm2: down_proj
    hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
        hidden_states=hidden_states,
        weight=w2,
        weight_scale=w2_scale,
        per_token_scale=swiglu_out_scale,
        group_list=group_list,
        group_list_type=group_list_type,
        input_dtype=output_dtype,
        act_quant_type=act_quant_type,
        weight_quant_type=weight_quant_type,
        scale_type=scale_type,
        per_token_scale_type=per_token_scale_type,
        use_bf16=use_bf16,
        use_mxfp_quant=use_mxfp_quant,
        bias=bias2,
        fallback_output_dtype=output_dtype,
        mxfp_quant_dtype=mxfp_quant_dtype,
    )
    return hidden_states, before_gmm2_evt


def unquant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    w1_bias: torch.Tensor = None,
    w2_bias: torch.Tensor = None,
    activation: str | None = None,
    group_list_type: int = 1,
    topk_scales: torch.Tensor | None = None,
    need_trans: bool = True,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    lora_context=None,
    expanded_row_idx: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, None]:
    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    # MoE LoRA: only attempt injection when an adapter wraps this layer and
    # the comm method provided routing metadata in lora_context.
    # Two paths are supported:
    #   - AllGather: expanded_row_idx + topk_ids from npu_moe_init_routing
    #   - AlltoAll:  lora_context.exchanged_lora_indices + group_list after all_to_all
    lora_routing = None
    if lora_context is not None:  # LoRA applied
        from vllm_ascend.lora.fused_moe import (
            _recover_moe_lora_routing_all2all,
            _recover_moe_lora_routing_allgather,
            moe_lora_apply_w2,
            moe_lora_apply_w13,
        )

        if expanded_row_idx is not None and topk_ids is not None:
            # AllGather path: use npu_moe_init_routing's expanded_row_idx.
            lora_routing = _recover_moe_lora_routing_allgather(lora_context, expanded_row_idx, topk_ids)
        elif getattr(lora_context, "exchanged_lora_indices", None) is not None:
            # AlltoAll path: tokens already sorted by expert after exchange.
            # Build per-row (expert_id, lora_id) directly from group_list.
            lora_routing = _recover_moe_lora_routing_all2all(
                lora_context,
                group_list=group_list,
            )
        else:
            raise AssertionError(
                "MoE LoRA requires either expanded_row_idx+topk_ids "
                "(AllGather) or lora_context.exchanged_lora_indices "
                "(AlltoAll). Neither was provided."
            )

        moe_lora_apply_w13(
            lora_context,
            gate_up_out=gate_up_out,
            hidden_states=hidden_states,
            lora_routing=lora_routing,
        )

    gate_up_out = apply_unquantized_activation(
        gate_up_out,
        resolve_mlp_activation(activation),
        hidden_size=w1.shape[-1],
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )

    if topk_scales is not None:
        gate_up_out *= topk_scales

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    # LoRA w2 delta: applied to the down-proj output, with the activation output
    # as the lora_a input. Reuses the per-row routing computed for w13.
    if lora_routing is not None:
        moe_lora_apply_w2(
            lora_context,
            down_out=hidden_states,
            silu_out=gate_up_out,
            lora_routing=lora_routing,
        )
    return hidden_states, None


def execute_mlp_plan(
    kernel_plan: MoEMLPPlan,
    mlp_compute_input: MoEMlpComputeInput,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Execute the kernels selected by an immutable MoE MLP plan."""
    hidden_states = mlp_compute_input.hidden_states
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    dynamic_scale = mlp_compute_input.dynamic_scale
    topk_scales = mlp_compute_input.topk_scales
    w1 = mlp_compute_input.weights.w1
    w2 = mlp_compute_input.weights.w2
    w1_bias = mlp_compute_input.weights.w1_bias
    w2_bias = mlp_compute_input.weights.w2_bias
    w1_scale = mlp_compute_input.weights.w1_scale
    w2_scale = mlp_compute_input.weights.w2_scale
    w1_scale_bias = mlp_compute_input.weights.w1_scale_bias
    w2_scale_bias = mlp_compute_input.weights.w2_scale_bias
    w1_offset = mlp_compute_input.weights.w1_offset
    w2_offset = mlp_compute_input.weights.w2_offset
    activation = mlp_compute_input.activation
    need_trans = mlp_compute_input.need_trans
    swiglu_limit = mlp_compute_input.swiglu_limit
    swiglu_alpha = mlp_compute_input.swiglu_alpha
    swiglu_beta = mlp_compute_input.swiglu_beta

    gate_up_kernel = kernel_plan.gate_up_kernel
    if gate_up_kernel == GateUpKernel.UNQUANTIZED:
        return unquant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            activation=activation,
            group_list=group_list,
            group_list_type=group_list_type,
            topk_scales=topk_scales,
            need_trans=need_trans,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            lora_context=mlp_compute_input.lora_context,
            expanded_row_idx=mlp_compute_input.expanded_row_idx,
            topk_ids=mlp_compute_input.topk_ids,
        )

    # Quantized MoE LoRA remains out of scope until a dedicated quantized
    # adapter kernel is available; the existing unquantized path stays intact.
    from vllm_ascend.lora.fused_moe import has_lora

    if has_lora(mlp_compute_input.lora_context):
        from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora

        return quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_compute_input)

    assert w1_scale is not None and w2_scale is not None
    output_dtype = mlp_compute_input.output_dtype
    if gate_up_kernel == GateUpKernel.ANTIQUANT:
        assert w1_offset is not None and w2_offset is not None
        return weight_only_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w1_offset=w1_offset,
            w2=w2,
            w2_scale=w2_scale,
            w2_offset=w2_offset,
            group_list=group_list,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
            activation=kernel_plan.key.activation,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )
    if w1_offset is not None or w2_offset is not None:
        raise ValueError("Antiquant offsets are only supported by the weight-only MLP path.")

    act_quant_type = torch.int8 if mlp_compute_input.quant.is_int_quant else torch.float8_e4m3fn
    weight_quant_type = torch.float8_e4m3fn
    scale_type = None
    per_token_scale_type = None
    use_bf16 = output_dtype == torch.bfloat16
    use_mxfp_quant = mlp_compute_input.quant.is_mxfp
    mxfp_quant_dtype = mlp_compute_input.quant.quant_type

    if use_mxfp_quant:
        mxfp = mlp_compute_input.quant.mxfp
        assert mxfp is not None, "mlp_compute_input.quant.mxfp is required when quant_type is W8A8MXFP."
        act_quant_type = mxfp.act_quant_type or act_quant_type
        if mxfp_quant_dtype == QuantType.W4A16MXFP:
            act_quant_type = mxfp.act_quant_type
        weight_quant_type = mxfp.weight_quant_type or weight_quant_type
        if mxfp_quant_dtype in [QuantType.W4A8MXFP, QuantType.W4A16MXFP]:
            weight_quant_type = mxfp.weight_quant_type
        scale_type = mxfp.scale_dtype
        per_token_scale_type = mxfp.per_token_scale_dtype
        use_bf16 = mxfp.use_bf16

    return quant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        output_dtype=output_dtype,
        group_list=group_list,
        dynamic_scale=dynamic_scale,
        group_list_type=group_list_type,
        w1_scale_bias=w1_scale_bias,
        w2_scale_bias=w2_scale_bias,
        use_mxfp_quant=use_mxfp_quant,
        mxfp_quant_dtype=mxfp_quant_dtype,
        act_quant_type=act_quant_type,
        weight_quant_type=weight_quant_type,
        scale_type=scale_type,
        per_token_scale_type=per_token_scale_type,
        use_bf16=use_bf16,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        kernel_plan=kernel_plan,
    )


def unified_apply_mlp(*, mlp_compute_input: MoEMlpComputeInput) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Compatibility entry point for callers outside MoECommMethod.

    Communication methods keep a long-lived planner and therefore avoid this
    per-call planner construction.
    """
    from vllm_ascend.ops.fused_moe.moe_mlp_plan import MoEMLPPlanner

    return MoEMLPPlanner().execute(mlp_compute_input)
