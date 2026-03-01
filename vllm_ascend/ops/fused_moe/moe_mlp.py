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


import torch
import torch_npu
from torch.nn.functional import pad
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul
from vllm_ascend.quantization.mxfp_compat import (
    ensure_mxfp8_moe_available,
)
from vllm_ascend.utils import (
    dispose_tensor,
    enable_custom_op,
    get_weight_prefetch_method,
)


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
    return fusion and dynamic_eplb and enable_custom_op()


def cumsum_group_list(
    group_list: torch.Tensor, src_list_type: int, dst_list_type: int, active_num: int = 0, expert_num: int = 0
) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(
            size=(expert_num,), fill_value=active_num, dtype=group_list.dtype, device=group_list.device
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development."
    )


def _require_single_tensor_for_swiglu_quant(
    tensor_or_list: list[torch.Tensor] | torch.Tensor, *, name: str
) -> torch.Tensor:
    if isinstance(tensor_or_list, list):
        if len(tensor_or_list) != 1:
            raise ValueError(f"{name} must be a tensor or a single-element list, but got {len(tensor_or_list)}.")
        return tensor_or_list[0]
    return tensor_or_list


def quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: list[torch.Tensor] | torch.Tensor,
    w1_scale: list[torch.Tensor] | torch.Tensor,
    w2: list[torch.Tensor] | torch.Tensor,
    w2_scale: list[torch.Tensor] | torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor = None,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    fusion: bool = False,
    dynamic_eplb: bool = False,
    **kwargs,
) -> torch.Tensor:
    # TODO(linfeng): Current massive parameter passing is quite severe; parameter differences introduced by different
    # quantization modes will be consolidated into a dataclass in a follow-up.
    use_mxfp_quant = kwargs.get("use_mxfp_quant", False)
    act_quant_type = torch.float8_e4m3fn
    weight_quant_type = None
    scale_type = None
    per_token_scale_type = None
    use_bf16 = True

    input_hidden_dtype = hidden_states.dtype
    use_gmm_swiglu_quant_fusion = use_mxfp_quant or (fusion and not dynamic_eplb)

    if use_mxfp_quant:
        act_quant_type = kwargs.get("act_quant_type", torch.float8_e4m3fn)
        weight_quant_type = kwargs.get("weight_quant_type", torch.float8_e4m3fn)
        scale_type = kwargs.get("scale_type")
        per_token_scale_type = kwargs.get("per_token_scale_type")
        use_bf16 = kwargs.get("use_bf16", True)

        ensure_mxfp8_moe_available("MXFP MoE MLP path")

        if w1_scale_bias is not None or w2_scale_bias is not None:
            raise NotImplementedError("MXFP path does not support scale_bias yet.")
        if w1_offset is not None or w2_offset is not None:
            raise NotImplementedError("MXFP path does not support antiquant offset yet.")

    if w1_offset is not None:
        unquantized_hidden_states = hidden_states
        quantized_hidden_states = None
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
        unquantized_hidden_states = None
        pertoken_scale = dynamic_scale
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    _output_dtype = w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype

    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(hidden_states)
    is_mc2 = get_forward_context().moe_comm_type == MoECommType.MC2
    if w1_scale_bias is None and w1_offset is None and is_mc2:
        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb) and not use_mxfp_quant:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
            )
        elif use_gmm_swiglu_quant_fusion:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=None,
                use_mxfp_quant=use_mxfp_quant,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            if w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
            # gmm1: gate_up_proj
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
            # act_fn: swiglu
            hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=w1_scale[0],
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=cumsum_group_list(group_list, group_list_type, 1),
                activate_left=True,
                quant_mode=1,
            )
        # gmm2: down_proj
        hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=w2,
            weight_scale=w2_scale,
            per_token_scale=swiglu_out_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            input_dtype=input_hidden_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=use_mxfp_quant,
            bias=None,
            fallback_output_dtype=w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype,
        )
    elif w1_offset is not None:
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[unquantized_hidden_states],
            weight=[w1],
            antiquant_scale=[w1_scale],
            antiquant_offset=[w1_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype,
        )[0]
        dispose_tensor(unquantized_hidden_states)
        # act_fn: swiglu
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            antiquant_scale=[w2_scale],
            antiquant_offset=[w2_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype,
        )[0]
    else:
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat([group_list[:1], torch.diff(group_list, dim=0)])
                group_list_type = 1
            bias1 = [w1_scale_bias] if not fusion else w1_scale_bias
            bias2 = [w2_scale_bias]
            # TODO w4a8 scene: dynamic acquisition of dtype in the future
            _output_dtype = torch.bfloat16

        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb) and not use_mxfp_quant:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                bias=bias1,
            )
        elif use_gmm_swiglu_quant_fusion:
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=bias1,
                use_mxfp_quant=use_mxfp_quant,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            w1_scale[0] = w1_scale[0].to(w2_scale[0].dtype)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                scale=w1_scale,
                bias=bias1,
                per_token_scale=[pertoken_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=_output_dtype,
            )[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            if HAS_TRITON:
                from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant

                hidden_states, swiglu_out_scale = swiglu_quant(
                    hidden_states, group_list=group_list, group_list_type=group_list_type
                )
            else:
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)
        # gmm2: down_proj
        hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=w2,
            weight_scale=w2_scale,
            per_token_scale=swiglu_out_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            input_dtype=input_hidden_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=use_mxfp_quant,
            bias=bias2,
            fallback_output_dtype=_output_dtype,
        )
    return hidden_states


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
) -> torch.Tensor:
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

    if activation == "swigluoai":
        num_experts, _, hidden_size = w1.shape
        gate_up_out = AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up_out.view(-1, hidden_size))
    else:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

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
    return hidden_states


def unified_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor | list[torch.Tensor],
    w2: torch.Tensor | list[torch.Tensor],
    group_list: torch.Tensor,
    w1_scale: list[torch.Tensor] | None = None,
    w2_scale: list[torch.Tensor] | None = None,
    activation: str | None = None,
    w1_bias: torch.Tensor = None,
    w2_bias: torch.Tensor = None,
    dynamic_scale: torch.Tensor = None,
    group_list_type: int = 1,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    topk_scales: torch.Tensor | None = None,
    with_quant: bool = False,
    fusion: bool = False,
    need_trans: bool = True,
    dynamic_eplb: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Unified MoE MLP entry.
    Quant path is dispatched by DeviceOperator with explicit quant-type flags.
    """
    if not with_quant:
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
        )

    assert w1_scale is not None and w2_scale is not None
    # TODO(linfeng): Current massive parameter passing is quite severe; parameter differences introduced by different
    # quantization modes will be consolidated into a dataclass in a follow-up.
    act_quant_type = kwargs.get("act_quant_type", torch.float8_e4m3fn)
    weight_quant_type = kwargs.get("weight_quant_type", torch.float8_e4m3fn)
    scale_type = kwargs.get("scale_type")
    per_token_scale_type = kwargs.get("per_token_scale_type")
    use_mxfp_quant = kwargs.get("use_mxfp_quant", False)
    return quant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        group_list=group_list,
        dynamic_scale=dynamic_scale,
        group_list_type=group_list_type,
        w1_scale_bias=w1_scale_bias,
        w2_scale_bias=w2_scale_bias,
        w1_offset=w1_offset,
        w2_offset=w2_offset,
        fusion=fusion,
        dynamic_eplb=dynamic_eplb,
        act_quant_type=act_quant_type,
        weight_quant_type=weight_quant_type,
        scale_type=scale_type,
        per_token_scale_type=per_token_scale_type,
        use_mxfp_quant=use_mxfp_quant,
        use_bf16=kwargs.get("use_bf16", True),
    )
