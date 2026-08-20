#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#


from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_utils import cumsum_group_list, maybe_normalize_mxfp_scale_layout
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: F401
from vllm_ascend.utils import FP8_METHOD, dispose_tensor

from ..base import (
    AscendLinearScheme,
    AscendMoEScheme,
    QuantType,
    TPWeightGatherSpec,
)
from ..registry import register_scheme

# CANN uses 36 to select FP8 E4M3FN output for situ_mx_quant.
SITU_MX_DST_TYPE_E4M3FN = 36


@register_scheme("W4A8_MXFP", "linear")
class AscendW4A8MXFPDynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A8_MXFP (Microscaling) quantization."""

    tp_weight_gather_specs = (
        TPWeightGatherSpec("weight"),
        TPWeightGatherSpec("weight_scale"),
    )
    tp_weight_output_gather_specs = (
        TPWeightGatherSpec("weight", gather_dim=1),
        TPWeightGatherSpec("weight_scale", gather_dim=1),
    )
    supports_tp_weight_switch = True

    def __init__(self):
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)

    @staticmethod
    def get_weight(input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, input_size // self.group_size, dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            quantized_x, dynamic_scale = x
            output_dtype = torch.bfloat16
        else:
            quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            output_dtype = x.dtype

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=output_dtype,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            group_sizes=[0, 0, self.group_size],
        )

        return output

    def process_weights_after_loading(self, layer):
        layer.weight.data = torch_npu.npu_format_cast(
            layer.weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
        layer.weight.data = layer.weight.data.transpose(-1, -2)
        n, k = layer.weight_scale.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n, k // 2, 2).transpose(-3, -2)


@register_scheme("W4A8_MXFP", "moe")
class AscendW4A8MXFPDynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoe method for Ascend W4A8_DYNAMIC."""

    supports_eplb = False
    quant_type: QuantType = QuantType.W4A8MXFP
    act_quant_type: torch.dtype = torch.float8_e4m3fn
    fused_activations = frozenset({"silu", "situ"})

    def __init__(self):
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        ascend_config = get_ascend_config()
        self.dynamic_eplb = False if vllm_config.use_v2_model_runner else ascend_config.eplb_config.dynamic_eplb

    @staticmethod
    def get_weight(
        num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // 2, dtype=torch.uint8
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // 2, dtype=torch.uint8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.uint8
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.uint8
        )
        return param_dict

    def apply(
        self,
        layer: "AscendRoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.dtype not in [torch.float8_e4m3fn]:
            topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=layer.activation,
                mxfp_act_quant_type=torch.float8_e4m3fn,
                mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
                mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_use_bf16=(x.dtype in [torch.bfloat16, torch.float8_e4m3fn]),
            ),
            quant_method=self,
        )

    def get_fused_mc2_weights(self, layer: torch.nn.Module) -> MoEWeights:
        if _EXTRA_CTX.use_mega_moe:
            # MegaMoe consumes the non-transposed per-expert weight/scale lists
            # built in process_weights_after_loading (the non-mega path uses the
            # transposed single tensors below).
            return MoEWeights(
                w1=layer.w13_weight.data,
                w2=layer.w2_weight.data,
                w1_scale=layer.w13_weight_scale.data,
                w2_scale=layer.w2_weight_scale.data,
                w1_scale_bias=None,
                w2_scale_bias=None,
            )
        else:
            return MoEWeights(
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_scale_bias=None,
                w2_scale_bias=None,
            )

    def process_weights_after_loading(self, layer):
        layer.w13_weight.data = torch_npu.npu_format_cast(
            layer.w13_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
        layer.w2_weight.data = torch_npu.npu_format_cast(
            layer.w2_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        g, n, k = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)
        g, n, k = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)

    def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput):
        hidden_states = mlp_compute_input.hidden_states
        hidden_states, pertoken_scale = self._quant_hidden_states(hidden_states, mlp_compute_input.dynamic_scale)
        layer = mlp_compute_input.layer
        assert layer is not None
        if mlp_compute_input.activation == MoEActivation.SITU:
            # SituAndMul: run the dequantized gmm1 first, then fuse the situ
            # activation with MXFP output quantization (Kimi K3 SITU activation).
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w13_weight],
                scale=None,
                antiquant_scale=[layer.w13_weight_scale],
                scale_dtype=None,
                per_token_scale=[pertoken_scale],
                per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                split_item=2,
                group_type=0,
                group_list=mlp_compute_input.group_list,
                group_list_type=mlp_compute_input.group_list_type,
                x_dtype=torch.float8_e4m3fn,
                weight_dtype=torch_npu.float4_e2m1fn_x2,
                output_dtype=torch.bfloat16,
            )[0]
            hidden_states, swiglu_out_scale = torch.ops._C_ascend.situ_mx_quant(
                x=hidden_states,
                beta=1.0 if mlp_compute_input.activation_situ_beta is None else mlp_compute_input.activation_situ_beta,
                linear_beta=mlp_compute_input.activation_situ_linear_beta or 0.0,
                activate_left=True,
                dst_type=SITU_MX_DST_TYPE_E4M3FN,
            )
            dispose_tensor(mlp_compute_input.hidden_states)
            return hidden_states, maybe_normalize_mxfp_scale_layout(swiglu_out_scale)

        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            scale=None,
            antiquant_scale=[layer.w13_weight_scale],
            scale_dtype=None,
            per_token_scale=[pertoken_scale],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=cumsum_group_list(mlp_compute_input.group_list, mlp_compute_input.group_list_type, 0),
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            output_dtype=torch.bfloat16,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        hidden_states, out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
            hidden_states,
            topk_weight=None,
            group_index=None,
            dst_type=torch.float8_e4m3fn,
            quant_mode=2,
            clamp_value=mlp_compute_input.swiglu_limit,
        )
        return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput):
        hidden_states = mlp_compute_input.hidden_states
        hidden_states, pertoken_scale = self._quant_hidden_states(hidden_states, mlp_compute_input.dynamic_scale)
        layer = mlp_compute_input.layer
        assert layer is not None
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            scale=[layer.w13_weight_scale],
            per_token_scale=[pertoken_scale],
            bias=None,
            split_item=2,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            output_dtype=torch.bfloat16,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states

    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
        return hidden_states, maybe_normalize_mxfp_scale_layout(dynamic_scale)

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        layer = mlp_compute_input.layer
        assert layer is not None
        input_dtype = mlp_compute_input.hidden_states.dtype
        use_bf16 = input_dtype in [torch.bfloat16, torch.float8_e4m3fn]
        output_dtype = (
            input_dtype
            if input_dtype in [torch.bfloat16, torch.float16]
            else (torch.bfloat16 if use_bf16 else torch.float16)
        )
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=None,
            antiquant_scale=[layer.w2_weight_scale],
            bias=None,
            per_token_scale=[act_out_scale],
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            output_dtype=output_dtype,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
        )[0]


@register_scheme(FP8_METHOD, "ds_w4a8_moe")
class AscendW4A8MXFPDSDynamicFusedMoEMethod(AscendW4A8MXFPDynamicFusedMoEMethod):
    """FusedMoe method for DS original w4a8 mxfp quantization."""

    model_dtype = None
    quant_type: QuantType = QuantType.W4A8MXFP

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.group_size,
            dtype=torch.float8_e8m0fnu,
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.float8_e8m0fnu
        )
        return param_dict

    def process_weights_after_loading(self, layer):
        layer.w13_weight.data = torch_npu.npu_format_cast(
            layer.w13_weight.data.view(torch.uint8),
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )
        layer.w2_weight.data = torch_npu.npu_format_cast(
            layer.w2_weight.data.view(torch.uint8),
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        g, n, k = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = (
            layer.w13_weight_scale.data.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)
        )
        g, n, k = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = (
            layer.w2_weight_scale.data.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)
        )
