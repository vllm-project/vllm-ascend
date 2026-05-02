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


from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config

from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    ensure_mxfp4_linear_available,
)
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendLinearScheme, QuantType
from .registry import register_scheme


@register_scheme("W4A8_MXFP", "linear")
class AscendW4A8MXFPDynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A8_MXFP (Microscaling) quantization."""
    def __init__(self):
        ensure_mxfp4_linear_available("W8A8_MXFP8 linear quantization")
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)


    @staticmethod
    def get_weight(input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        
        params_dict = {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}
        
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(
            output_size: int,
            params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {}

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

        quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

        x2_scale = (
            layer.weight_scale.transpose(-1, -2).reshape(-1, layer.weight_scale.shape[0] // 2, 2).transpose(-3, -2)
        )
        output_dtype = x.dtype

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            x2_scale,
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
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)


@register_scheme("W4A8_MXFP", "moe")
class AscendW4A8MXFPDynamicFusedMoEMethod:
    """FusedMoe method for Ascend W4A8_DYNAMIC."""
    quant_type: QuantType = QuantType.W4A8MXFP

    def __init__(self):
        self.ep_group = get_ep_group()

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb
        self.additional_quant_config = None

    def update_addtional_quant_config(self, addtional_quant_config: dict):
        self.additional_quant_config = addtional_quant_config


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
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert router_logits.shape[1] == global_num_experts - global_redundant_expert_num, (
            "Number of global experts mismatch (excluding redundancy)"
        )
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts,
        )

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(
                topk_ids.size(0), global_num_experts - global_redundant_expert_num, device=topk_ids.device
            )
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                mxfp_act_quant_type=torch.float8_e4m3fn,
                mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
                mxfp_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_use_bf16=(x.dtype == torch.bfloat16),
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
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
