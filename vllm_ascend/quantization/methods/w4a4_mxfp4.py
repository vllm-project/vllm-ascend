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


from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    ensure_mxfp4_linear_available,
    ensure_mxfp4_moe_available,
)

from vllm_ascend.quantization.quant_parser import get_rollback_quant_type

from .base import AscendLinearScheme, AscendMoEScheme, QuantType
from .registry import register_scheme


@register_scheme("W4A4_MXFP4", "linear")
class AscendW4A4MXFP4DynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A4_MXFP4 (Microscaling FP4) quantization.
    """
    def __init__(self):
        self.transpose_weight = True
        ensure_mxfp4_linear_available("W4A4_MXFP4 linear quantization")
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)


    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(
            output_size: int,
            params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        return {}

    def get_pergroup_param(self, input_size: int, output_size: int,
                           params_dtype: torch.dtype, layer_type: Optional[str] = None) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(
            output_size, input_size // self.group_size, dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:

        quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2, round_mode="round")

        pertoken_scale = dynamic_scale
        output_dtype = x.dtype

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            bias=bias,
            output_dtype=output_dtype,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            group_sizes=[1, 1, self.group_size]
        )

        return output

    def process_weights_after_loading(self, layer):
        layer.weight.data = torch_npu.npu_dtype_cast(layer.weight.data, torch_npu.float4_e2m1fn_x2)
        n_dim, k_dim = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim//2, 2)
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1) # 图模式需要有


@register_scheme("W4A4_MXFP4", "moe")
class AscendW4A4MXFP4DynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoe method for Ascend W4A4_MXFP4 (Microscaling FP4) quantization.
    """
    def __init__(self, additional_quant_config=None):
        ensure_mxfp4_moe_available("W4A4_MXFP4 MoE quantization")
        self.ep_group = get_ep_group()
        
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb
        self.additional_quant_config = additional_quant_config


    @staticmethod
    def get_weight(num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(num_experts,
                                               2 * intermediate_size_per_partition,
                                               hidden_sizes,
                                               dtype=torch.float8_e4m3fn)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                              hidden_sizes,
                                              intermediate_size_per_partition,
                                              dtype=torch.float8_e4m3fn)
        return param_dict

    def get_dynamic_quant_param(self, num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
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
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
                   1] == global_num_experts - global_redundant_expert_num, "Number of global experts mismatch (excluding redundancy)"
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
            global_num_experts=global_num_experts)

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
            hidden_states=x,
            w1=layer.w13_weight,
            w1_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_int8_w8a8=False,
            expert_map=expert_map,
            log2phy=log2phy,
            dynamic_eplb=self.dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask", None),
            use_mxfp_quant=True,
            act_quant_type=torch_npu.float4_e2m1fn_x2,
            weight_quant_type=torch_npu.float4_e2m1fn_x2,
            scale_type=FLOAT8_E8M0FNU_DTYPE,
            per_token_scale_type=FLOAT8_E8M0FNU_DTYPE,
            round_mode="round",
            rollback_quant_config=self.additional_quant_config
        )

    def process_weights_after_loading(self, layer):
        layer.w13_weight.data = torch_npu.npu_dtype_cast(layer.w13_weight.data, torch_npu.float4_e2m1fn_x2)
        if self.additional_quant_config is None: # moe层gate\up\down都是w4a4
            layer.w2_weight.data = torch_npu.npu_dtype_cast(layer.w2_weight.data, torch_npu.float4_e2m1fn_x2)
            rollback_quant_type = None
        else:
            rollback_quant_type = get_rollback_quant_type(self.additional_quant_config) # W4A8_MXFP
            if rollback_quant_type == "W4A8_MXFP":
                layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data.to(torch.float32), 29, torch.float8_e4m3fn) # cast nz
                layer.w2_weight.data = torch_npu.npu_convert_weight_to_int4pack(layer.w2_weight.data)
            else:
                raise ValueError(f"rollback_quant_type {rollback_quant_type} is not supported")
        g_num, n_size, k_size = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g_num, n_size, k_size//2, 2)
        g_num, n_size, k_size = layer.w2_weight_scale.shape
        if rollback_quant_type != "W4A8_MXFP": # W4A8mxfp scale是两维
            layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g_num, n_size, k_size//2, 2)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2)
