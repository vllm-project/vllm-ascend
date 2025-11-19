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
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.experts_selector import select_experts

GROUP_SIZE = 32


class AscendW8A8MXFP8DynamicLinearMethod:
    """ Liner method for Ascend W8A8_DYNAMIC
    """
    model_dtype = None

    def __init__(self):
        self.transpose_weight = True

    @staticmethod
    def get_weight(
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
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

    def get_pergroup_param(self,
                           input_size: int,
                           output_size: int,
                           params_dtype: torch.dtype,
                           layer_type: Optional[str] = None) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(
            output_size, input_size // GROUP_SIZE, dtype= torch.uint8)
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:

        config = getattr(layer, "_ascend_quant_config", {})
        if not isinstance(x, tuple):
            if AscendW8A8MXFP8DynamicLinearMethod.model_dtype is None:
                AscendW8A8MXFP8DynamicLinearMethod.model_dtype = x.dtype
            quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        else:
            quantized_x, dynamic_scale = x
        pertoken_scale = (dynamic_scale if config.get("pertoken_scale", True) else None)
        has_dequant = True if config.get("pertoken_scale", True) else False
        output_dtype = AscendW8A8MXFP8DynamicLinearMethod.model_dtype if has_dequant else torch.float32
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=output_dtype,
            group_sizes=[1,1,GROUP_SIZE]
        )
        return output

    def process_weights_after_loading(self, layer):
        n_dim, k_dim = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim//2, 2)
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0,1)


class AscendW8A8MXFP8DynamicFusedMoEMethod:
    """FusedMoe method for Ascend W8A8_DYNAMIC.
    """
    model_dtype = None

    def __init__(self):
        self.transpose_weight = True
        self.ep_group = get_ep_group()
        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.level == CompilationLevel.PIECEWISE and
            not vllm_config.model_config.enforce_eager and
            not ascend_config.torchair_graph_config.enabled)
        self.dynamic_eplb = ascend_config.dynamic_eplb or ascend_config.expert_map_record_path



    @staticmethod
    def get_weight(num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(num_experts,
                                               2 *
                                               intermediate_size_per_partition,
                                               hidden_sizes,
                                               dtype=torch.float8_e4m3fn,
                                               requires_grad=False)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                              hidden_sizes,
                                              intermediate_size_per_partition,
                                              dtype=torch.float8_e4m3fn,
                                              requires_grad=False)
        return param_dict

    @staticmethod
    def get_dynamic_quant_param(num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // GROUP_SIZE,
            dtype=torch.uint8)

        param_dict["w2_weight_scale"] = torch.empty(num_experts,
                                                    hidden_sizes,
                                                    intermediate_size_per_partition // GROUP_SIZE,
                                                    dtype=torch.uint8)

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
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
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

        if enable_force_load_balance:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)
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
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            dynamic_eplb=self.dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask", None),
            use_A5_quant=True,
            use_fp8_comm = True,
            act_quant_type = torch.float8_e4m3fn,
            weight_quant_type=torch.float8_e4m3fn,
            scale_type=torch.float8_e8m0fnu,
        per_token_scale_type=torch_npu.float8_e8m0fnu
            comm_quant_mode=4)

    def process_weights_after_loading(self, layer):
        g_num, n_size, k_size = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g_num, n_size, k_size//2,2)
        g_num, n_size, k_size= layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g_num, n_size, k_size//2,2)
        if self.transpose_weight:
            layer.w13_weight.data = layer.w13_weight.data.transpose(1,2)
            layer.w2_weight.data = layer.w2_weight.data.transpose(1,2)
            layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1,2)
            layer.w2_weight_scale.data =  layer.w2_weight_scale.data.transpose(1,2)
