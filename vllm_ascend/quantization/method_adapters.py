#
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
#

from collections.abc import Callable

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_flashcomm2_otp_group, get_mlp_tp_group, get_otp_group
from vllm_ascend.utils import flashcomm2_enable, mlp_tp_enable, oproj_tp_enable

from .methods import AscendAttentionScheme, AscendLinearScheme, AscendMoEScheme, is_mx_quant_type


class AscendLinearMethod(LinearMethodBase):
    """Linear method for Ascend quantization.

    This wrapper class delegates to the actual quantization scheme implementation.
    The scheme is determined by the Config class and passed directly to this wrapper.

    Args:
        scheme: The quantization scheme instance (e.g., AscendW8A8DynamicLinearMethod).
    """

    def __init__(self, scheme: AscendLinearScheme) -> None:
        self.quant_method = scheme

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = self.quant_method.get_weight(input_size_per_partition, output_size_per_partition, params_dtype)

        # Extract packing information (if present)
        packed_dim = weight_dict.pop("_packed_dim", None)
        packed_factor = weight_dict.pop("_packed_factor", None)

        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})

            # Set packing attributes if the weight is packed
            if packed_dim is not None and packed_factor is not None:
                set_weight_attrs(param, {"packed_dim": packed_dim, "packed_factor": packed_factor})

            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(data=pertensor_param, weight_loader=weight_loader)
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)
            param.weight_loader = extra_weight_attrs.get("weight_loader")

        perchannel_dict = self.quant_method.get_perchannel_param(output_size_per_partition, params_dtype)
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        # NOTE: In w4a8 quantization implementation,
        # for down_proj and o_proj scale_bias shape is [output_size, 16],
        # others are [output_size, 1]
        layer_type = "row" if isinstance(layer, RowParallelLinear) else "others"

        pergroup_dict = self.quant_method.get_pergroup_param(
            input_size_per_partition, output_size_per_partition, params_dtype, layer_type=layer_type
        )
        for pergroup_name, pergroup_param in pergroup_dict.items():
            param = torch.nn.Parameter(pergroup_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(pergroup_name, param)
            set_weight_attrs(param, extra_weight_attrs)
            if (
                "weight_scale_second" in pergroup_name
                or "weight_offset_second" in pergroup_name
                or is_mx_quant_type(self.quant_method)
            ):
                param.input_dim = 1
                param.input_dim = 1

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def get_computed_params(self) -> set[str]:
        """Return parameter name patterns that are computed, not loaded.

        These parameters are computed during process_weights_after_loading
        rather than loaded from checkpoint:
        - weight_offset: Zero for symmetric quantization
        - quant_bias: Computed from weight statistics
        - deq_scale: Computed as input_scale * weight_scale
        - weight_scale: May be computed or have default values for some models
        """
        return {"weight_offset", "quant_bias", "deq_scale", "weight_scale"}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(layer, RowParallelLinear):
            if layer.prefix.find("o_proj") != -1 and oproj_tp_enable():
                tp_rank = get_otp_group().rank_in_group
            elif layer.prefix.find("down_proj") != -1 and mlp_tp_enable():
                tp_rank = get_mlp_tp_group().rank_in_group
            elif (layer.prefix.find("o_proj") != -1 or layer.prefix.find("out_proj") != -1) and flashcomm2_enable():
                if get_ascend_config().flashcomm2_oproj_tensor_parallel_size == 1:
                    tp_rank = 0
                else:
                    tp_rank = get_flashcomm2_otp_group().rank_in_group
            else:
                tp_rank = get_tensor_model_parallel_rank()
        else:
            tp_rank = 0
        return self.quant_method.apply(layer, x, bias, tp_rank)


class AscendKVCacheMethod(BaseKVCacheMethod):
    """KVCache method for Ascend quantization.

    This wrapper class delegates to the actual attention quantization scheme.

    Args:
        scheme: The attention quantization scheme instance.
    """

    def __init__(self, scheme: AscendAttentionScheme) -> None:
        self.quant_method = scheme

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Different from linear method, there are no weight processing/slicing
        # steps for attention in vllm. So the whole process of create weights
        # is hidden into the specific quant method.
        self.quant_method.create_weights(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, query, key, value, kv_cache, attn_metadata, attn_type, scale, output)


class AscendFusedMoEMethod(FusedMoEMethodBase):
    """FusedMoE method for Ascend quantization.

    This wrapper class delegates to the actual MoE quantization scheme.

    Args:
        scheme: The MoE quantization scheme instance.
        moe_config: The FusedMoE configuration.
    """

    def __init__(self, scheme: AscendMoEScheme, moe_config: FusedMoEConfig) -> None:
        super().__init__(moe_config)
        self.quant_method = scheme

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_param = self.quant_method.get_weight(
            num_experts, intermediate_size_per_partition, hidden_size, params_dtype
        )
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        per_group_param = (
            ["weight_scale_second", "weight_offset_second", "scale_bias"] + ["weight_scale", "weight_offset"]
            if hasattr(self.quant_method, "group_size") and self.quant_method.group_size > 0
            else []
        )
        dynamic_quant_param = self.quant_method.get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_size, params_dtype
        )
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)
            if any(fields in param_key for fields in per_group_param):
                param.quant_method = FusedMoeWeightScaleSupported.GROUP.value

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
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num=0,
        **kwargs,
    ) -> torch.Tensor:
        return self.quant_method.apply(
            layer,
            x,
            router_logits,
            top_k,
            renormalize,
            use_grouped_topk,
            global_num_experts,
            expert_map,
            topk_group,
            num_expert_group,
            custom_routing_function,
            scoring_func,
            routed_scaling_factor,
            e_score_correction_bias,
            is_prefill,
            enable_force_load_balance,
            log2phy,
            global_redundant_expert_num,
            **kwargs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        pass

    @property
    def supports_eplb(self):
        supports_eplb = getattr(self.quant_method, "supports_eplb", False)
        return supports_eplb


class AscendEmbeddingMethod(AscendLinearMethod):
    """Embedding method for Ascend quantization.

    This is essentially the same as AscendLinearMethod, just with a different name
    for clarity when used with VocabParallelEmbedding layers.

    Args:
        scheme: The quantization scheme instance.
    """

    def __init__(self, scheme: AscendLinearScheme) -> None:
        self.quant_method = scheme
