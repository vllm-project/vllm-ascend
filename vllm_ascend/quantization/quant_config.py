#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
import torch_npu  # noqa: F401
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.layer import \
    UnquantizedFusedMoEMethod
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs

from .quantizer import AscendQuantizer


@register_quantization_config("ascend")
class AscendQuantConfig(QuantizationConfig):
    """Config class for Ascend
    
    This class is a general class that parse quantization configs
    that are supported on ascend hardware.
    """

    def __init__(self, quant_config: Dict[str, Any]):
        self.quant_description = quant_config

    def __repr__(self) -> str:
        return "AscendQuantConfig:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return "ascend"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_model_description.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AscendQuantConfig":
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        if torch.npu.is_available():
            return "ascend"
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return AscendLinearMethod(self, prefix,
                                      self.packed_modules_mapping)
        elif isinstance(layer, Attention) and \
            'fa_quant_type' in self.quant_description.keys() and \
            self.quant_description['fa_quant_type'] is not None:
            return AscendKVCacheMethod(self, prefix)
        elif isinstance(layer, FusedMoE):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return UnquantizedFusedMoEMethod()
            return AscendFusedMoEMethod(self, prefix,
                                        self.packed_modules_mapping)
        return None

    def is_layer_skipped_ascend(
        self,
        prefix: str,
        fused_mapping: Mapping[str, List[str]] = MappingProxyType({})):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = self.quant_description[shard_prefix +
                                                          '.weight'] == "FLOAT"

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision.")
        else:
            is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []


class AscendLinearMethod(LinearMethodBase):
    """Linear method for Ascend quantization.

    This class calls AscendQuantizer to search a specific quantization
    implementations supported on ascend hardware for linear methods.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str,
                 packed_modules_mapping: Dict[str, Any]) -> None:
        self.quantizer = AscendQuantizer.get_quantizer(
            quant_config.quant_description, prefix, packed_modules_mapping)
        self.quant_method = self.quantizer.build_linear_method()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = self.quant_method.get_weight(input_size_per_partition,
                                                   output_size_per_partition,
                                                   params_dtype)
        for weight_name, weight_param in weight_dict.items():
            layer.register_parameter(
                weight_name,
                ModelWeightParameter(data=weight_param,
                                     input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader))

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(data=pertensor_param,
                                            weight_loader=weight_loader)
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)

        perchannel_dict = self.quant_method.get_perchannel_param(
            output_size_per_partition, params_dtype)
        for perchannel_name, perchannel_param in perchannel_dict.items():
            layer.register_parameter(
                perchannel_name,
                ChannelQuantScaleParameter(data=perchannel_param,
                                           output_dim=0,
                                           weight_loader=weight_loader))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(layer, RowParallelLinear):
            tp_rank = get_tensor_model_parallel_rank()
            return self.quant_method.apply(layer, x, bias, tp_rank)
        return self.quant_method.apply(layer, x, bias)


class AscendKVCacheMethod(BaseKVCacheMethod):
    """KVCache method for Ascend quantization.

    This class calls AscendQuantizer to search a specific quantization
    implementations supported on ascend hardware for kvcache methods.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str) -> None:
        self.quantizer = AscendQuantizer.get_quantizer(
            quant_config.quant_description, prefix)
        self.quant_method = self.quantizer.build_attention_method()

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Different from linear method, there are no weight processing/slicing
        # steps for attention in vllm. So the whole process of create weights
        # is hidden into the specific quant method.
        self.quant_method.create_weights(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(self,
              layer: torch.nn.Module,
              query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              k_cache: List[torch.Tensor],
              v_cache: List[torch.Tensor],
              scale: torch.Tensor,
              block_tables: torch.Tensor,
              isPrefill: bool,
              attn_metadata,
              output,
              seq_lens_tensor_cpu: Optional[int] = None) -> torch.Tensor:
        return self.quant_method.apply(layer,
                                       query,
                                       key,
                                       value,
                                       k_cache,
                                       v_cache,
                                       scale,
                                       block_tables,
                                       isPrefill,
                                       attn_metadata.attn_mask,
                                       attn_metadata.slot_mapping,
                                       output,
                                       seq_lens_tensor_cpu=seq_lens_tensor_cpu)


def fused_moe_perchannel_weight_loader(param: torch.nn.Parameter,
                                       loaded_weight: torch.Tensor,
                                       weight_name: str, shard_id: str,
                                       expert_id: int) -> None:

    if shard_id not in ("w1", "w2", "w3"):
        raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                         f"got {shard_id}.")

    # Fetch the dim to shard the parameter/loaded weight
    # based on the shard id. This will be whatever
    # dimension intermediate_size_per_partition is used.
    SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

    expert_data = param.data[expert_id]
    tp_rank = get_tensor_model_parallel_rank()

    # is_transposed: if the dim to shard the weight
    # should be flipped. Required by GPTQ, compressed-tensors
    # should be whatever dimension intermediate_size_per_partition is
    is_transposed = getattr(param, "is_transposed", False)
    shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
    if is_transposed:
        shard_dim = int(not shard_dim)

    if shard_id == "w2":
        expert_data.copy_(loaded_weight)
    elif shard_id in ("w1", "w3"):
        shard_size = expert_data.shape[shard_dim] // 2
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank,
                                             shard_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)


class AscendFusedMoEMethod(FusedMoEMethodBase):
    """FusedMoE method for Ascend quantization.

    This class calls AscendQuantizer to search a specific quantization
    implementations supported on ascend hardware for kvcache methods.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str,
                 packed_modules_mapping: Dict[str, Any]):
        self.quantizer = AscendQuantizer.get_quantizer(
            quant_config.quant_description, prefix, packed_modules_mapping)
        self.quant_method = self.quantizer.build_moe_method()

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
            num_experts, intermediate_size_per_partition, hidden_size,
            params_dtype)
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        # load `offset` weight in `fused_moe_perchannel_weight_loader`, the original weight load in vllm 0.7.3 could only load `scale` and `zero`
        extra_weight_attrs.update(
            {"weight_loader": fused_moe_perchannel_weight_loader})
        dynamic_quant_param = self.quant_method.get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_size,
            params_dtype)
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, x, use_grouped_topk, top_k,
                                       router_logits, renormalize, topk_group,
                                       num_expert_group,
                                       custom_routing_function, scoring_func,
                                       e_score_correction_bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)
