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
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch_npu  # noqa: F401
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
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

from .quantizer import AscendQuantizer

logger = init_logger(__name__)


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
        return []

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
        if isinstance(layer, Attention) and \
            'fa_quant_type' in self.quant_description.keys():
            return AscendKVCacheMethod(self, prefix)
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
