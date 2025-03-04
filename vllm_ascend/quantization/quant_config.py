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
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           ModelWeightParameter)

from .quantizer import AscendQuantizer

logger = init_logger(__name__)


@register_quantization_config("ascend")
class AscendQuantConfig(QuantizationConfig):
    """Config class for Ascend"""

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
            return AscendLinearMethod(self)
        if isinstance(layer, Attention) and \
            'fa_quant_type' in self.quant_description.keys():
            return AscendQKVQuantAttentionMethod(self)
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

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig) -> None:
        self.quantizer = AscendQuantizer.get_quantizer(
            quant_config.quant_description)
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
        del output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weights = self.quant_method.create_weights(input_size_per_partition,
                                                   output_size_per_partition,
                                                   params_dtype)

        weight_name = self.quant_method.get_weight()
        if weight_name in weights.keys():
            layer.register_parameter(
                weight_name,
                ModelWeightParameter(data=weights[weight_name].transpose(0, 1),
                                     input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader))
        else:
            raise ValueError(
                f"{weight_name} is nor registered. Please check your linear quant method implementation."
            )

        pertensor_names = self.quant_method.get_pertensor_param()
        for pertensor_name in pertensor_names:
            if pertensor_name in weights.keys():
                param = BasevLLMParameter(data=weights[pertensor_name],
                                          weight_loader=weight_loader)
                # disable warning
                param.ignore_warning = True
                layer.register_parameter(pertensor_name, param)
            else:
                raise ValueError(
                    f"{pertensor_name} is nor registered. Please check your linear quant method implementation."
                )

        perchannel_names = self.quant_method.get_perchannel_param()
        for perchannel_name in perchannel_names:
            if perchannel_name in weights.keys():
                layer.register_parameter(
                    perchannel_name,
                    ChannelQuantScaleParameter(data=weights[perchannel_name],
                                               output_dim=0,
                                               weight_loader=weight_loader))
            else:
                raise ValueError(
                    f"{perchannel_name} is nor registered. Please check your linear quant method implementation."
                )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method,
                   'transpose_weight') and self.quant_method.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(1, 0)

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


class AscendQKVQuantAttentionMethod(BaseKVCacheMethod):
    """Linear method for Ascend quantization.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig) -> None:
        self.quantizer = AscendQuantizer.get_quantizer(
            quant_config.quant_description)
        self.quant_method = self.quantizer.build_attention_method()

    def create_weights(self, layer: torch.nn.Module) -> None:
        self.quant_method.create_weights(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(self, layer: torch.nn.Module, query: torch.Tensor,
              key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor,
              value_cache: torch.Tensor, scale: torch.Tensor,
              seq_lens_tensor_cpu: int, block_tables: torch.Tensor,
              isPrefill: bool, attn_metadata, output) -> torch.Tensor:
        return self.quant_method.apply(layer, query, key, value, key_cache,
                                       value_cache, scale, seq_lens_tensor_cpu,
                                       block_tables, isPrefill, attn_metadata,
                                       output)
