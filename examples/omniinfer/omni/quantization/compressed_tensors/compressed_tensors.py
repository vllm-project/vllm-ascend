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
# By using quantization case, this file is called before worker patch achieve,
from typing import Any, Dict, List, Optional, cast
from pydantic import BaseModel
import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target, 
    is_activation_quantization_format,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod, CompressedTensorsKVCacheMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import CompressedTensorsScheme
from omni.layers.moe.fused_moe.layer import FusedMoE
from omni.layers.fused_mlp import FusedMLP, W8A8DynamicFusedMLPMethod
from omni.adaptors.vllm.utils import ASCEND_COMPRESSED_TENSORS
from .schemes.compressed_tensors_w8a8_int8 import AscendCompressedTensorsW8A8Int8LinearMethod
from .schemes.compressed_tensors_w4a8_int8 import AscendCompressedTensorsW4A8Int8LinearMethod
from .compressed_tensors_moe import AscendCompressedTensorsW8A8Int8MoEMethod, AscendCompressedTensorsW4A8Int8MoEMethod


@register_quantization_config(ASCEND_COMPRESSED_TENSORS)
class AscendCompressedTensorsConfig(CompressedTensorsConfig):
    """Config class for Ascend
    
    This class is a general class that parse quantization configs
    that are supported on ascend hardware.
    """

    def __repr__(self) -> str:
        return "AscendCompressedTensorsConfig:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return ASCEND_COMPRESSED_TENSORS

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

    def _get_weight_num_bits(self,
                            layer_name: str,
                            weight_quant: BaseModel) -> bool:
        if isinstance(weight_quant.num_bits, dict):
            for module, module_num_bits in weight_quant.num_bits.items():
                if module in layer_name:
                    return module_num_bits
            raise ValueError(f"weight name mismatch, please check weights num_bits in config.json and model weight name. layer_name={layer_name}")
    
        else:
            return weight_quant.num_bits
    
    
    def _is_dynamic_token_w8a8(self,
                            weight_quant: BaseModel,
                            input_quant: BaseModel,
                            weight_num_bits: int) -> bool:
        is_8_bits = weight_num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value)
        is_token = (weight_strategy and input_quant.strategy
                    == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic
    
        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and weight_quant.symmetric and is_dynamic
    
    
    def _is_dynamic_token_w4a8(self,
                            weight_quant: BaseModel,
                            input_quant: BaseModel,
                            weight_num_bits: int) -> bool:
        is_w4a8_bits = (weight_num_bits == 4) and (input_quant.num_bits == 8)
        weight_strategy = (
                weight_quant.strategy == QuantizationStrategy.TENSOR.value
                or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
                or weight_quant.strategy == QuantizationStrategy.GROUP.value)
        is_token = (weight_strategy and input_quant.strategy
                    == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic
    
        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_w4a8_bits and is_token and weight_quant.symmetric and is_dynamic
    
    
    def _get_scheme_from_parts(
            self,
            layer_name: str,
            weight_quant: BaseModel,
            input_quant: BaseModel) -> "CompressedTensorsScheme":
        # Detect If Activation Quantization.
        # TODO @dsikka: clean-up conditions
        if is_activation_quantization_format(self.quant_format):        
            weight_num_bits = self._get_weight_num_bits(layer_name, weight_quant)
            if self._is_dynamic_token_w8a8(weight_quant, input_quant, weight_num_bits):
                return AscendCompressedTensorsW8A8Int8LinearMethod(
                strategy=weight_quant.strategy,
                is_static_input_scheme=False,
                input_symmetric=input_quant.symmetric)
    
            if self._is_dynamic_token_w4a8(weight_quant, input_quant, weight_num_bits):
                return AscendCompressedTensorsW4A8Int8LinearMethod(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=False,
                    input_symmetric=input_quant.symmetric,
                    group_size=weight_quant.group_size)
    
        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found.")


    def get_scheme(
            self,
            layer: torch.nn.Module,
            layer_name: Optional[str] = None) -> "CompressedTensorsScheme":
        """
        compressed-tensors supports non uniform in the following way:
    
        ignore: List of layer_names or nn.Module names to be ignored.
        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.
    
        We first check whether a layer is in the ignore group and use
        CompressedTensorsUnquantized (i.e. fp16/bf16) scheme for the layer
    
        We then detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for infernece.
        """
    
        # Find the "target" in the compressed-tensors config
        # that our layer conforms to.
        # so we do not have to re-write these functions
        # need to make accelerate optional in ct to do this
        matched_target = find_matched_target(
            layer_name=layer_name,
            module=layer,
            targets=self.target_scheme_map.keys())
    
        # Find the quant_scheme
        scheme_dict = self.target_scheme_map[matched_target]
    
        # Adapter: pass layer_name
        scheme = self._get_scheme_from_parts(
            layer_name=layer_name,
            weight_quant=scheme_dict["weights"],
            input_quant=scheme_dict["input_activations"])    
        return scheme

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedTensorsConfig":
        cls.quant_description = config
        target_scheme_map: Dict[str, Any] = dict()
        ignore = cast(List[str], config.get("ignore"))
        quant_format = cast(str, config.get("format"))
    
        # The quant_config has multiple config_groups, each containing
        # an input_activations key with details about how the activations are
        # quantized, a weights key indicating how the weights are quantized,
        # and a list of targets under the `targets` key, dictating which
        # layers are impacted by the quantization details. The quantization
        # details follow the structure defined by the QuantizationArgs
        # pydantic model, which is used to verify the structure of the
        # quant_config and also store the details for later use.
        for _, quant_config in config["config_groups"].items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                # adapt: do not validate parameters
                module_num_bits = quant_config.get("weights").get("num_bits")
                quant_config["weights"]["num_bits"] = 0
                target_scheme_map[target][
                    "weights"] = QuantizationArgs.parse_obj(quant_config.get("weights"))
                quant_config["weights"]["num_bits"] = module_num_bits
                target_scheme_map[target]["weights"].num_bits = module_num_bits
                try:
                    target_scheme_map[target][
                        "input_activations"] = QuantizationArgs.parse_obj(
                        quant_config.get("input_activations"))
                except Exception:
                    target_scheme_map[target]["input_activations"] = None
    
        return cls(target_scheme_map=target_scheme_map,
                ignore=ignore,
                quant_format=quant_format,
                kv_cache_scheme=config.get("kv_cache_scheme"),
                sparsity_scheme_map=None,
                sparsity_ignore_list=None)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        quant_method = hf_quant_cfg['quant_method']
        if torch.npu.is_available() and quant_method == 'compressed-tensors':
            return ASCEND_COMPRESSED_TENSORS
        return None

    def get_moe_method(self, prefix):
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = self.target_scheme_map["Linear"].get("weights")
        input_quant = self.target_scheme_map["Linear"].get(
            "input_activations")

        weight_num_bits = self._get_weight_num_bits("mlp.experts", weight_quant)
        if self._is_dynamic_token_w8a8(weight_quant, input_quant, weight_num_bits):
            return (AscendCompressedTensorsW8A8Int8MoEMethod(), weight_num_bits)
        elif self._is_dynamic_token_w4a8(weight_quant, input_quant, weight_num_bits):
            return (AscendCompressedTensorsW4A8Int8MoEMethod(self), weight_num_bits)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            layer.scheme = scheme
            return AscendCompressedTensorsLinearMethod(self)
        elif isinstance(layer, FusedMLP):
            return W8A8DynamicFusedMLPMethod(self)
        elif isinstance(layer, FusedMoE):
            layer.weight_num_bits = 0
            moe_method, weight_num_bits = self.get_moe_method(prefix)
            layer.weight_num_bits = weight_num_bits
            return moe_method
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class AscendCompressedTensorsLinearMethod(CompressedTensorsLinearMethod):
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              module_name: Optional[str] = "",
              x_transform: Optional[str] = None,
              is_prefill: Optional[bool] = True
              ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias, module_name=module_name,
                                    x_transform=x_transform, is_prefill=is_prefill)
