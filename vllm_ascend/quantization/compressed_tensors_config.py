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
"""LLM-Compressor (compressed_tensors) quantization configuration for Ascend."""

from typing import Any, Optional, cast

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy, QuantizationType
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    is_activation_quantization_format,
    should_ignore_layer,
)
from vllm.model_executor.models.utils import WeightsMapper

from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD

from .methods import AscendLinearScheme, AscendMoEScheme

logger = init_logger(__name__)


# Remove the original compressed_tensors method to replace with our implementation
def _remove_quantization_method():
    if COMPRESSED_TENSORS_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(COMPRESSED_TENSORS_METHOD)


_remove_quantization_method()

QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, "QuantizationArgs"] | None]


@register_quantization_config(COMPRESSED_TENSORS_METHOD)
class AscendCompressedTensorsConfig(QuantizationConfig):
    """Config class for LLM-Compressor (compressed_tensors) quantization on Ascend.

    This class adapts the compressed_tensors format to work with Ascend's
    quantization implementations.
    """

    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        # Map from [target -> scheme]
        self.target_scheme_map = target_scheme_map
        self.quant_description = config

    def get_name(self) -> str:
        return "compressed-tensors"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _add_fused_moe_to_target_scheme_map(self):
        """
        Helper function to update target_scheme_map
        since linear layers get fused into FusedMoE
        targeting 'Linear' needs to also match
        FusedMoE modules.
        """
        if "Linear" not in self.target_scheme_map or "FusedMoE" in self.target_scheme_map:
            return
        self.target_scheme_map["FusedMoE"] = self.target_scheme_map["Linear"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendCompressedTensorsConfig":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))
        target_scheme_map = cls._quantization_scheme_map_from_config(config=config)

        return cls(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            config=config,
        )

    @classmethod
    def _quantization_scheme_map_from_config(cls, config: dict[str, Any]) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """Build target scheme map from config.

        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        """

        target_scheme_map: dict[str, Any] = dict()
        quant_format = cast(str, config.get("format"))

        config_groups = config.get("config_groups", dict())
        for _, quant_config in config_groups.items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                target_scheme_map[target]["weights"] = QuantizationArgs.model_validate(quant_config.get("weights"))

                target_scheme_map[target]["input_activations"] = None
                target_scheme_map[target]["format"] = quant_config.get("format")
                format = target_scheme_map[target].get("format")
                # If no per-config format defined, use global format in config
                act_quant_format = (
                    is_activation_quantization_format(format)
                    if format is not None
                    else is_activation_quantization_format(quant_format)
                )
                input_activations = quant_config.get("input_activations")
                if act_quant_format and input_activations is not None:
                    target_scheme_map[target]["input_activations"] = QuantizationArgs.model_validate(
                        quant_config.get("input_activations")
                    )
        return target_scheme_map

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from .method_adapters import AscendFusedMoEMethod, AscendLinearMethod

        if isinstance(layer, LinearBase):
            layer.ascend_quant_method = COMPRESSED_TENSORS_METHOD
            # Get the scheme for this layer
            linear_scheme = self._get_linear_scheme(layer=layer, layer_name=prefix)

            # Return unquantized method if no scheme found
            if linear_scheme is None:
                return UnquantizedLinearMethod()

            # Store scheme on layer for reference (optional, for debugging)
            layer.scheme = linear_scheme
            logger.info_once("Using the vLLM Ascend llmcompressor Quantization now!")
            return AscendLinearMethod(linear_scheme)

        if isinstance(layer, FusedMoE):
            # Delayed import to avoid circular import
            from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod

            layer.ascend_quant_method = COMPRESSED_TENSORS_METHOD
            layer_name = prefix + ".0.gate_proj"
            # Get the scheme for this layer
            moe_scheme = self._get_moe_scheme(layer=layer, layer_name=layer_name)

            # Return unquantized method if no scheme found
            if moe_scheme is None:
                return AscendUnquantizedFusedMoEMethod(layer.moe_config)

            # Store scheme on layer for reference (optional, for debugging)
            layer.scheme = moe_scheme
            logger.info_once("Using the vLLM Ascend llmcompressor Quantization now!")
            return AscendFusedMoEMethod(moe_scheme, layer.moe_config)

        return None

    def _get_linear_scheme(self, layer: torch.nn.Module, layer_name: str | None = None) -> AscendLinearScheme | None:
        """Get the linear quantization scheme for a layer.

        Returns:
            An AscendLinearScheme instance, or None if the layer
            should use unquantized method.
        """
        weight_quant, input_quant, format = self._get_quant_args(layer, layer_name)
        if weight_quant is None:
            return None

        scheme = self._create_scheme_for_layer_type(
            weight_quant=weight_quant,
            input_quant=input_quant,
            format=format,
            layer_type="linear",
        )
        return cast(AscendLinearScheme, scheme)

    def _get_moe_scheme(self, layer: torch.nn.Module, layer_name: str | None = None) -> AscendMoEScheme | None:
        """Get the MoE quantization scheme for a layer.

        Returns:
            An AscendMoEScheme instance, or None if the layer
            should use unquantized method.
        """
        # Add FusedMoE to target scheme map if needed
        self._add_fused_moe_to_target_scheme_map()

        weight_quant, input_quant, format = self._get_quant_args(layer, layer_name)
        if weight_quant is None:
            return None

        scheme = self._create_scheme_for_layer_type(
            weight_quant=weight_quant,
            input_quant=input_quant,
            format=format,
            layer_type="moe",
        )
        return cast(AscendMoEScheme, scheme)

    def _get_quant_args(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> tuple[Optional["QuantizationArgs"], Optional["QuantizationArgs"], str | None]:
        """Extract quantization arguments for a layer.

        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target.

        Returns:
            A tuple of (weight_quant, input_quant, format). weight_quant is
            None if the layer should use unquantized method.
        """
        scheme_dict = self.get_scheme_dict(layer, layer_name)
        weight_quant = None
        input_quant = None
        format = None
        if scheme_dict:
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")
            format = scheme_dict.get("format")

        if weight_quant is None:
            logger.warning_once(
                "Acceleration for non-quantized schemes is "
                "not supported by Compressed Tensors. "
                "Falling back to UnquantizedLinearMethod"
            )

        return weight_quant, input_quant, format

    def get_scheme_dict(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> dict[str, QuantizationArgs | str | None] | None:
        """
        Extract the QuantizationArgs for a given layer.

        Returns:
            dict with {
                "weights": QuantizationArgs,
                "input_activations": QuantizationArgs | None,
                "format": str | None
            } | None
        """
        if should_ignore_layer(layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping):
            return None

        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )
            scheme_dict = self.target_scheme_map[matched_target]
            if scheme_dict.get("format") is None:
                scheme_dict["format"] = self.quant_format
            return scheme_dict

        return None

    def _create_scheme_for_layer_type(
        self,
        weight_quant: "QuantizationArgs",
        input_quant: Optional["QuantizationArgs"],
        format: str | None,
        layer_type: str,
    ) -> AscendLinearScheme | AscendMoEScheme:
        """Create the appropriate Ascend scheme based on quantization args and layer type.

        Args:
            weight_quant: Weight quantization arguments.
            input_quant: Input activation quantization arguments.
            format: Per-layer format, if defined.
            layer_type: Type of layer ("linear" or "moe").

        Returns:
            An instance of the appropriate Ascend quantization scheme.
        """
        from .methods import get_scheme_class

        # Determine the quantization type
        quant_type = self._detect_quant_type(weight_quant, input_quant, format)

        # Get the scheme class from registry
        scheme_cls = get_scheme_class(quant_type, layer_type)
        if scheme_cls is None:
            raise NotImplementedError(
                f"No compressed-tensors compatible scheme was found for "
                f"quant_type={quant_type}, layer_type={layer_type}."
            )

        return scheme_cls()

    def _detect_quant_type(
        self,
        weight_quant: "QuantizationArgs",
        input_quant: Optional["QuantizationArgs"],
        format: str | None,
    ) -> str:
        """Detect the quantization type from quantization arguments.

        Args:
            weight_quant: Weight quantization arguments.
            input_quant: Input activation quantization arguments.
            format: Per-layer format, if defined.

        Returns:
            A string representing the quantization type (e.g., "W8A8", "W8A8_DYNAMIC").
        """
        # use the per-layer format if defined, otherwise, use global format
        format = format if format is not None else self.quant_format
        act_quant_format = is_activation_quantization_format(format)

        if act_quant_format and input_quant is not None:
            if self._is_static_tensor_w8a8(weight_quant, input_quant):
                return "W8A8"

            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                return "W8A8_DYNAMIC"

            if self._is_dynamic_token_w4a8(weight_quant, input_quant):
                return "W4A8_DYNAMIC"

        if self._is_w4a16(weight_quant, input_quant):
            return "W4A16"

        raise NotImplementedError("No compressed-tensors compatible quantization type was found.")

    def _is_static_tensor_w8a8(self, weight_quant: "QuantizationArgs", input_quant: "QuantizationArgs") -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        is_tensor = weight_strategy and input_quant.strategy == QuantizationStrategy.TENSOR.value
        is_static = not weight_quant.dynamic and not input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        # Only symmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_tensor and is_symmetric and is_static

    def _is_dynamic_token_w8a8(self, weight_quant: "QuantizationArgs", input_quant: "QuantizationArgs") -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        is_token = weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        # Only symmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and is_symmetric and is_dynamic

    def _is_dynamic_token_w4a8(self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs) -> bool:
        is_4_bits = weight_quant.num_bits == 4
        is_8_bits = input_quant.num_bits == 8
        weight_strategy = (weight_quant.strategy == QuantizationStrategy.CHANNEL.value) or (
            weight_quant.strategy == QuantizationStrategy.GROUP.value
        )
        is_token = weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        # Adapt for AscendW4A8DynamicFusedMoEMethod
        assert self.quant_description is not None, "quant_description should not be None"
        if weight_strategy:
            self.quant_description["group_size"] = weight_quant.group_size if weight_quant.group_size else 0

        self.quant_description["version"] = "0"
        self.quant_description["ascend_quant_method"] = COMPRESSED_TENSORS_METHOD
        self.quant_description["weight_strategy"] = str(weight_quant.strategy)

        # Only symmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_4_bits and is_8_bits and is_token and is_symmetric and is_dynamic

    def _is_w4a16(self, weight_quant: "QuantizationArgs", input_quant: Optional["QuantizationArgs"]) -> bool:
        # Confirm weights quantized.
        if weight_quant is None:
            return False

        # Confirm we have integer type.
        if weight_quant.type != QuantizationType.INT:
            return False

        input_quant_none = input_quant is None
        is_4_bits = weight_quant.num_bits == 4
        is_group = weight_quant.strategy == QuantizationStrategy.GROUP.value
        is_static = not weight_quant.dynamic

        return input_quant_none and is_4_bits and is_group and is_static

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        self.target_scheme_map = hf_to_vllm_mapper.apply_dict(self.target_scheme_map)
        self.ignore = hf_to_vllm_mapper.apply_list(self.ignore)
