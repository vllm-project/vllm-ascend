from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS, register_quantization_config)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsScheme
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target, is_activation_quantization_format,
    should_ignore_layer)

from vllm_ascend.quantization.quant_config import (AscendLinearMethod,
                                                   AscendQuantConfig)
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.quantization.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

QUANTIZATION_SCHEME_MAP_TYPE = dict[str, Optional[dict[str, QuantizationArgs]]]


def remove_quantization_method():
    if COMPRESSED_TENSORS_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(COMPRESSED_TENSORS_METHOD)


remove_quantization_method()


@register_quantization_config(COMPRESSED_TENSORS_METHOD)
class AscendCompressedTensorsConfig(QuantizationConfig):

    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        config: Optional[dict[str, Any]] = None,
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
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str,
                                      Any]) -> "AscendCompressedTensorsConfig":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))
        target_scheme_map = cls._quantization_scheme_map_from_config(
            config=config)

        return cls(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            config=config,
        )

    @classmethod
    def _quantization_scheme_map_from_config(
            cls, config: dict[str, Any]) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        """
        target_scheme_map: dict[str, Any] = dict()
        quant_format = cast(str, config.get("format"))

        # The quant_config has multiple config_groups, each containing
        # an input_activations key with details about how the activations are
        # quantized, a weights key indicating how the weights are quantized,
        # and a list of targets under the `targets` key, dictating which
        # layers are impacted by the quantization details. The quantization
        # details follow the structure defined by the QuantizationArgs
        # pydantic model, which is used to verify the structure of the
        # quant_config and also store the details for later use.

        config_groups = config.get("config_groups", dict())
        for _, quant_config in config_groups.items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                target_scheme_map[target][
                    "weights"] = QuantizationArgs.model_validate(
                        quant_config.get("weights"))

                target_scheme_map[target]["input_activations"] = None
                target_scheme_map[target]["format"] = quant_config.get(
                    "format")
                format = target_scheme_map[target].get("format")
                # If no per-config format defined, use global format in config
                act_quant_format = (
                    is_activation_quantization_format(format)
                    if format is not None else
                    is_activation_quantization_format(quant_format))
                input_activations = quant_config.get("input_activations")
                if act_quant_format and input_activations is not None:
                    target_scheme_map[target]["input_activations"] = (
                        QuantizationArgs.model_validate(
                            quant_config.get("input_activations")))
        return target_scheme_map

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            layer.ascend_quant_method = COMPRESSED_TENSORS_METHOD
            # collect schemes
            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)

            # choose quantization method
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                ascend_quant_config = AscendQuantConfig(self.quant_description
                                                        or {})
                quant_method = AscendLinearMethod(ascend_quant_config, prefix,
                                                  None, layer)
            return quant_method
        return None

    def get_scheme(self,
                   layer: torch.nn.Module,
                   layer_name: Optional[str] = None
                   ) -> Optional["CompressedTensorsScheme"]:
        """
        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for inference.
        """

        # Find the "target" in the compressed-tensors config
        # that our layer conforms to.
        if should_ignore_layer(layer_name,
                               ignore=self.ignore,
                               fused_mapping=self.packed_modules_mapping):
            return None

        # Will be empty for models with only sparsity
        weight_quant = input_quant = None
        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )

            scheme_dict = self.target_scheme_map[matched_target]
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")

        if weight_quant is None:
            logger.warning_once("Acceleration for non-quantized schemes is "
                                "not supported by Compressed Tensors. "
                                "Falling back to UnquantizedLinearMethod")
            return None

        else:
            # Find the quant_scheme
            scheme = self._get_scheme_from_parts(
                weight_quant=weight_quant,
                input_quant=input_quant,
            )
        return scheme

    def _get_scheme_from_parts(
            self, weight_quant: QuantizationArgs,
            input_quant: QuantizationArgs) -> "CompressedTensorsScheme":
        act_quant_format = is_activation_quantization_format(self.quant_format)
        if act_quant_format and input_quant is not None:
            if self._is_static_tensor_w8a8(weight_quant, input_quant):
                return AscendW8A8LinearMethod()

            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                return AscendW8A8DynamicLinearMethod()

        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found.")

    def _is_static_tensor_w8a8(self, weight_quant: QuantizationArgs,
                               input_quant: QuantizationArgs) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value)
        is_tensor = (weight_strategy and input_quant.strategy
                     == QuantizationStrategy.TENSOR.value)
        is_static = not weight_quant.dynamic and not input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        # Only symmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_tensor and is_symmetric and is_static

    def _is_dynamic_token_w8a8(self, weight_quant: QuantizationArgs,
                               input_quant: QuantizationArgs) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value)
        is_token = (weight_strategy and input_quant.strategy
                    == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        # Only symmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and is_symmetric and is_dynamic

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        self.target_scheme_map = hf_to_vllm_mapper.apply_dict(
            self.target_scheme_map)
        self.ignore = hf_to_vllm_mapper.apply_list(self.ignore)
