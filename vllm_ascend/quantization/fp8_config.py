from typing import Any, Optional, cast

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import MoERunner, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_ascend.utils import FP8_METHOD

from .methods import get_scheme_class


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    return isinstance(layer, (MoERunner, RoutedExperts))


QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]


def remove_quantization_method():
    if FP8_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(FP8_METHOD)
    if "deepseek_v4_fp8" in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove("deepseek_v4_fp8")


remove_quantization_method()


def create_scheme_for_layer(
    quant_description: dict[str, Any],
    prefix: str,
    layer_type: str,
    packed_modules_mapping: dict[str, Any] | None = None,
):
    """Create a quantization scheme instance for a layer.

    Args:
        quant_description: The quantization description dictionary.
        prefix: The layer prefix.
        layer_type: The type of layer ("linear", "moe", "attention").
        packed_modules_mapping: Mapping for packed/fused modules.

    Returns:
        An instance of the appropriate quantization scheme class.
    """
    logger.info_once("Using the vLLM Ascend fp8 Quantization now!")
    quant_type = "FP8"

    # Use registry to get scheme class
    scheme_cls = get_scheme_class(quant_type, layer_type)
    if scheme_cls is not None:
        return scheme_cls(quant_description)

    raise NotImplementedError(f"Currently, vLLM Ascend doesn't support {quant_type} for {layer_type}.")


@register_quantization_config(FP8_METHOD)
class AscendFp8Config(QuantizationConfig):
    def __init__(
        self,
        ignore: list[str],
        quant_format: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.ignored_layers = ignore
        self.ignored_layers_match_mode = "substring"
        self.quant_format = quant_format
        self.quant_description = config if config is not None else {}
        self.weight_block_size = self.quant_description.get("weight_block_size")
        self.activation_scheme = self.quant_description.get(
            "activation_scheme", "dynamic"
        )
        self.is_per_tensor_fp8 = self.weight_block_size is None
        self.mistral4_dynamic_channelwise = False

    def __repr__(self) -> str:
        return "Fp8Config:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return FP8_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float8_e4m3fn, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendFp8Config":
        ignore = config.get("ignore") or config.get("ignored_layers")
        if not ignore:
            ignore = config.get("modules_to_not_convert", [])
        ignore = cast(list[str], ignore)
        quant_format = cast(str, config.get("format"))

        return cls(
            ignore=ignore,
            quant_format=quant_format,
            config=config,
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        # Ignore entries name modules, while model weight prefix mappings end
        # in a dot. Preserve that boundary so a generic prefix (for example
        # ``model.``) cannot win over ``model.vision_tower.``.
        module_prefixes = [f"{name}." for name in self.ignore]
        mapped_prefixes = hf_to_vllm_mapper.apply_list(module_prefixes)
        self.ignore = [name.removesuffix(".") for name in mapped_prefixes]
        self.ignored_layers = self.ignore

    @staticmethod
    def _is_mistral4_model() -> bool:
        hf_config = get_current_vllm_config().model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        return getattr(text_config, "model_type", None) == "mistral4"

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from .method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        self.mistral4_dynamic_channelwise = (
            self.is_per_tensor_fp8 and self._is_mistral4_model()
        )

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return UnquantizedLinearMethod()
            layer.ascend_quant_method = FP8_METHOD

            if self.mistral4_dynamic_channelwise:
                scheme_cls = get_scheme_class("W8A8FP8_DYNAMIC", "linear")
                assert scheme_cls is not None
                scheme = scheme_cls()
                logger.warning_once(
                    "A2 per-tensor FP8 keeps serialized weights but uses "
                    "dynamic activation quantization; checkpoint static "
                    "activation scales are not consumed."
                )
            else:
                scheme = create_scheme_for_layer(
                    self.quant_description,
                    prefix,
                    "ds_linear",
                    self.packed_modules_mapping,
                )
            quant_method = AscendLinearMethod(scheme)
            return quant_method
        if _is_fused_moe_layer(layer):
            layer.ascend_quant_method = FP8_METHOD
            if self.mistral4_dynamic_channelwise:
                scheme_cls = get_scheme_class("W8A8FP8_DYNAMIC", "moe")
                assert scheme_cls is not None
                scheme = scheme_cls()
            else:
                scheme = create_scheme_for_layer(
                    self.quant_description,
                    prefix,
                    "w4a8_moe",
                    self.packed_modules_mapping,
                )
            quant_method = AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None


# deepseek_v4_fp8 is handled identically to fp8 on Ascend — reuse the same config.
register_quantization_config("deepseek_v4_fp8")(AscendFp8Config)
