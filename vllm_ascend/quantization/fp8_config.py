from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase,
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

from vllm_ascend.utils import FP8_METHOD

from .methods import get_scheme_class

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

QUANTIZATION_SCHEME_MAP_TYPE = dict[str, Optional[dict[str, QuantizationArgs]]]


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
        logger.info_once("Using the vLLM Ascend modelslim Quantization now!")
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
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        self.quant_description = config

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
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str,
                                      Any]) -> "AscendFp8Config":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))

        return cls(
            ignore=ignore,
            quant_format=quant_format,
            config=config,
        )

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
        if isinstance(layer, LinearBase):
            layer.ascend_quant_method = FP8_METHOD

            scheme = create_scheme_for_layer(self.quant_description, prefix, "ds_linear", self.packed_modules_mapping)
            quant_method = AscendLinearMethod(scheme)
            return quant_method
        if isinstance(layer, FusedMoE):
            layer.ascend_quant_method = FP8_METHOD
            scheme = create_scheme_for_layer(self.quant_description, prefix, "w4a8_moe", self.packed_modules_mapping)
            quant_method = AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        self.target_scheme_map = hf_to_vllm_mapper.apply_dict(self.target_scheme_map)
        self.ignore = hf_to_vllm_mapper.apply_list(self.ignore)


# deepseek_v4_fp8 is handled identically to fp8 on Ascend — reuse the same config.
register_quantization_config("deepseek_v4_fp8")(AscendFp8Config)
