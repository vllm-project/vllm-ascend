from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.fused_moe import MoERunner, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.models.deepseek_v4 import DeepseekV4FP8Config

from vllm_ascend.utils import FP8_METHOD

from .methods import get_scheme_class


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    return isinstance(layer, (MoERunner, RoutedExperts))


QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]


@register_quantization_config(FP8_METHOD)
class AscendFp8Config(Fp8Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        raise NotImplementedError


@register_quantization_config("deepseek_v4_fp8")
class AscendDeepseekV4FP8Config(DeepseekV4FP8Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}

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
            scheme_class = get_scheme_class(FP8_METHOD, "ds_linear")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/ds_linear"
            quant_method = AscendLinearMethod(scheme_class(self.weight_block_size))
            return quant_method
        if _is_fused_moe_layer(layer):
            if self.expert_dtype == "fp4":
                scheme_class = get_scheme_class(FP8_METHOD, "ds_w4a8_moe")
                assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/ds_w4a8_moe"
            else:
                raise NotImplementedError
            quant_method = AscendFusedMoEMethod(scheme_class(), layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None
