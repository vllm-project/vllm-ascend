from typing import Any, Optional, cast

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.logger import logger
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.models.deepseek_v4 import DeepseekV4FP8Config


from vllm_ascend.utils import FP8_METHOD

from .methods import get_scheme_class

QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]


@register_quantization_config(FP8_METHOD)
class AscendFp8Config(Fp8Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}

    def __repr__(self) -> str:
        return "Fp8Config:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return FP8_METHOD

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

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
            quant_method = AscendLinearMethod(scheme_class(self.weight_block_size))
            return quant_method
        if isinstance(layer, FusedMoE):
            scheme_class = get_scheme_class(FP8_METHOD, "ds_w8a8_moe")
            quant_method = AscendFusedMoEMethod(scheme_class(self.weight_block_size), layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None


@register_quantization_config("deepseek_v4_fp8")
class AscendDeepseekV4FP8Config(DeepseekV4FP8Config):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}
    
    @classmethod
    def get_name(cls) -> str:
        return "deepseek_v4_fp8"
    
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
            quant_method = AscendLinearMethod(scheme_class(self.weight_block_size))
            return quant_method
        if isinstance(layer, FusedMoE):
            if self.expert_dtype == 'fp4':
                scheme_class = get_scheme_class(FP8_METHOD, "ds_w4a8_moe")
            else:
                scheme_class = get_scheme_class(FP8_METHOD, "ds_w8a8_moe")
            quant_method = AscendFusedMoEMethod(scheme_class(), layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None
