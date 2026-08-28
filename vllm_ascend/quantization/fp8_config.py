import inspect
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import MoERunner, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v4 import DeepseekV4FP8Config

from vllm_ascend.utils import FP8_METHOD

from .methods import get_scheme_class

# vLLM 0.27.1's is_layer_skipped has no match_mode; newer trees default to exact.
_IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE = "match_mode" in inspect.signature(is_layer_skipped).parameters


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    return isinstance(layer, (MoERunner, RoutedExperts))


QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]


@register_quantization_config(FP8_METHOD)
class AscendFp8Config(Fp8Config):
    """Serve checkpoints published under the generic HF ``fp8`` quant method.

    Only the block-wise flavour is supported: weights in ``float8_e4m3fn`` paired
    with a float32 ``weight_scale_inv`` holding one scale per
    ``weight_block_size`` tile. Layers the checkpoint left in bfloat16 are named
    in ``ignored_layers`` / ``modules_to_not_convert`` and fall back to the
    unquantized methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    def _verify_block_quantization(self) -> None:
        if self.weight_block_size is None:
            raise NotImplementedError(
                "vLLM Ascend serves native FP8 checkpoints only when their scales are stored per "
                "weight block, that is when `quantization_config.weight_block_size` is present in "
                "config.json. This checkpoint uses per-tensor or per-channel FP8 scales, which have "
                "no Ascend execution path yet. Re-quantize it with ModelSlim and serve with "
                "`--quantization ascend`, or use an llm-compressor checkpoint with "
                "`--quantization compressed-tensors`."
            )
        if self.activation_scheme != "dynamic":
            raise NotImplementedError(
                f"vLLM Ascend supports dynamic activation quantization for native FP8 checkpoints, "
                f"but this one declares `activation_scheme: {self.activation_scheme}`."
            )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        # Delayed imports: both modules reach back into the quantization package.
        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        from .method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        is_linear = isinstance(layer, LinearBase)
        is_moe = _is_fused_moe_layer(layer)
        if not is_linear and not is_moe:
            return None

        skip_kwargs = {}
        if _IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE:
            skip_kwargs["match_mode"] = getattr(self, "ignored_layers_match_mode", "exact")
        if is_layer_skipped(
            prefix,
            self.ignored_layers,
            self.packed_modules_mapping,
            **skip_kwargs,
        ):
            if is_linear:
                return AscendUnquantizedLinearMethod()
            return AscendUnquantizedFusedMoEMethod(layer.moe_config, tid2eid)

        self._verify_block_quantization()
        logger.info_once("Using the vLLM Ascend block-wise fp8 Quantization now!")

        if is_linear:
            scheme_class = get_scheme_class(FP8_METHOD, "linear")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/linear"
            return AscendLinearMethod(scheme_class(self.weight_block_size))

        if is_moe:
            scheme_class = get_scheme_class(FP8_METHOD, "moe")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/moe"
            return AscendFusedMoEMethod(
                scheme_class(self.weight_block_size, layer.moe_config),
                layer.moe_config,
                tid2eid=tid2eid,
            )

        return None


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
