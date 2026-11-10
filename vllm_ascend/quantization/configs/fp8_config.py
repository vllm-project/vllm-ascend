import inspect
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.deepseek_v4 import DeepseekV4FP8Config

from vllm_ascend.utils import FP8_METHOD

from ..methods import get_scheme_class
from ..utils import is_fused_moe_layer

# vLLM 0.27.1's is_layer_skipped has no match_mode; newer trees default to exact.
_IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE = "match_mode" in inspect.signature(is_layer_skipped).parameters


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

    @staticmethod
    def _mxfp_c8_enabled() -> bool:
        vllm_config = get_current_vllm_config_or_none()
        return (
            vllm_config is not None
            and vllm_config.cache_config is not None
            and vllm_config.cache_config.cache_dtype == "mxfp8"
        )

    def get_cache_scale_mapper(self) -> WeightsMapper:
        mapper = super().get_cache_scale_mapper()
        if not self._mxfp_c8_enabled():
            return mapper

        # Native block-FP8 checkpoints may carry the same ModelSlim C8 recipe
        # as MXFP8 checkpoints. Route the static per-channel V scale to the
        # parameter installed by AscendC8MXFPKVCacheAttentionMethod.
        c8_mapper = WeightsMapper(
            orig_to_new_suffix={".fa_v.scale": ".attn.v_cache_scale"}
        )
        return c8_mapper | mapper

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

    def _is_layer_skipped(self, prefix: str) -> bool:
        """Allow flat checkpoint prefixes to match layers under a model wrapper."""
        if not getattr(self, "ignored_layers", None):
            return False

        skip_kwargs = {}
        if _IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE:
            skip_kwargs["match_mode"] = getattr(self, "ignored_layers_match_mode", "exact")
        if is_layer_skipped(prefix, self.ignored_layers, self.packed_modules_mapping, **skip_kwargs):
            return True

        if not _IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE or skip_kwargs.get("match_mode") == "suffix":
            return False

        if is_layer_skipped(prefix, self.ignored_layers, self.packed_modules_mapping, match_mode="suffix"):
            logger.warning_once(
                "Layer '%s' was left unquantized because it matched an entry in this "
                "checkpoint's `ignored_layers` / `modules_to_not_convert` only via a "
                "module-prefix suffix, not an exact match. This usually means the "
                "checkpoint's `quantization_config` was written for a different "
                "module-prefix convention than the one vLLM constructed for this model "
                "(for example, a nested conditional-generation wrapper whose language "
                "model lives under a `language_model.` submodule). The layer is being "
                "treated as intentionally unquantized, which is very likely correct, but "
                "you should fix the checkpoint's `ignored_layers` prefixes to match "
                "vLLM's module names directly so this fallback is no longer needed.",
                prefix,
            )
            return True

        return False

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        # Delayed imports: both modules reach back into the quantization package.
        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        from ..method_adapters import (
            AscendFusedMoEMethod,
            AscendKVCacheMethod,
            AscendLinearMethod,
        )

        if isinstance(layer, Attention):
            if not self._mxfp_c8_enabled():
                return None

            from ..methods.kv_cache.mxfp_c8 import AscendC8MXFPKVCacheAttentionMethod

            logger.info_once(
                "C8-MXFP KV cache enabled for native block-wise FP8 weights."
            )
            return AscendKVCacheMethod(AscendC8MXFPKVCacheAttentionMethod({}, prefix))

        is_linear = isinstance(layer, LinearBase)
        is_moe = is_fused_moe_layer(layer)
        if not is_linear and not is_moe:
            return None

        if self._is_layer_skipped(prefix):
            if is_linear:
                return AscendUnquantizedLinearMethod()
            return AscendUnquantizedFusedMoEMethod(layer.moe_config)

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
    ) -> Optional["QuantizeMethodBase"]:
        from ..method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        if isinstance(layer, LinearBase):
            scheme_class = get_scheme_class(FP8_METHOD, "ds_linear")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/ds_linear"
            quant_method = AscendLinearMethod(scheme_class(self.weight_block_size))
            return quant_method
        if is_fused_moe_layer(layer):
            if self.expert_dtype == "fp4":
                scheme_class = get_scheme_class(FP8_METHOD, "ds_w4a8_moe")
                assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/ds_w4a8_moe"
            else:
                raise NotImplementedError
            quant_method = AscendFusedMoEMethod(scheme_class(), layer.moe_config)
            return quant_method
        return None
