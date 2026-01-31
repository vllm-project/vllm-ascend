from __future__ import annotations

from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

# Important: trigger 310P method registrations (register into 310P-local registry)
from vllm_ascend._310p.quantization import methods as _methods_310p  # noqa: F401
from vllm_ascend._310p.quantization.methods.registry import get_scheme_class as get_scheme_class_310p
from vllm_ascend.quantization.method_adapters import (
    AscendLinearMethod,
)
from vllm_ascend.quantization.modelslim_config import (
    AscendModelSlimConfig,
    packed_modules_model_mapping,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD

logger = init_logger(__name__)


def get_linear_quant_type(
    quant_description: dict[str, Any], prefix: str, packed_modules_mapping: dict[str, Any]
) -> str | None:
    """Determine the quantization type for a linear layer.

    Args:
        quant_description: The quantization description dictionary.
        prefix: The layer prefix.
        packed_modules_mapping: Mapping for packed/fused modules.

    Returns:
        The quantization type string (e.g., "W8A8_DYNAMIC").
    """
    proj_name = prefix.split(".")[-1]
    if proj_name in packed_modules_mapping:
        quant_type = None
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name) for shard_proj_name in packed_modules_mapping[proj_name]
        ]
        for shard_prefix in shard_prefixes:
            shard_quant_type = quant_description[shard_prefix + ".weight"]

            if quant_type is None:
                quant_type = shard_quant_type
            elif shard_quant_type != quant_type:
                raise ValueError(
                    f"Not all shards of {prefix} are quantized with same quant type."
                    f"Shard {proj_name} uses {shard_quant_type}, but another shard"
                    f"use {quant_type}. Please check quantization config."
                )
    else:
        quant_type = quant_description[prefix + ".weight"]
    return quant_type


def create_scheme_for_layer_310p(
    cfg: AscendModelSlimConfig,
    quant_description: dict[str, Any],
    prefix: str,
    layer_type: str,
    packed_modules_mapping: dict[str, Any] | None = None,
):
    """Create 310P quant scheme (mainline-like).

    - If quant_type cannot be determined: raise ValueError
    - If quant_type is determined but not supported on 310P: raise NotImplementedError
    """
    logger.info_once("Using 310P ModelSlim Quantization routing.")

    if layer_type != "linear":
        raise NotImplementedError(f"310P quantization: layer_type={layer_type} is not supported yet (TODO).")

    quant_type = cfg._get_linear_quant_type(prefix)
    if quant_type is None:
        raise ValueError(f"310P quantization: could not determine quant_type for layer={prefix}.")

    scheme_cls = get_scheme_class_310p(quant_type, "linear")
    if scheme_cls is None:
        raise NotImplementedError(f"310P quantization: quant_type={quant_type} for linear is not supported yet (TODO).")

    return scheme_cls()


@register_quantization_config(ASCEND_QUANTIZATION_METHOD)
class AscendModelSlimConfig310(AscendModelSlimConfig):
    """310P override for ModelSlim quantization config.

    - Uses 310P-local scheme registry to create scheme by (quant_type, layer_type).
    - MUST keep packed_modules_mapping behavior consistent with base, otherwise
      fused modules (qkv_proj / gate_up_proj) will miss and fallback to base,
      causing NZ/transpose issues on 310P.
    """

    def _get_linear_quant_type(self, prefix: str) -> str | None:
        """Packed-aware quant type lookup.

        ModelSlim may describe fused modules by their shards.
        Example:
          prefix = "...qkv_proj" -> shards "...q_proj.weight", "...k_proj.weight", "...v_proj.weight"
        """
        fused_mapping = getattr(self, "packed_modules_mapping", {}) or {}
        proj_name = prefix.split(".")[-1]

        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name) for shard_proj_name in fused_mapping[proj_name]
            ]
            quant_types: list[str] = []
            for sp in shard_prefixes:
                qt = self.quant_description.get(sp + ".weight")
                if isinstance(qt, str):
                    quant_types.append(qt)

            if not quant_types:
                return None

            first = quant_types[0]
            if any(q != first for q in quant_types[1:]):
                raise ValueError(
                    f"310P quantization: not all shards of fused layer '{prefix}' "
                    f"share the same quant type. shards={shard_prefixes}, types={quant_types}"
                )
            return first

        qt = self.quant_description.get(prefix + ".weight")
        return qt if isinstance(qt, str) else None

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        vllm_config = get_current_vllm_config()
        model_type = vllm_config.model_config.hf_config.model_type

        if model_type in packed_modules_model_mapping:
            self.packed_modules_mapping = packed_modules_model_mapping[model_type]

        prefix = self.quant_prefix_mapper(model_type, prefix)
        if prefix.startswith("language_model"):
            prefix = prefix.split(".", 1)[-1]

        if isinstance(layer, LinearBase):
            packed = getattr(self, "packed_modules_mapping", {})
            if self.is_layer_skipped_ascend(prefix, packed):
                from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

                return AscendUnquantizedLinearMethod()

            scheme = create_scheme_for_layer_310p(
                cfg=self,
                quant_description=self.quant_description,
                prefix=prefix,
                layer_type="linear",
                packed_modules_mapping=packed,
            )
            return AscendLinearMethod(scheme)

        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()

        if isinstance(layer, FusedMoE):
            raise NotImplementedError(
                "310P quantization: FusedMoE is not supported yet. "
                "TODO: add 310P MoE quant schemes and routing. "
                "Workaround: use a non-MoE model."
            )

        return super().get_quant_method(layer, prefix)
