#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from vllm_ascend._310p.quantization.methods.registry import (
    get_scheme_class,
)
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from vllm_ascend.quantization.modelslim_config import (
    AscendModelSlimConfig,
    get_quant_type_for_layer,
    packed_modules_model_mapping,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD

logger = init_logger(__name__)


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
    quant_type = get_quant_type_for_layer(quant_description, prefix, layer_type, packed_modules_mapping)

    if quant_type is None:
        raise ValueError(f"Could not determine quantization type for layer {prefix}.")

    # Use registry to get scheme class
    scheme_cls = get_scheme_class(quant_type, layer_type)
    if scheme_cls is not None:
        return scheme_cls()
    else:
        raise NotImplementedError(f"Currently, vLLM Ascend doesn't support {quant_type} for {layer_type}.")


@register_quantization_config(ASCEND_QUANTIZATION_METHOD)
class AscendModelSlimConfig310(AscendModelSlimConfig):
    """310P override for ModelSlim quantization config.

    - Uses 310P-local scheme registry to create scheme by (quant_type, layer_type).
    - MUST keep packed_modules_mapping behavior consistent with base, otherwise
      fused modules (qkv_proj / gate_up_proj) will miss and fallback to base,
      causing NZ/transpose issues on 310P.
    """

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
                from vllm_ascend._310p.ops.linear import AscendUnquantizedLinearMethod310

                return AscendUnquantizedLinearMethod310()

            scheme = create_scheme_for_layer(
                quant_description=self.quant_description,
                prefix=prefix,
                layer_type="linear",
                packed_modules_mapping=packed,
            )
            return AscendLinearMethod(scheme)

        elif isinstance(layer, FusedMoE):
            if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
                from vllm_ascend._310p.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod310

                return AscendUnquantizedFusedMoEMethod310(layer.moe_config)
            scheme = create_scheme_for_layer(self.quant_description, prefix, "moe", self.packed_modules_mapping)
            return AscendFusedMoEMethod(scheme, layer.moe_config)

        elif isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()

        return super().get_quant_method(layer, prefix)
