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
"""310P compressed-tensors quantization configuration."""

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.logger import logger
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from vllm_ascend._310p.quantization.methods.registry import get_scheme_class
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods import AscendLinearScheme, AscendMoEScheme
from vllm_ascend.quantization.utils import is_fused_moe_layer
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD

# 310P static W8A8/W8A8S/W8A8SC schemes expect ModelSlim-specific checkpoint
# parameters (deq_scale, quant_bias, input_offset) and do not support
# compressed-tensors checkpoints.
_SUPPORTED_310P_QUANT_TYPES = frozenset({"W8A8_DYNAMIC"})


@register_quantization_config(COMPRESSED_TENSORS_METHOD)
class AscendCompressedTensorsConfig310(AscendCompressedTensorsConfig):
    """310P override for LLM-Compressor compressed-tensors quantization."""

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, VocabParallelEmbedding):
            if self.get_scheme_dict(layer, prefix) is not None:
                raise NotImplementedError(f"Quantized VocabParallelEmbedding is not supported on 310P for {prefix}.")
            from vllm_ascend._310p.ops.vocab_parallel_embedding import AscendUnquantizedEmbeddingMethod310

            return AscendUnquantizedEmbeddingMethod310()

        if is_fused_moe_layer(layer):
            moe_scheme = self._get_moe_scheme(layer=layer, layer_name=f"{prefix}.0.gate_proj")
            if moe_scheme is None:
                from vllm_ascend._310p.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod310

                return AscendUnquantizedFusedMoEMethod310(layer.moe_config)

            layer.scheme = moe_scheme
            logger.info_once("Using the vLLM Ascend llmcompressor Quantization now!")
            return AscendFusedMoEMethod(moe_scheme, layer.moe_config)

        return super().get_quant_method(layer, prefix)

    def _create_scheme_for_layer_type(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        format: str | None,
        layer_type: str,
    ) -> AscendLinearScheme | AscendMoEScheme:
        quant_type = self._detect_quant_type(weight_quant, input_quant, format)
        scheme_cls = get_scheme_class(quant_type, layer_type) if quant_type in _SUPPORTED_310P_QUANT_TYPES else None
        if scheme_cls is None:
            raise NotImplementedError(
                f"No 310P compressed-tensors compatible scheme was found for "
                f"quant_type={quant_type}, layer_type={layer_type}."
            )

        return scheme_cls()
