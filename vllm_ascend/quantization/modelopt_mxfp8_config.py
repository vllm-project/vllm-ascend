# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from typing import Optional

import torch
from vllm.model_executor.layers.fused_moe import MoERunner, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    return isinstance(layer, (MoERunner, RoutedExperts))


@register_quantization_config("mxfp8")
@register_quantization_config("modelopt_mxfp8")
class AscendModelOptMxFp8Config(ModelOptMxFp8Config):
    """Load ModelOpt-compatible MXFP8 checkpoints with Ascend methods."""

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware does not support "get_min_capability" feature.')

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        from .method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )
        from .methods.w8a8_mxfp8 import (
            AscendW8A8MXFP8DynamicFusedMoEMethod,
            AscendW8A8MXFP8DynamicLinearMethod,
        )

        if self.is_layer_excluded(prefix):
            if isinstance(layer, VocabParallelEmbedding):
                return UnquantizedEmbeddingMethod()
            if isinstance(layer, LinearBase):
                return AscendUnquantizedLinearMethod()
            if _is_fused_moe_layer(layer):
                return AscendUnquantizedFusedMoEMethod(layer.moe_config, tid2eid)
            return None

        # Match the compatibility behavior retained by ModelOpt for older
        # checkpoints whose vision modules were not listed in exclude_modules.
        if any(module_name in prefix for module_name in ("vision_tower", "vision_model", "vit_large_projector")):
            if isinstance(layer, VocabParallelEmbedding):
                return UnquantizedEmbeddingMethod()
            if isinstance(layer, LinearBase):
                return AscendUnquantizedLinearMethod()
            return None

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return AscendLinearMethod(AscendW8A8MXFP8DynamicLinearMethod())

        if _is_fused_moe_layer(layer):
            return AscendFusedMoEMethod(
                AscendW8A8MXFP8DynamicFusedMoEMethod(),
                layer.moe_config,
                tid2eid,
            )

        # Embeddings and the BF16 KV cache keep their native methods.
        return None
