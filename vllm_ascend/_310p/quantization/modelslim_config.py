#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod, AscendLinearMethod
from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig as _BaseModelSlimConfig
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD

from .methods.w8a8_static import AscendW8A8LinearMethod310P


@register_quantization_config(ASCEND_QUANTIZATION_METHOD)
class AscendModelSlimConfig(_BaseModelSlimConfig):
    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = vllm_config.model_config.hf_config.model_type

        if model_type in ["minimax", "minimax_m2"]:
            prefix = prefix.replace("mlp", "block_sparse_moe")
            parts = prefix.split(".")
            if "experts" in parts and len(parts) > 2:
                exp_idx = parts.index("experts")
                if exp_idx + 1 < len(parts) and parts[exp_idx + 1].isdigit():
                    parts = parts[: exp_idx + 1]
                    prefix = ".".join(parts)

        if (
            model_type
            in (
                __import__("vllm_ascend.quantization.modelslim_config", fromlist=["packed_modules_model_mapping"])
            ).packed_modules_model_mapping
        ):
            from vllm_ascend.quantization.modelslim_config import packed_modules_model_mapping

            self.packed_modules_mapping = packed_modules_model_mapping[model_type]

        prefix = self.quant_prefix_mapper(model_type, prefix)
        if prefix.startswith("language_model"):
            prefix = prefix.split(".", 1)[-1]

        if isinstance(layer, LinearBase):
            if self.is_layer_skipped_ascend(prefix, getattr(self, "packed_modules_mapping", {})):
                from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

                return AscendUnquantizedLinearMethod()

            quant_type = self.quant_description.get(prefix + ".weight", None)
            if quant_type == "W8A8":
                scheme = AscendW8A8LinearMethod310P()
                return AscendLinearMethod(scheme)

        if isinstance(layer, VocabParallelEmbedding):
            if self.is_layer_skipped_ascend(prefix, getattr(self, "packed_modules_mapping", {})):
                from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

                return UnquantizedEmbeddingMethod()

            quant_type = self.quant_description.get(prefix + ".weight", None)
            if quant_type == "W8A8":
                scheme = AscendW8A8LinearMethod310P()
                return AscendEmbeddingMethod(scheme)

        return super().get_quant_method(layer, prefix)
