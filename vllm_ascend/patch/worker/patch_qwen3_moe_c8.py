#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Patch Qwen3MoeModel.load_weights to support per-channel C8 KV cache scales.

Upstream vLLM asserts `loaded_weight.numel() == 1` when loading kv_cache_scale
weights, which prevents loading QuaRot C8-quantized models whose kv_cache_scale
has num_kv_heads * head_dim elements (per-channel quantization scales).

This patch intercepts per-channel C8 KV scale weights before they reach the
upstream assertion and loads them directly into the corresponding parameters
created by AscendC8KVCacheAttentionMethod.create_weights.
"""
from collections.abc import Iterable

import torch
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel

_orig_qwen3_moe_model_load_weights = Qwen3MoeModel.load_weights


def _patched_qwen3_moe_model_load_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    """Load weights with support for per-channel C8 KV cache scales.

    Intercepts kv_cache_scale/offset weights whose numel > 1 (per-channel
    QuaRot scales) and loads them via a custom weight_loader that handles
    parameter resizing, bypassing the upstream scalar assertion.
    """
    quant_config = self.quant_config
    if quant_config is None or not callable(
        getattr(quant_config, "get_cache_scale", None)
    ):
        return _orig_qwen3_moe_model_load_weights(self, weights)

    params_dict = dict(self.named_parameters())
    c8_loaded_params: set[str] = set()

    def _intercept_c8_scales(
        raw_weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, loaded_weight in raw_weights:
            scale_name = quant_config.get_cache_scale(name)
            if scale_name is not None and scale_name in params_dict:
                param = params_dict[scale_name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight.squeeze())
                c8_loaded_params.add(scale_name)
            else:
                yield name, loaded_weight

    loaded_params = _orig_qwen3_moe_model_load_weights(
        self, _intercept_c8_scales(weights)
    )
    loaded_params.update(c8_loaded_params)
    return loaded_params


Qwen3MoeModel.load_weights = _patched_qwen3_moe_model_load_weights
