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
"""W8A8 Prefill-Decode Mix quantization methods.

This module provides quantization methods that use different strategies
for prefill and decode phases:
- Prefill: Uses dynamic W8A8 quantization
- Decode (KV consumer): Uses static W8A8 quantization
"""

from typing import Any

import torch
from vllm.config import get_current_vllm_config

from .base import AscendLinearScheme
from .registry import register_scheme
from .w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod, AscendW8A8DynamicLinearMethod
from .w8a8_static import AscendW8A8LinearMethod


@register_scheme("W8A8_MIX", "linear")
class AscendW8A8PDMixLinearMethod(AscendLinearScheme):
    """Linear method for W8A8 prefill-decode mix quantization.

    This scheme uses composition to delegate to the appropriate quantization
    method based on the execution phase:
    - Static W8A8 for KV consumer (decode phase)
    - Dynamic W8A8 for prefill phase

    The static method is used for weight/parameter specifications since
    it requires more parameters (input_scale, deq_scale, etc.) that are
    needed for static quantization during decode.
    """

    def __init__(self):
        self._static_method = AscendW8A8LinearMethod()
        self._dynamic_method = AscendW8A8DynamicLinearMethod()

        kv_transfer_config = get_current_vllm_config().kv_transfer_config
        self._is_kv_consumer = kv_transfer_config is not None and kv_transfer_config.is_kv_consumer

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return self._static_method.get_weight(input_size, output_size, params_dtype)

    def get_pertensor_param(self, params_dtype: torch.dtype) -> dict[str, Any]:
        return self._static_method.get_pertensor_param(params_dtype)

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return self._static_method.get_perchannel_param(output_size, params_dtype)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if layer.is_kv_consumer:
            return self._static_method.apply(layer, x, bias, tp_rank)
        else:
            return self._dynamic_method.apply(layer, x, bias, tp_rank)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._static_method.process_weights_after_loading(layer)
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.is_kv_consumer = self._is_kv_consumer


@register_scheme("W8A8_MIX", "moe")
class AscendW8A8PDMixFusedMoeMethod(AscendW8A8DynamicFusedMoEMethod):
    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = super().get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype
        )
        param_dict["w2_deq_scale"] = torch.empty(num_experts, hidden_sizes, dtype=torch.float32)
        param_dict["w13_deq_scale"] = torch.empty(num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32)
        param_dict["w2_input_offset"] = torch.empty(num_experts, 1, dtype=torch.int8)
        param_dict["w13_input_offset"] = torch.empty(num_experts, 1, dtype=torch.int8)

        return param_dict
