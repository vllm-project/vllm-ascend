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

from typing import Any

import torch

from .base import QuantType
from .registry import register_scheme
from .w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod, AscendW8A8DynamicLinearMethod


@register_scheme("W8A8FP8_DYNAMIC", "linear")
class AscendW8A8FP8DynamicLinearMethod(AscendW8A8DynamicLinearMethod):
    """Linear method for Ascend W8A8FP8_DYNAMIC.

    This scheme uses FP8 dynamic per-token quantization for activations
    and FP8 per-channel quantization for weights.
    """

    act_quant_type: torch.dtype = torch.float8_e4m3fn

    def __init__(self):
        pass

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)}
        return params_dict

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        output = super().apply(layer, x, bias, tp_rank)
        # TODO: there is a bug in npu_quant_matmul for fp8 with bias
        # after the bug is fixed, the whole apply method can be removed.
        if bias is not None:
            output = (output + bias).to(x.dtype)
        return output


@register_scheme("W8A8FP8_DYNAMIC", "moe")
class AscendW8A8FP8DynamicFusedMoEMethod(AscendW8A8DynamicFusedMoEMethod):
    """FusedMoE method for Ascend W8A8FP8_DYNAMIC."""

    quant_type: QuantType = QuantType.W8A8FP8

    def __init__(self):
        super().__init__()

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.float8_e4m3fn
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.float8_e4m3fn
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
        )
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=torch.float32)
        param_dict["w2_weight_offset"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        return param_dict
