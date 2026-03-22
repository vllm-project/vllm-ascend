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
import torch_npu

from vllm_ascend.utils import maybe_trans_nz

from .base import AscendLinearScheme
from .registry import register_scheme


@register_scheme("W8A16", "linear")
class AscendW8A16LinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A16.

    This scheme uses 8-bit quantized weights with 16-bit activations.
    """

    def __init__(self) -> None:
        pass

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        output = torch_npu.npu_weight_quant_batchmatmul(
            x=x,
            weight=layer.weight,
            antiquant_scale=layer.weight_scale,
            antiquant_offset=layer.weight_offset,
            bias=bias,
        )
        return output

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = maybe_trans_nz(layer.weight.data)
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
