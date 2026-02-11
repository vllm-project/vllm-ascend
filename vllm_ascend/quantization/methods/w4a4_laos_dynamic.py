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

from .base import AscendLinearScheme
from .registry import register_scheme


@register_scheme("W4A4_DYNAMIC", "linear")
class AscendW4A4LaosDynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A4_DYNAMIC.

    This class implements W4A4 quantization with LAOS approach and dynamic activation quantization.
    - Weight: 4-bit quantization (per-channel) with scale and offset, stored as int8.
    - Activation: 4-bit dynamic quantization.
    """

    def __init__(self):
        self.transpose_weight = True

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=torch.float32)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        dtype = x.dtype
        x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
        pertoken_scale = pertoken_scale.reshape(-1, 1)
        pertoken_scale = pertoken_scale.squeeze(-1)
        output = torch_npu.npu_quant_matmul(
            x,
            layer.weight.data,
            scale=layer.weight_scale.data.view(-1),
            pertoken_scale=pertoken_scale,
            bias=None,
            output_dtype=dtype,
        )
        if bias is not None:
            output = output + bias.to(dtype)
        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale.data = layer.weight_scale.data.to(torch.float32)
        layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(layer.weight.data.to(torch.int32))
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(-1, -2)
