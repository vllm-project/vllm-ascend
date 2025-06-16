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

from typing import Any, Dict, Optional

import torch
import torch_npu


def quant_per_tensor(in_tensor: torch.Tensor, input_scale: torch.Tensor,
                     input_offset: torch.Tensor):
    return torch_npu.npu_quantize(in_tensor, input_scale, input_offset,
                                  torch.qint8, -1, False)


class AscendW8A8LinearMethod:
    """Linear method for Ascend W8A8.

    Args:
        w_sym: whether the linear weight is symmetrically quantized.
    """
    params_dtype: torch.dtype = torch.bfloat16

    def __init__(self) -> None:
        # aclnn quant matmul requires to transpose matrix B, set to true by default.
        self.transpose_weight = True

    @staticmethod
    def get_weight(
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=torch.int8)
        AscendW8A8LinearMethod.params_dtype = params_dtype
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size,
                                                   dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size,
                                                   dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
        AscendW8A8LinearMethod.params_dtype = params_dtype
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            x = quant_per_tensor(
                x,
                layer.aclnn_input_scale,
                layer.aclnn_input_offset,
            )
        quant_bias = layer.quant_bias if tp_rank == 0 else None
        return torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=AscendW8A8LinearMethod.params_dtype,
        )

    def process_weights_after_loading(self, layer):
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor),
            requires_grad=False).to(layer.aclnn_input_scale.dtype)
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
