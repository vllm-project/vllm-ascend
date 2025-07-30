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
from vllm.config import get_current_vllm_config


class AscendW4A8DynamicLinearMethod:
    """Linear method for Ascend W4A8_DYNAMIC
    """

    def __init__(self):
        self.transpose_weight = True
        try:
            self.group_size = get_current_vllm_config(
            ).quant_config.quant_description.get("group_size", 256)
        except AttributeError:
            self.group_size = 256

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(output_size: int,
                             params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    def get_pergroup_param(self, input_size: int, output_size: int,
                           params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
        params_dict["weight_scale_second"] = torch.empty(output_size,
                                                         input_size //
                                                         self.group_size,
                                                         dtype=params_dtype)
        params_dict["weight_offset_second"] = torch.empty(output_size,
                                                          input_size //
                                                          self.group_size,
                                                          dtype=params_dtype)
        return params_dict

    @staticmethod
    def process_scale_second(weight: torch.Tensor, scale: torch.Tensor,
                             per_group_scale: torch.Tensor):
        k, n = weight.shape
        group_num, n = per_group_scale.shape
        weight_high = weight.to(torch.float32).reshape(
            group_num, -1, n) * per_group_scale.reshape(group_num, 1, n)
        weight_high = weight_high.reshape(k, n)
        bias = 8 * (weight_high.to(torch.float32) * scale).sum(dim=0)
        antiquant_scale = (scale * per_group_scale).reshape(group_num, n)
        return antiquant_scale.npu(), bias

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = None,
    ) -> torch.Tensor:
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            layer.weight,
            antiquant_scale=layer.weight_scale_second.to(x.dtype),
            antiquant_group_size=self.group_size,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten().to(
            torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        layer.weight_scale_second.data, scale_bias = self.process_scale_second(
            layer.weight.data,
            layer.weight_scale.data,
            layer.weight_scale_second.data.transpose(0, 1).contiguous(),
        )
        param = torch.nn.Parameter(scale_bias, requires_grad=False)
        layer.register_parameter("weight_scale_bias", param)
        layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(
            layer.weight.data.to(torch.int32))
