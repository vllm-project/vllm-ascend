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
from typing import Any, Callable, Dict, Optional
import torch
import torch_npu
import torch.nn.functional as F

class AscendW4A4LaosDynamicLinearMethod:
    """Linear method for Ascend W4A4_LAOS_DYNAMIC.
    This class implements W4A4 quantization with LAOS approach and dynamic activation quantization.
    - Weight: 4-bit quantization (per-channel) with scale and offset, stored as int8.
    - Activation: 4-bit dynamic quantization.
    """
    def __init__(self):
        self.transpose_weight = True
        self.rotation_type = None

    def set_rotation_config(self, prefix, metadata):
        layer_idx = prefix.split(".")[2]
        if prefix.endswith("o_proj"):
            layers = metadata["quarot"]["heads_rotation"]["layers"]
            if layer_idx in layers:
                return "heads_rotation"
        if prefix.endswith("down_proj"):
            layers = metadata["quarot"]["kronecker_rotation"]["layers"]
            if layer_idx in layers:
                return "kronecker_rotation"

    @staticmethod
    def get_weight(input_size: int, output_size: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict
    
    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}
    
    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=torch.float32)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=torch.float32)
        if self.rotation_type == "heads_rotation":
            params_dict["heads_rotation"] = torch.zeros((64, 64), dtype=torch.float32)
        if self.rotation_type == "kronecker_rotation":
            params_dict["kronecker_rotation_n"] = torch.zeros((160, 160), dtype=torch.float32)
            params_dict["kronecker_rotation_m"] = torch.zeros((160, 160), dtype=torch.float32)
        return params_dict

    def get_pergroup_param(self,
                           input_size: int,
                           output_size: int,
                           params_dtype: torch.dtype, 
                           layer_type: Optional[str] = None) -> Dict[str, Any]:
        return {}
    
    def apply_rotation(self, layer, x):
        init_shape = x.shape
        dtype = x.dtype
        if self.rotation_type == "heads_rotation":
            Q1 = layer.heads_rotation
            scaled_x = x.reshape(-1, Q1.shape[1], 128)
            scaled_x = torch.matmul(Q1.T, scaled_x).reshape(init_shape)
            return scaled_x.to(dtype)
        if self.rotation_type == "kronecker_rotation":
            Q1 = layer.kronecker_rotation_m
            Q2 = layer.kronecker_rotation_n
            scaled_x = x.reshape(-1, Q1.shape[0], Q2.shape[0])
            scaled_x = torch.matmul(scaled_x, Q2)
            scaled_x = torch.matmul(Q1.T, scaled_x)
            scaled_x = scaled_x.reshape(init_shape)
            return scaled_x.to(dtype)
        return x

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = None,
    ) -> torch.Tensor:
        dtype = x.dtype
        x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
        pertoken_scale = pertoken_scale.reshape(-1, 1)
        pertoken_scale = pertoken_scale.squeeze(-1)
        y2 = torch_npu.npu_quant_matmul(x, layer.weight.data, scale=layer.weight_scale.data.view(-1), pertoken_scale=pertoken_scale, bias=None, output_dtype=dtype)
        return y2

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight_scale.data = layer.weight_scale.data.to(torch.float32)
        layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(layer.weight.data.to(torch.int32))
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(-1, -2)
