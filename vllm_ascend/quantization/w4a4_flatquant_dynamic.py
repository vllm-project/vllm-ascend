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
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch_npu

KRONECKER_QUANT_MAX_BATCH_SIZE = 32768


def pack_int4_weights(weight_tensor: torch.Tensor) -> torch.Tensor:
    original_device = weight_tensor.device
    weight_tensor_npu = weight_tensor.npu()
    weight_int4_packed = torch_npu.npu_convert_weight_to_int4pack(
        weight_tensor_npu.to(torch.int32), inner_k_tiles=1)
    return weight_int4_packed.to(original_device)


def get_decompose_dim(n):
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


# TODO: This function is a temporary workaround for the npu_kronecker_quant operator,
# which has a limitation on the maximum batch size (dim0). This wrapper should be
# removed once the operator supports larger inputs natively.
def batched_kronecker_quant(
    x: torch.Tensor,
    left_trans: torch.Tensor,
    right_trans: torch.Tensor,
    clip_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_tokens = x.shape[0]
    if batch_tokens <= KRONECKER_QUANT_MAX_BATCH_SIZE:
        return torch_npu.npu_kronecker_quant(x,
                                             left_trans,
                                             right_trans,
                                             clip_ratio=clip_ratio,
                                             dst_dtype=torch.int32)
    x_chunks = torch.split(x, KRONECKER_QUANT_MAX_BATCH_SIZE, dim=0)
    processed_chunks = [
        torch_npu.npu_kronecker_quant(chunk,
                                      left_trans,
                                      right_trans,
                                      clip_ratio=clip_ratio,
                                      dst_dtype=torch.int32)
        for chunk in x_chunks
    ]
    quantized_list, scale_list = zip(*processed_chunks)
    x_quantized_int4 = torch.cat(quantized_list, dim=0)
    activation_scale = torch.cat(scale_list, dim=0)
    return x_quantized_int4, activation_scale


class AscendW4A4FlatQuantDynamicLinearMethod:
    """Linear method for Ascend W4A4_FLATQUANT_DYNAMIC.
    
    This class implements W4A4 quantization with FlatQuant approach and dynamic activation quantization.
    - Weight: 4-bit quantization (per-channel) with scale and offset, stored as int8 and packed to int32 during loading
    - Activation: 4-bit dynamic quantization with FlatQuant transform matrices (left_trans, right_trans) for distribution smoothing
    - Parameters: clip_ratio for controlling quantization clipping, weight_offset for asymmetric quantization, loaded from external weights
    """
    input_size = 0

    def __init__(self):
        self.transpose_weight = False
        self.sym = True

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        if input_size % 8 != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by 8 for int4 packing"
            )
        AscendW4A4FlatQuantDynamicLinearMethod.input_size = input_size
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        left_trans_dim, right_trans_dim = get_decompose_dim(
            AscendW4A4FlatQuantDynamicLinearMethod.input_size)
        params_dict["left_trans"] = torch.empty(left_trans_dim,
                                                left_trans_dim,
                                                dtype=params_dtype)
        params_dict["right_trans"] = torch.empty(right_trans_dim,
                                                 right_trans_dim,
                                                 dtype=params_dtype)
        params_dict["clip_ratio"] = torch.empty(1, dtype=torch.float32)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=torch.float32)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=torch.float32)
        return params_dict

    def get_pergroup_param(self, input_size: int, output_size: int,
                           params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        input_shape = x.shape
        in_features = input_shape[-1]
        left_dim = layer.left_trans.shape[0]
        right_dim = layer.right_trans.shape[0]
        if left_dim * right_dim != in_features:
            raise ValueError(
                f"FlatQuant transform matrices dimension mismatch: "
                f"left_dim({left_dim}) * right_dim({right_dim}) != in_features({in_features})"
            )
        left_trans_matched = layer.left_trans.to(original_dtype)
        right_trans_matched = layer.right_trans.to(original_dtype)
        x_reshaped = x.view(-1, left_dim, right_dim)
        x_quantized_int4, activation_scale = batched_kronecker_quant(
            x_reshaped, left_trans_matched, right_trans_matched,
            layer.aclnn_clip_ratio)
        x_quantized_reshaped = x_quantized_int4.view(-1,
                                                     left_dim * right_dim // 8)
        pertoken_scale = activation_scale.view(-1).to(torch.float32)
        output = torch_npu.npu_quant_matmul(x_quantized_reshaped,
                                            layer.weight_packed.t(),
                                            layer.weight_scale.view(-1).to(
                                                torch.float32),
                                            pertoken_scale=pertoken_scale,
                                            bias=None,
                                            output_dtype=original_dtype)
        output = output.view(*input_shape[:-1], -1)
        if bias is not None:
            output = output + bias.to(original_dtype)
        return output

    def process_weights_after_loading(self, layer):
        weight_packed = pack_int4_weights(layer.weight.data)
        if self.transpose_weight:
            weight_packed = weight_packed.transpose(0, 1).contiguous()
        layer.register_parameter(
            'weight_packed',
            torch.nn.Parameter(weight_packed, requires_grad=False))
        del layer.weight
        layer.weight_scale.data = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.to(torch.float32)
        layer.left_trans = torch.nn.Parameter(
            layer.left_trans.data.t().contiguous())
        layer.right_trans = torch.nn.Parameter(layer.right_trans.data)
        layer.clip_ratio = torch.nn.Parameter(
            layer.clip_ratio.data.to(torch.float32))
        layer.aclnn_clip_ratio = layer.clip_ratio.item()
