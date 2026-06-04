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
from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_ascend.device.mxfp_compat import ensure_mxfp4_flatquant_linear_available

from .base import AscendLinearScheme
from .registry import register_scheme

# Maximum supported dimension for Kronecker quantization left_trans_dim and right_trans_dim
MAX_SUPPORT_DIM = 256


def get_decompose_dim(n: int, m: int) -> tuple[int, int]:
    """Get decomposed dimensions for Kronecker quantization.
    Args:
        n: Dimension to decompose
        m: Tensor parallelism size
    Returns:
        tuple[int, int]: Left decomposed dim, right decomposed dim
    Raises:
        ValueError: If decomposed dimension exceeds MAX_SUPPORT_DIM
    """
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1

    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1

    if (a + b) > MAX_SUPPORT_DIM:
        raise ValueError(
            f"Kronecker quantization left_trans_dim and right_trans_dim should be less than {MAX_SUPPORT_DIM}"
        )

    if (a - b) * m > MAX_SUPPORT_DIM:
        return MAX_SUPPORT_DIM, m * n // MAX_SUPPORT_DIM

    return a - b, a + b


@register_scheme("W4A4_MXFP4_FLATQUANT", "linear")
class AscendW4A4MXFP4FlatQuantDynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A4_MXFP4_FLATQUANT_DYNAMIC."""

    def __init__(self):
        ensure_mxfp4_flatquant_linear_available("W4A4_MXFP4_FLATQUANT linear quantization")
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        self.max_supported_tp = vllm_config.quant_config.quant_description.get("max_supported_tp", 4)
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > self.max_supported_tp:
            raise ValueError(
                f"For W4A4_MXFP4_FLATQUANT, TP size ({self.tp_size}) is not supported. "
                f"Max supported TP size is {self.max_supported_tp}, "
                f"according to the max_supported_tp parameter in quant_description."
            )

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        if input_size % 2 != 0:
            raise ValueError(f"input_size ({input_size}) must be divisible by 2 for fp4 packing")
        self.input_size = input_size
        params_dict = {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}

        return params_dict

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        params_dict = {}
        layer_type = kwargs.get("layer_type")
        if layer_type == "row":
            origin_size = self.input_size * self.tp_size
            _, right_trans_dim = get_decompose_dim(origin_size // self.max_supported_tp, self.max_supported_tp)
            left_trans_dim = origin_size // right_trans_dim
        else:
            left_trans_dim, right_trans_dim = get_decompose_dim(self.input_size, 1)

        params_dict["left_trans"] = torch.empty(left_trans_dim, left_trans_dim, dtype=params_dtype)
        params_dict["right_trans"] = torch.empty(right_trans_dim, right_trans_dim, dtype=params_dtype)
        params_dict["clip_ratio"] = torch.empty(1, dtype=torch.float32)
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, input_size // self.group_size, dtype=torch.uint8)

        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
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
        x_reshaped = x.view(-1, left_dim, right_dim)
        x_quantized_fp4, pertoken_scale = torch_npu.npu_kronecker_quant(
            x_reshaped,
            layer.left_trans,
            layer.right_trans,
            layer.aclnn_clip_ratio,
            dst_dtype=torch_npu.float4_e2m1fn_x2,
        )

        output = torch_npu.npu_quant_matmul(
            x_quantized_fp4,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=original_dtype,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            group_sizes=[1, 1, self.group_size],
        )
        output = output.view(*input_shape[:-1], -1)
        return output

    def process_weights_after_loading(self, layer):
        if isinstance(layer, RowParallelLinear):
            """
            Process weights after loading with TP diagonal block extraction. 
            This is the special weight loading logic for FlatQuant row parallelism. 
            """
            left_dim = layer.left_trans.data.shape[0]
            # Calculate block sizes
            left_block_size = left_dim // layer.tp_size
            # Extract diagonal block for current rank
            layer.left_trans.data = layer.left_trans.data[
                layer.tp_rank * left_block_size : (layer.tp_rank + 1) * left_block_size,
                layer.tp_rank * left_block_size : (layer.tp_rank + 1) * left_block_size,
            ]

        layer.weight_scale.data = layer.weight_scale.data.view(-1, layer.weight_scale.shape[-1] // 2, 2)
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)

        layer.left_trans = torch.nn.Parameter(layer.left_trans.data.t().contiguous())
        layer.right_trans = torch.nn.Parameter(layer.right_trans.data)
        layer.clip_ratio = torch.nn.Parameter(layer.clip_ratio.data.to(torch.float32))
        layer.aclnn_clip_ratio = layer.clip_ratio.item()
