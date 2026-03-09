#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import math
from typing import Any

import torch
import torch_npu
from vllm.distributed import get_tensor_model_parallel_rank

from vllm_ascend.ops.linear import AscendRowParallelLinear
from vllm_ascend.quantization.methods.base import AscendLinearScheme

from .registry import register_scheme


@register_scheme("W8A8SC", "linear")
class AscendW8A8SCLinearMethod310(AscendLinearScheme):
    """310P-only W8A8SC static linear scheme.

    Notes:
      - This scheme is discovered via 310P local registry.
    """

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.float16,
    ) -> dict[str, Any]:
        """
        Get the weight tensors for the W8A8SC quantization scheme.

        Args:
            input_size: Size of the input dimension (k)
            output_size: Size of the output dimension (n)
            params_dtype: Data type for parameters, default is torch.float16

        Returns:
            A dictionary containing:
            - "weight": The compressed weight tensor with shape [c], where c is greater than 0
              and not larger than k * n
            - "index": Compression index generated simultaneously with compressed weights,
              with shape [x], where x = k_index * n_index * 8, k_index = ceil(k1 / tilingK),
              n_index = ceil(n1 / tilingN), k1 = k / 32, n1 = n / 16
            - "info": Compression information with length 5, containing compression block
              information tilingN, tilingK, original shape of the pre-compression x2 matrix,
              and identifier for the compression block traversal direction
        """
        self.input_size = input_size
        index_len = math.ceil(input_size / 256) * math.ceil(output_size / 128) * 8
        return {
            "weight": torch.empty(input_size * output_size, dtype=torch.int8),
            "index": torch.empty(index_len, dtype=torch.int8),
            "info": torch.empty(5, dtype=torch.int64),
        }

    def get_pertensor_param(self, params_dtype: torch.dtype) -> dict[str, Any]:
        return {
            "input_scale": torch.empty(1, dtype=params_dtype),
            "input_offset": torch.empty(1, dtype=torch.int8),
        }

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {
            "quant_bias": torch.empty(output_size, dtype=torch.int32),
            "deq_scale": torch.empty(output_size, dtype=torch.int64),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            x = torch.ops.vllm.quantize(
                x,
                layer.aclnn_input_scale,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
            )

        return torch_npu.npu_matmul_compress_dequant(
            x,
            layer.weight,
            layer.index,
            layer.quant_bias,
            layer.deq_scale,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.aclnn_input_scale = layer.input_scale.data.repeat(self.input_size)
        layer.aclnn_input_scale_reciprocal = 1.0 / layer.aclnn_input_scale.data
        layer.aclnn_input_offset = layer.input_offset.data.repeat(self.input_size).to(layer.aclnn_input_scale.dtype)
        layer.deq_scale.data = layer.deq_scale.data.unsqueeze(0).to(torch.uint64)
        layer.quant_bias.data = layer.quant_bias.data.unsqueeze(0)
        # Only apply bias on row_parallel_linear when tp_rank is 0.
        # torch_npu.npu_matmul_compress_dequant's quant_bias cannot be None.
        if isinstance(layer, AscendRowParallelLinear) and get_tensor_model_parallel_rank() != 0:
            layer.quant_bias.data = torch.zeros_like(layer.quant_bias)
