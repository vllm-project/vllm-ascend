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

from typing import List, Optional

import torch
import torch_npu
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsScheme
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_310p, is_enable_nz

logger = init_logger(__name__)


def quant_per_tensor(in_tensor: torch.Tensor,
                     input_scale: torch.Tensor,
                     input_offset: torch.Tensor,
                     function=False):
    return torch_npu.npu_quantize(in_tensor, input_scale, input_offset,
                                  torch.qint8, -1, function)


class CompressedTensorsW8A8(CompressedTensorsScheme):

    def __init__(self) -> None:
        # aclnn quant matmul requires to transpose matrix B, set to true by default.
        self.transpose_weight = not is_310p()

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.output_partition_sizes = output_partition_sizes
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(output_size_per_partition,
                             input_size_per_partition,
                             dtype=torch.int8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((output_size_per_partition, 1),
                             dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=params_dtype),
            weight_loader=weight_loader,
        )
        input_scale[:] = torch.finfo(params_dtype).min
        layer.register_parameter("input_scale", input_scale)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        if x.dtype != torch.int8:
            x = quant_per_tensor(
                x,
                layer.aclnn_input_scale_reciprocal,
                None,
            )

        if is_310p():
            # On 300I Duo platform, we need transpose again if
            # using nz. This transpose can be skipped in torchair.
            output = torch_npu.npu_quant_matmul(
                x,
                layer.weight.data.transpose(1, 0),
                layer.deq_scale,
                bias=bias,
                output_dtype=layer.params_dtype,
            )
        else:
            output = torch_npu.npu_quant_matmul(
                x,
                layer.weight,
                layer.deq_scale,
                bias=bias,
                output_dtype=layer.params_dtype,
            )
        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.input_scale = torch.nn.Parameter(layer.input_scale.max(),
                                               requires_grad=False)
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        if is_enable_nz():
            layer.weight.data = torch_npu.npu_format_cast(
                layer.weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        deq_scale = layer.input_scale.data * layer.weight_scale.data
        layer.deq_scale = torch.nn.Parameter(deq_scale, requires_grad=False)
