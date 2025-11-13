# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional

import torch
import torch_npu
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsScheme
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter)

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_enable_nz

logger = init_logger(__name__)


class CompressedTensorsW8A8Dynamic(CompressedTensorsScheme):

    def __init__(self) -> None:
        # aclnn quant matmul requires to transpose matrix B, set to true by default.
        self.transpose_weight = True

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

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, tuple):
            output_dtype = x.dtype
            quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(x)
        else:
            output_dtype = layer.weight_scale.dtype
            quantized_x, dynamic_scale = x

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=bias,
            output_dtype=output_dtype,
        )
        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # cast quantized weight tensors in NZ format for higher inference speed
        if is_enable_nz():
            layer.weight.data = torch_npu.npu_format_cast(
                layer.weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
