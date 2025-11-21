# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch_npu
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsScheme
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

BEFORE_INIT = 0
AFTER_INIT = 1


class AscendCompressedTensorsW8A8Int8LinearMethod(CompressedTensorsScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(self, strategy: str, is_static_input_scheme: bool,
                 input_symmetric: bool):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric

    @classmethod
    def get_min_capability(cls) -> int:
        # turing and up
        return 75

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        self.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=torch.int8),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)

        layer.register_parameter("weight", weight)

        if self.strategy == QuantizationStrategy.TENSOR:
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes),
                dtype=torch.float32
                if params_dtype == torch.float16 else torch.bfloat16),
                                                   weight_loader=weight_loader)
            weight_offset = None
        else:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                 dtype=torch.float32 if params_dtype
                                 == torch.float16 else torch.bfloat16),
                output_dim=0,
                weight_loader=weight_loader)
            weight_offset = ChannelQuantScaleParameter(
                data=torch.zeros((sum(output_partition_sizes), 1),
                                 dtype=torch.float32 if params_dtype
                                 == torch.float16 else torch.bfloat16),
                output_dim=0,
                weight_loader=weight_loader)

        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_offset", weight_offset)

        self.empty_out = torch.empty(1, dtype=params_dtype)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        weight_scale = layer.weight_scale
        if getattr(layer, 'throw_dequant', False):
            weight_scale = weight_scale.to(torch.float32)
        weight_offset = layer.weight_offset
        weight = torch_npu.npu_format_cast(weight.t().contiguous(), 29)
        layer.weight = Parameter(weight, requires_grad=False)

        layer.weight_scale = Parameter(weight_scale.view(-1),
                                       requires_grad=False)
        layer.weight_offset = Parameter(weight_offset.view(-1).float(),
                                        requires_grad=False)
        return

    def apply_weights(
            self, layer: torch.nn.Module, x: torch.Tensor,
            bias: Optional[torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, Any]]:

        # activation per-token dynamic quant
        if isinstance(x, Dict):
            x_int8 = x.get('x_int8')
            pertoken_scale = x.get('pertoken_scale')
        else:
            x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)

        throw_dequant = getattr(layer, 'throw_dequant', False)
        if throw_dequant and bias is None:
            out = (torch_npu.npu_quant_matmul(x_int8,
                                              layer.weight,
                                              layer.weight_scale,
                                              bias=None,
                                              output_dtype=torch.int32),
                   pertoken_scale)
        else:
            out = torch_npu.npu_quant_matmul(x_int8,
                                             layer.weight,
                                             layer.weight_scale,
                                             offset=None,
                                             pertoken_scale=pertoken_scale,
                                             bias=bias,
                                             output_dtype=torch.bfloat16)
        return out
