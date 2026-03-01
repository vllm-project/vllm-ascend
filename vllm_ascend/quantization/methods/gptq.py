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
"""GPTQ quantization method for Ascend NPU.

This module implements GPTQ quantization for linear layers on Ascend NPU,
inheriting from vLLM's standard GPTQLinearMethod.
"""

import torch
import torch_npu
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod


def unpack_from_int32(
    weight: torch.Tensor,
    num_bits: int,
    packed_dim: int = 1,
) -> torch.Tensor:
    """
    Unpacks quantized weights from int32 format back to original bits.

    :param weight: The packed int32 tensor containing quantized weights
    :param num_bits: The number of bits used for quantization (<= 8)
    :param packed_dim: Dimension along which weights are packed (0 or 1), defaults to 1
    :return: Unpacked tensor with int8 dtype after applying offset correction
    """
    assert weight.dtype == torch.int32, f"Expecting `weight.dtype` is torch.int32 but got {weight.dtype}."
    assert num_bits <= 8, f"Expecting `num_bits` should not be larger than 8 but got {num_bits}."

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] * pack_factor),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[:, i::pack_factor] = (weight >> (num_bits * i)) & mask
    else:
        unpacked_weight = torch.zeros(
            (weight.shape[0] * pack_factor, weight.shape[1]),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[i::pack_factor, :] = (weight >> (num_bits * i)) & mask
    offset = pow(2, num_bits) // 2
    unpacked_weight = (unpacked_weight - offset).to(torch.int8)
    return unpacked_weight


class AscendGPTQLinearMethod(GPTQLinearMethod):
    """GPTQ Linear method for Ascend NPU.

    Inherits from vLLM's standard GPTQLinearMethod and overrides
    the process_weights_after_loading and apply methods for NPU.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Call parent's create_weights to use standard vLLM parameter creation
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        if self.quant_config.desc_act:
            raise ValueError("Currently, desc_act (True) is not supported by GPTQ quantization on npu.")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing for NPU.

        Unpacks qzeros and qweight from int32 format and converts to NPU format.
        """
        # Unpack qzeros from int32 to actual bit width
        layer.qzeros = torch.nn.Parameter(
            unpack_from_int32(
                layer.qzeros.data.contiguous(),
                self.quant_config.weight_bits,
                packed_dim=1,
            ).to(layer.scales.dtype),
            requires_grad=False,
        )

        # Apply offset correction based on checkpoint format
        # GPTQ v1 format uses offset of 1, v2 format uses offset of 0
        if not self.use_v2_format:
            layer.qzeros += 1

        # Unpack qweight from int32
        qweight_tmp = unpack_from_int32(layer.qweight.data.contiguous(), self.quant_config.weight_bits, packed_dim=0)

        # For 4-bit, convert to int4pack format for efficient NPU computation
        # For other bit widths, keep as int8
        if self.quant_config.weight_bits == 4:
            layer.qweight = torch.nn.Parameter(
                torch_npu.npu_convert_weight_to_int4pack(qweight_tmp.to(torch.int32)),
                requires_grad=False,
            )
        else:
            layer.qweight = torch.nn.Parameter(
                qweight_tmp,
                requires_grad=False,
            )

        # Convert g_idx to regular parameter for NPU
        layer.g_idx = torch.nn.Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward computation using NPU quantized matmul.

        Args:
            layer: The linear layer module.
            x: Input tensor.
            bias: Optional bias tensor.

        Returns:
            Output tensor after quantized linear operation.
        """
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros

        # Reshape input to 2D for matmul
        reshaped_x = x.reshape(-1, x.shape[-1])

        # Convert bfloat16 bias to float32 for NPU compatibility
        if bias is not None and bias.dtype == torch.bfloat16:
            bias = bias.float()

        # Calculate output shape
        # For 4-bit weight packed to int32(8 x int4)
        if self.quant_config.weight_bits == 4:
            out_shape = x.shape[:-1] + (qweight.shape[-1] * 8,)
        else:
            out_shape = x.shape[:-1] + (qweight.shape[-1],)

        # Use NPU quantized matmul operator
        out = torch_npu.npu_weight_quant_batchmatmul(
            reshaped_x,
            qweight,
            antiquant_scale=scales,
            antiquant_offset=qzeros,
            antiquant_group_size=self.quant_config.group_size,
            bias=bias,
        )

        return out.reshape(out_shape)
