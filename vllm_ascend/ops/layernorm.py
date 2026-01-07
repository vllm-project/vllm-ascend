#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from typing import Optional, Tuple, Union

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def add_rmsnorm_bias_kernel(input_ptr, residual_ptr, norm_weight_ptr,
                            norm_bias_ptr, output_ptr, output2_ptr, batch_size,
                            hidden_size: tl.constexpr, eps: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    valid_mask = cols < hidden_size
    norm_weight_values = tl.load(norm_weight_ptr + cols,
                                 mask=valid_mask,
                                 other=0.0)
    input_offsets = row_start * hidden_size + cols
    for _ in tl.range(row_start, batch_size, row_step):
        # add
        buffered_values = tl.load(input_ptr + input_offsets,
                                  mask=valid_mask,
                                  other=0.0)
        buffered_values += tl.load(residual_ptr + input_offsets,
                                   mask=valid_mask,
                                   other=0.0)
        tl.store(output2_ptr + input_offsets, buffered_values, mask=valid_mask)
        buffered_values = buffered_values.to(tl.float32)
        # rmsnorm
        squares = buffered_values * buffered_values
        variance = tl.sum(squares) / hidden_size
        reciprocal_std = 1 / tl.sqrt(variance + eps)
        buffered_values = buffered_values * reciprocal_std
        buffered_values = buffered_values * norm_weight_values
        # add bias
        norm_bias_values = tl.load(norm_bias_ptr + cols,
                                   mask=valid_mask,
                                   other=0.0)
        buffered_values = buffered_values + norm_bias_values
        tl.store(output_ptr + input_offsets, buffered_values, mask=valid_mask)

        input_offsets += row_step * hidden_size


def add_rmsnorm_bias(input: torch.Tensor, residual: torch.Tensor,
                     norm_weight: torch.Tensor,
                     norm_bias: Optional[torch.Tensor],
                     eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    input = input.contiguous()
    residual = residual.contiguous()
    norm_weight = norm_weight.contiguous()
    norm_bias = norm_bias.contiguous(
    ) if norm_bias is not None else torch.zeros_like(norm_weight).contiguous()
    num_vectorcore = 40
    batch_size = input.shape[0]
    hidden_size = input.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    n_rows = min(batch_size, num_vectorcore)
    output = torch.empty(batch_size,
                         hidden_size,
                         device=input.device,
                         dtype=input.dtype)
    output2 = torch.empty(batch_size,
                          hidden_size,
                          device=input.device,
                          dtype=input.dtype)
    add_rmsnorm_bias_kernel[(n_rows, 1, 1)](input, residual, norm_weight,
                                            norm_bias, output, output2,
                                            batch_size, hidden_size, eps,
                                            BLOCK_SIZE)
    return output, output2


def add_rmsnorm_bias_impl_fake(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input)
    output2 = torch.empty_like(input)
    return output, output2


direct_register_custom_op(op_name="add_rmsnorm_bias",
                          op_func=add_rmsnorm_bias,
                          fake_impl=add_rmsnorm_bias_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")


class AscendRMSNorm(RMSNorm):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        vllm_config = get_current_vllm_config()
        self.bias = None
        # quantization with anti_method m4 will generate none-zero norm bias
        if vllm_config.quant_config is not None and \
                any("norm.bias" in name for name in vllm_config.quant_config.quant_description.keys()):
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size),
                                           requires_grad=False)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
        if residual is not None:
            if get_ascend_device_type() == AscendDeviceType._310P:
                orig_dtype = residual.dtype
                x = x + residual.to(x.dtype)
                residual = x.to(orig_dtype)
                x, _ = torch_npu.npu_rms_norm(x, self.weight,
                                              self.variance_epsilon)
            else:
                x, residual = torch.ops.vllm.add_rmsnorm_bias(
                    input=x,
                    residual=residual,
                    norm_weight=self.weight,
                    norm_bias=self.bias,
                    eps=self.variance_epsilon)
            return x, residual
        residual = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        x, residual = torch.ops.vllm.add_rmsnorm_bias(
            input=x,
            residual=residual,
            norm_weight=self.weight,
            norm_bias=self.bias,
            eps=self.variance_epsilon)
        return x


class AscendGemmaRMSNorm(GemmaRMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
        if residual is not None:
            if get_ascend_device_type() == AscendDeviceType._310P:
                orig_dtype = residual.dtype
                x = x + residual.to(x.dtype)
                residual = x.to(orig_dtype)
                x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight,
                                              self.variance_epsilon)
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(
                    x, residual, 1.0 + self.weight, self.variance_epsilon)
            return x, residual

        x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight,
                                      self.variance_epsilon)
        return x
