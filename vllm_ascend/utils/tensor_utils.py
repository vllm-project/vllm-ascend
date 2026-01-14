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

"""
Tensor format and shape utilities for vLLM Ascend.

This module provides functions for:
- Converting tensors to NZ format
- Padding and reshaping tensors
- Disposing tensors to free memory
"""

import torch
import torch_npu  # noqa: F401

from vllm_ascend import envs_ascend

ACL_FORMAT_FRACTAL_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29


def _round_up(x: int, align: int):
    # round up x to align, for example, if align is 16, x will be rounded up to 16, 32, 48, etc.
    # input: 15, 16 -> output: 16
    # input: 17, 16 -> output: 32
    # input: 30, 16 -> output: 32
    # input: 33, 16 -> output: 48
    # ...
    return (x + align - 1) // align * align


def _custom_pad(x, pad_dims):
    # pad the input tensor to the shape of pad_dims
    # input: (13, 30), pad_dims: [0, 2, 0, 3]
    # output: (16, 32)
    return torch.nn.functional.pad(x, pad_dims)


def _custom_reshape(x, target_shape):
    # reshape the input tensor to the shape of target_shape
    # input: (16, 32), target_shape: [1, 16, 2, 16]
    # output: (1, 16, 2, 16)
    return x.reshape(target_shape)


def _custom_transpose(x, dim1, dim2):
    # transpose the input tensor
    # input: (1, 16, 2, 16), dim1: 1, dim2: 2
    # output: (1, 2, 16, 16)
    return x.transpose(dim1, dim2)


def nd_to_nz_2d(in_tensor: torch.Tensor) -> torch.Tensor:
    # in_tensor: (13, 30)
    aux_dims = [1, 0, 0, 16]
    # aux_dims[1]: 16
    aux_dims[1] = _round_up(in_tensor.size(0), 16)
    # aux_dims[2]: 2
    aux_dims[2] = _round_up(in_tensor.size(1), 16) // 16

    # after: aux_dims: [1, 16, 2, 16]

    pad_dims = [0, 0, 0, 0]
    # pad_dims[1]: 2
    pad_dims[1] = _round_up(in_tensor.size(1), 16) - in_tensor.size(1)
    # pad_dims[3]: 3
    pad_dims[3] = _round_up(in_tensor.size(0), 16) - in_tensor.size(0)

    # after: pad_dims: [0, 2, 0, 3]

    # return: (1, 2, 16, 16)
    return _custom_transpose(
        _custom_reshape(_custom_pad(in_tensor, pad_dims), aux_dims), 1,
        2).contiguous()


def nd_to_nz_spec(mask_tensor: torch.Tensor) -> torch.Tensor:
    num_tokens = mask_tensor.shape[0]
    max_seq_len = mask_tensor.shape[1]

    tokens_pad = (num_tokens + 15) // 16 * 16
    max_seq_len_pad = (max_seq_len + 15) // 16 * 16

    mask_tensor_pad = \
        torch.zeros((1, tokens_pad, max_seq_len_pad), dtype=mask_tensor.dtype, device=mask_tensor.device)
    mask_tensor_pad[0][:num_tokens, :max_seq_len] = mask_tensor
    mask = mask_tensor_pad.reshape(
        (1, tokens_pad, max_seq_len_pad // 16, 16)).permute(0, 2, 1, 3)
    return mask


def aligned_16(tensor: torch.Tensor):
    """Aligned tensor for 310P"""

    # Get the size of the current 0th dimension
    n = tensor.size(0)

    # Calculate the aligned size
    n_aligned = ((n + 15) // 16) * 16

    # If already aligned, return the original tensor
    if n == n_aligned:
        return tensor

    # Create a new tensor with shape (n_aligned, H, W) and fill it with zeros
    new_tensor = torch.zeros(n_aligned,
                             *tensor.shape[1:],
                             dtype=tensor.dtype,
                             device=tensor.device)

    # Copy the original tensor to the first N positions of the new tensor
    new_tensor[:n] = tensor

    return new_tensor


def maybe_trans_nz(weight: torch.Tensor):
    if not envs_ascend.VLLM_ASCEND_ENABLE_NZ:
        # NZ is not enabled
        return weight
    if weight.dtype == torch.float:
        # fp32 can not support NZ
        return weight
    elif weight.dtype in {torch.bfloat16, torch.float16}:
        # bf16/fp16 will trans nz when VLLM_ASCEND_ENABLE_NZ is 2
        if envs_ascend.VLLM_ASCEND_ENABLE_NZ == 2:
            return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
        else:
            return weight
    else:
        # quant weight will trans nz by default
        return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)


def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0, ), device=x.device, dtype=x.dtype))


def dispose_layer(layer: any):
    for attr_name in dir(layer):
        attr_value = getattr(layer, attr_name)
        if isinstance(attr_value, torch.Tensor):
            dispose_tensor(attr_value)
