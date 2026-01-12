# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/batch_invariant.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import torch
from triton.runtime import driver  # type: ignore
from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,  # 新增参数：总行数
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute RMS normalization along the last dimension of a 2D tensor.
    RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    Each program handles multiple rows of the input tensor.
    """
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    rows_per_program = (n_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_rows)

    for row_idx in range(start_row, end_row):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        output_row_start_ptr = output_ptr + row_idx * output_row_stride

        # Step 1: Compute sum of squares in float32 to avoid overflow
        sum_sq = tl.zeros([1], dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols

            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sq_vals = vals_f32 * vals_f32
            sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

        # Step 2: Compute RMS (root mean square) in float32
        mean_sq = sum_sq / n_cols
        rms = tl.sqrt(mean_sq + eps)
        inv_rms = 1.0 / rms

        # Step 3: Normalize and apply weight
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
            vals_f32 = vals.to(tl.float32)
            weight_f32 = weight.to(tl.float32)
            output_f32 = vals_f32 * inv_rms * weight_f32
            output = output_f32.to(vals.dtype)
            tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def rms_norm(
    input_: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute RMS normalization using Triton kernel with fixed grid size.

    RMS Norm normalizes the input by the root mean square and scales by weight:
    output = input / sqrt(mean(input^2) + eps) * weight

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tensor with RMS normalization applied along the last dimension
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input_.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input_.shape[-1]}) must match weight dimension ({weight.shape[0]})"
    )

    # Flatten all dimensions except the last one
    original_shape = input_.shape
    input_2d = input_.reshape(-1, input_.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d, dtype=input_.dtype)
    BLOCK_SIZE = 1024
    max_grid_size = driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"]

    grid = (min(n_rows, max_grid_size),)

    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


def rms_norm_batch_invariant(
    input_: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Batch-invariant wrapper for RMS normalization.

    This function provides a deterministic, batch-invariant implementation
    of RMS normalization for use with the batch_invariant mode.
    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        RMS normalized tensor
    """
    return rms_norm(input_, weight, eps=eps)
