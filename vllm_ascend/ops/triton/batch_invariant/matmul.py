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
def matmul_bias_persistent_kernel(
    # Input tensor pointers
    x_ptr,
    y_ptr,
    bias_ptr,
    output_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Stride information
    stride_xm,
    stride_xk,  # Strides of x: [M, K]
    stride_yk,
    stride_yn,  # Strides of y: [K, N]
    stride_bias,  # Stride of bias: [N]
    stride_outm,
    stride_outn,  # Strides of output: [M, N]
    # Whether to use bias
    has_bias: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # Row block ID
    pid_n = tl.program_id(1)  # Column block ID

    # Calculate the starting position of the current block in the matrix
    rm_start = pid_m * BLOCK_M
    rn_start = pid_n * BLOCK_N

    # Create index ranges
    rm = rm_start + tl.arange(0, BLOCK_M)  # Row index range [BLOCK_M]
    rn = rn_start + tl.arange(0, BLOCK_N)  # Column index range [BLOCK_N]
    rk = tl.arange(0, BLOCK_K)  # K dimension index range [BLOCK_K]

    # Initialize accumulator to 0
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension, processing BLOCK_K elements per iteration
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        # Calculate pointer offsets for x (row-major)
        x_ptrs = x_ptr + rm[:, None] * stride_xm + (rk[None, :] + k_start) * stride_xk
        # Calculate pointer offsets for y (row-major)
        y_ptrs = y_ptr + (rk[:, None] + k_start) * stride_yk + rn[None, :] * stride_yn

        # Create masks to prevent out-of-bounds access
        x_mask = (rm[:, None] < M) & ((rk[None, :] + k_start) < K)
        y_mask = ((rk[:, None] + k_start) < K) & (rn[None, :] < N)

        # Load data chunks from global memory
        x_chunk = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        y_chunk = tl.load(y_ptrs, mask=y_mask, other=0.0).to(tl.float32)

        # Compute matrix multiplication accumulation
        acc += tl.dot(x_chunk, y_chunk, allow_tf32=False)

    # Add bias if the has_bias flag is set
    if has_bias:
        # Load bias values (broadcast to all rows)
        bias_ptrs = bias_ptr + rn * stride_bias
        bias_mask = rn < N
        bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
        # Add bias to accumulator (automatic broadcasting)
        acc += bias_vals[None, :]

    # Calculate output pointer positions
    out_ptrs = output_ptr + rm[:, None] * stride_outm + rn[None, :] * stride_outn
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Store result to global memory
    tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=out_mask)


def matmul_persistent(x, y, bias=None):
    """
    Implement matrix multiplication with optional bias using Triton: x @ y + bias (if bias is not None)

    Parameters:
        x: torch.Tensor, shape [M, K]
        y: torch.Tensor, shape [K, N]
        bias: torch.Tensor, shape [N] or None

    Returns:
        output: torch.Tensor, shape [M, N]
    """
    # Validate input shapes
    assert x.dim() == 2, "x must be a 2D tensor"
    assert y.dim() == 2, "y must be a 2D tensor"
    assert x.shape[1] == y.shape[0], f"Matrix dimension mismatch: x.shape[1]={x.shape[1]}, y.shape[0]={y.shape[0]}"

    # Convert tensors to contiguous memory layout.
    # This prevents transposed tensors from causing incorrect stride() values,
    # which would lead to miscalculated data transfer volumes in subsequent operations.
    x = x.contiguous()
    y = y.contiguous()

    M, K = x.shape
    _, N = y.shape
    # Validate bias shape (if not None)
    if bias is not None:
        assert bias.dim() == 1, "bias must be a 1D tensor"
        assert y.shape[1] == bias.shape[0], (
            f"Bias dimension mismatch: y.shape[1]={y.shape[1]}, bias.shape[0]={bias.shape[0]}"
        )

    # Allocate output tensor (same data type as x)
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Define block sizes (can be adjusted based on hardware)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

    # Calculate grid size (one thread per block)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Handle case when bias is None
    if bias is None:
        # Create a dummy bias tensor (will not be used as has_bias=False)
        dummy_bias = torch.empty(0, dtype=x.dtype, device=x.device)
        has_bias = False
        bias_stride = 0
        bias_to_pass = dummy_bias
    else:
        has_bias = True
        bias_stride = bias.stride(0)
        bias_to_pass = bias
    # Launch kernel
    matmul_bias_persistent_kernel[grid](
        x,
        y,
        bias_to_pass,
        output,  # Input/Output tensors
        M,
        N,
        K,  # Matrix dimensions
        x.stride(0),
        x.stride(1),  # Strides of x
        y.stride(0),
        y.stride(1),  # Strides of y
        bias_stride,  # Stride of bias (0 if bias is None)
        output.stride(0),
        output.stride(1),  # Strides of output
        has_bias,  # Flag indicating whether to use bias
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output


@triton.jit
def linear_persistent_kernel(
    a_ptr,  # Pointer to tensor a, shape [M, K]
    b_ptr,  # Pointer to tensor b, shape [N, K]
    c_ptr,  # Pointer to output tensor c, shape [M, N]
    M,  # Number of rows in tensor a
    N,  # Number of rows in tensor b (number of columns in output c)
    K,  # Number of columns in both tensor a and tensor b
    stride_am,  # Stride of tensor a along dimension M (typically K)
    stride_ak,  # Stride of tensor a along dimension K (typically 1)
    stride_bn,  # Stride of tensor b along dimension N (typically K)
    stride_bk,  # Stride of tensor b along dimension K (typically 1)
    stride_cm,  # Stride of tensor c along dimension M (typically N)
    stride_cn,  # Stride of tensor c along dimension N (typically 1)
    BLOCK_M: tl.constexpr,  # Block size for M dimension
    BLOCK_N: tl.constexpr,  # Block size for N dimension
    BLOCK_K: tl.constexpr,  # Block size for K dimension
    NUM_BLOCKS_M: tl.constexpr,  # New: Number of blocks in M dimension
    NUM_BLOCKS_N: tl.constexpr,  # New: Number of blocks in N dimension
    GRID_SIZE: tl.constexpr,  # New: Fixed 1D grid size
):
    # Get current program's 1D index (1D grid)
    pid = tl.program_id(0)
    total_blocks = NUM_BLOCKS_M * NUM_BLOCKS_N  # Total number of output blocks

    # Loop over multiple blocks assigned to the current program
    for block_index in range(pid, total_blocks, GRID_SIZE):
        # Convert 1D block index to 2D coordinates (m_block, n_block)
        m_block = block_index // NUM_BLOCKS_N
        n_block = block_index % NUM_BLOCKS_N

        # Calculate starting indices of the current output block
        start_m = m_block * BLOCK_M
        start_n = n_block * BLOCK_N

        # Create row and column index ranges within the current block
        m_indices = start_m + tl.arange(0, BLOCK_M)
        n_indices = start_n + tl.arange(0, BLOCK_N)

        # Create masks to handle boundaries
        m_mask = m_indices < M
        n_mask = n_indices < N

        # Initialize accumulator to 0
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over K dimension with step size BLOCK_K
        for k_offset in range(0, K, BLOCK_K):
            k_indices = k_offset + tl.arange(0, BLOCK_K)
            k_mask = k_indices < K

            # Load block of tensor a: shape [BLOCK_M, BLOCK_K]
            a_ptrs = a_ptr + m_indices[:, None] * stride_am + k_indices[None, :] * stride_ak
            a_vals = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            # Load block of tensor b: shape [BLOCK_N, BLOCK_K]
            b_ptrs = b_ptr + n_indices[:, None] * stride_bn + k_indices[None, :] * stride_bk
            b_vals = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

            # Explicitly transpose b matrix using tl.trans: shape becomes [BLOCK_K, BLOCK_N]
            b_vals_transposed = tl.trans(b_vals)

            # Compute matrix multiplication: a_vals × b_vals_transposed
            product = tl.dot(a_vals, b_vals_transposed)
            acc += product
        # Store result to output tensor c
        c_ptrs = c_ptr + m_indices[:, None] * stride_cm + n_indices[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def linear_persistent(x, y):
    """
    Implement matrix multiplication with Triton: x @ y^T
    Uses a fixed-size 1D grid

    Parameters:
        x: torch.Tensor, shape [M, K]
        y: torch.Tensor, shape [N, K]

    Returns:
        output: torch.Tensor, shape [M, N]
    """
    # Validate input shapes
    assert x.dim() == 2, "x must be a 2D tensor"
    assert y.dim() == 2, "y must be a 2D tensor"
    assert x.shape[1] == y.shape[1], f"Matrix dimension mismatch: x.shape[1]={x.shape[1]}, y.shape[1]={y.shape[1]}"

    M, K = x.shape
    N, _ = y.shape

    # Allocate output tensor (same data type as x)
    output = torch.zeros((M, N), dtype=x.dtype, device=x.device)

    grid_size = driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"] // 2

    # Define block sizes (can be adjusted based on hardware)
    BLOCK_K = 256
    if x.dtype == torch.float32:
        BLOCK_K = BLOCK_K // 2
    grid_size_div4 = grid_size // 4
    if M == 0 or N == 0:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256
    elif M < 256:
        BLOCK_M = M
        if grid_size * 128 <= N:
            if M <= 128:
                BLOCK_N = 256
            else:
                BLOCK_N = 128
        elif grid_size * 32 >= N:
            if M > N:
                BLOCK_M = triton.cdiv(M, grid_size_div4)
                BLOCK_N = triton.cdiv(N, 4)
            else:
                BLOCK_M = triton.cdiv(M, 4)
                BLOCK_N = triton.cdiv(N, grid_size_div4)
        else:
            BLOCK_N = triton.next_power_of_2(triton.cdiv(N, grid_size))
    elif M >= 256 and M < 1024:
        if M < N:
            BLOCK_M = 256
            nums_m = triton.cdiv(M, BLOCK_M)
            nums_n = grid_size // nums_m
            if 128 * nums_n <= N:
                BLOCK_N = 128
            else:
                BLOCK_N = min(triton.next_power_of_2(triton.cdiv(N, nums_n)), 128)
        else:
            BLOCK_M = min(triton.cdiv(M, grid_size_div4), 256)
            BLOCK_N = min(triton.cdiv(N, 4), 128)
    else:
        if M > N:
            BLOCK_M, BLOCK_N = 256, 128
            nums_m = triton.cdiv(M, BLOCK_M)
            nums_n = triton.cdiv(N, BLOCK_N)
            if nums_m * nums_n < grid_size:
                BLOCK_M = triton.cdiv(M, grid_size_div4)
                BLOCK_N = triton.cdiv(N, 4)
        else:
            BLOCK_M, BLOCK_N = 128, 256

    # Calculate number of blocks per dimension (ceil division)
    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)

    # Set fixed 1D grid size
    grid = (grid_size,)

    # Launch kernel
    linear_persistent_kernel[grid](
        a_ptr=x,
        b_ptr=y,
        c_ptr=output,
        M=M,
        N=N,
        K=K,
        stride_am=x.stride(0),
        stride_ak=x.stride(1),
        stride_bn=y.stride(0),
        stride_bk=y.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BLOCKS_M=num_blocks_m,  # Number of blocks in M dimension
        NUM_BLOCKS_N=num_blocks_n,  # Number of blocks in N dimension
        GRID_SIZE=grid_size,  # Fixed grid size
    )

    return output


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    # Process each batch separately with our persistent kernel
    if a.ndim == 3 and b.ndim == 3:
        results = []
        for i in range(a.shape[0]):
            results.append(matmul_persistent(a[i], b[i]))
        result = torch.stack(results, dim=0)

        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(f"bmm_batch_invariant expects 3D tensors, got shapes {a.shape} and {b.shape}")


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def matmul_batch_invariant(a, b, *, out=None):
    # torch.matmul can handle various dimensions
    # For 2D x 2D, it's the same as matmul
    if a.ndim == 2 and b.ndim == 2:
        result = matmul_persistent(a, b)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim == 3 and b.ndim == 3:
        # Handle batched case like bmm
        return bmm_batch_invariant(a, b, out=out)
    elif a.ndim == 3 and b.ndim == 2:
        # Handle 3D x 2D: common for linear layers
        # (batch, seq, hidden) @ (hidden, out) -> (batch, seq, out)
        # Reshape to 2D, do mm, reshape back
        batch, seq, hidden = a.shape
        a_2d = a.reshape(-1, hidden)
        result_2d = matmul_persistent(a_2d, b)
        result = result_2d.reshape(batch, seq, -1)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim == 2 and b.ndim == 3:
        # Handle 2D x 3D: (M, K) @ (B, K, N) -> (B, M, N)
        # By broadcasting `a` to 3D, we can reuse the batched matrix
        # multiplication logic.
        a_expanded = a.unsqueeze(0).expand(b.shape[0], -1, -1)
        return bmm_batch_invariant(a_expanded, b, out=out)
    elif a.ndim == 4 and b.ndim == 4:
        # Handle 4D attention tensors: [batch, heads, seq, dim]
        # Reshape to 3D, process, reshape back
        batch, heads, seq_a, dim_a = a.shape
        _, _, dim_b, seq_b = b.shape

        # Reshape to [batch*heads, seq_a, dim_a]
        a_3d = a.reshape(batch * heads, seq_a, dim_a)
        b_3d = b.reshape(batch * heads, dim_b, seq_b)

        # Do batched matmul
        result_3d = bmm_batch_invariant(a_3d, b_3d)

        # Reshape back to [batch, heads, seq_a, seq_b]
        result = result_3d.reshape(batch, heads, seq_a, seq_b)

        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(
            f"matmul_batch_invariant currently only supports 2D x 2D, 3D x 3D, "
            f"3D x 2D, 2D x 3D, and 4D x 4D, "
            f"got shapes {a.shape} and {b.shape}"
        )


def linear_batch_invariant(input_, weight, bias=None):
    output = linear_persistent(input_, weight)

    if bias is not None:
        output = output + bias
    return output
