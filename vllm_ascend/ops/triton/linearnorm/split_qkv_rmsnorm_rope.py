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
"""
Fused Triton kernel: QKV Split + per-head RMSNorm + RoPE (neox-style).

This module fuses three operations into a single kernel to eliminate
intermediate global memory reads/writes between RMSNorm and RoPE:

  QKV tensor ─┬─ Q ─→ RMSNorm ─→ RoPE ─→ Q_out
               ├─ K ─→ RMSNorm ─→ RoPE ─→ K_out
               └─ V ─→ copy ────────────→ V_out

Supports both full RoPE (ROPE_DIM == HEAD_DIM) and partial RoPE
(ROPE_DIM < HEAD_DIM, e.g., Qwen3-Next with partial_rotary_factor=0.5).

For partial RoPE, only the first ROPE_DIM dimensions of each attention head
are rotated; the remaining HEAD_DIM - ROPE_DIM dimensions pass through
with RMSNorm applied but no rotation.

Design decisions:
  - All intermediate computation (RMSNorm, RoPE) is done in float32 for
    numerical precision, matching the standalone _triton_rope kernel behavior.
  - cos/sin are expected in duplicated HuggingFace format:
    [cos_θ₀, cos_θ₁, ..., cos_θ_{d/2-1}, cos_θ₀, cos_θ₁, ..., cos_θ_{d/2-1}]
  - The kernel uses a 2D grid (n_rows, n_cols) for row/column parallelism
    across the batch and hidden dimensions respectively.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def split_qkv_rmsnorm_rope_kernel(
        # Input/output pointers
        input_ptr,
        sin_ptr,
        cos_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        # Weight pointers
        q_weight_ptr,
        q_bias_ptr,
        k_weight_ptr,
        k_bias_ptr,
        # Scalar args
        batch_size,
        # Compile-time constant args
        q_hidden_size: tl.constexpr,
        kv_hidden_size: tl.constexpr,
        total_hidden_size: tl.constexpr,
        eps: tl.constexpr,
        Q_BLOCK_SIZE: tl.constexpr,
        KV_BLOCK_SIZE: tl.constexpr,
        BIAS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROPE_DIM: tl.constexpr,
        HALF_ROPE_DIM: tl.constexpr,
):
    """
    Fused QKV Split + RMSNorm + RoPE kernel.

    Grid layout: (n_rows, n_cols) where
      - n_rows handles batch dimension parallelism
      - n_cols handles hidden dimension parallelism (tiled by KV_BLOCK_SIZE)

    Requirements:
      - HEAD_DIM must be a power of 2 and equal to KV_BLOCK_SIZE
      - Q_BLOCK_SIZE must be divisible by HEAD_DIM
      - ROPE_DIM <= HEAD_DIM, and ROPE_DIM must be even
      - cos/sin are [1, batch_size, 1, ROPE_DIM] flattened, stride ROPE_DIM per token
    """
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)

    # ==================== Q Section ====================
    # Data flow: split Q from QKV → per-head RMSNorm → RoPE → store
    weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))

    input_offset = row_pid * total_hidden_size
    output_offset = row_pid * q_hidden_size
    input_offset_step = row_step * total_hidden_size
    output_offset_step = row_step * q_hidden_size

    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
        valid_mask = col_indices < q_hidden_size

        # Load Q slice: [Q_BLOCK_SIZE] → reshape to [num_heads_in_block, HEAD_DIM]
        input_values = (
            tl.load(
                input_ptr + input_offset + col_indices,
                mask=valid_mask,
                other=0.0,
                )
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )

        # Per-head RMSNorm: x_norm = x / sqrt(mean(x^2) + eps) * weight [+ bias]
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(
            Q_BLOCK_SIZE // HEAD_DIM, 1
        )
        normalized_values = input_values * reciprocal_std

        # Apply RMSNorm weight and optional bias (still in float32)
        if BIAS:
            output_values = normalized_values * weight_values + bias_values
        else:
            output_values = normalized_values * weight_values
        # output_values shape: [Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM], dtype: float32

        # --- RoPE (neox-style, in float32 for precision) ---
        # cos/sin layout: flattened [1, batch, 1, ROPE_DIM], stride = ROPE_DIM per token
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin_vals = (
            tl.load(sin_ptr + sc_offsets).to(tl.float32).reshape(1, ROPE_DIM)
        )
        cos_vals = (
            tl.load(cos_ptr + sc_offsets).to(tl.float32).reshape(1, ROPE_DIM)
        )

        # Start with full output (preserves passthrough dims for partial RoPE)
        roped_q = output_values

        # Extract the first ROPE_DIM columns for rotation
        rope_part = tl.extract_slice(
            output_values,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )

        # Neox-style rotation: out = [-x2, x1] * sin + [x1, x2] * cos
        # where x1 = first_half, x2 = second_half
        x1 = tl.extract_slice(
            rope_part,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            rope_part,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros(
            (Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.float32
        )
        cat_x = tl.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = tl.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        roped_part = cat_x * sin_vals + rope_part * cos_vals

        # Insert rotated part back (for full RoPE this replaces everything;
        # for partial RoPE this only replaces the first ROPE_DIM columns)
        roped_q = tl.insert_slice(
            roped_q,
            roped_part,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )

        # Store Q output (convert float32 → output dtype at store boundary)
        tl.store(
            q_ptr + output_offset + col_indices,
            roped_q.to(q_ptr.dtype.element_ty).reshape(Q_BLOCK_SIZE),
            mask=valid_mask,
            )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # ==================== K Section ====================
    # Data flow: split K from QKV → per-head RMSNorm → RoPE → store
    weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

    input_offset = row_pid * total_hidden_size + q_hidden_size
    output_offset = row_pid * kv_hidden_size
    output_offset_step = row_step * kv_hidden_size

    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size

        # Load K slice: [KV_BLOCK_SIZE] → reshape to [num_kv_heads_in_block, HEAD_DIM]
        input_values = (
            tl.load(
                input_ptr + input_offset + col_indices,
                mask=valid_mask,
                other=0.0,
                )
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )

        # Per-head RMSNorm
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(
            KV_BLOCK_SIZE // HEAD_DIM, 1
        )
        normalized_values = input_values * reciprocal_std

        if BIAS:
            output_values = normalized_values * weight_values + bias_values
        else:
            output_values = normalized_values * weight_values

        # --- RoPE for K (same logic as Q, using KV_BLOCK_SIZE) ---
        # NOTE: PR #6109 had a bug here using Q_BLOCK_SIZE instead of KV_BLOCK_SIZE
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin_vals = (
            tl.load(sin_ptr + sc_offsets).to(tl.float32).reshape(1, ROPE_DIM)
        )
        cos_vals = (
            tl.load(cos_ptr + sc_offsets).to(tl.float32).reshape(1, ROPE_DIM)
        )

        roped_k = output_values

        rope_part = tl.extract_slice(
            output_values,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )

        x1 = tl.extract_slice(
            rope_part,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            rope_part,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros(
            (KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.float32
        )
        cat_x = tl.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = tl.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        roped_part = cat_x * sin_vals + rope_part * cos_vals

        roped_k = tl.insert_slice(
            roped_k,
            roped_part,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )

        # Store K output
        tl.store(
            k_ptr + output_offset + col_indices,
            roped_k.to(k_ptr.dtype.element_ty).reshape(KV_BLOCK_SIZE),
            mask=valid_mask,
            )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # ==================== V Section ====================
    # Data flow: split V from QKV → copy to output (no norm, no RoPE)
    input_offset = (
            row_pid * total_hidden_size + q_hidden_size + kv_hidden_size
    )
    output_offset = row_pid * kv_hidden_size

    for _ in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = tl.load(
            input_ptr + input_offset + col_indices,
            mask=valid_mask,
            other=0.0,
            )
        tl.store(
            v_ptr + output_offset + col_indices,
            input_values,
            mask=valid_mask,
            )
        input_offset += input_offset_step
        output_offset += output_offset_step


def split_qkv_rmsnorm_rope_impl(
        input: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        q_hidden_size: int,
        kv_hidden_size: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Host-side launcher for the fused QKV Split + RMSNorm + RoPE kernel.

    Args:
        input: [batch_size, q_hidden_size + 2 * kv_hidden_size]
        sin, cos: [1, batch_size, 1, rotary_dim] or any shape that flattens
                  to [batch_size * rotary_dim] with stride rotary_dim per token.
                  Values are in duplicated HuggingFace format.
        q_weight, k_weight: [head_dim] per-head RMSNorm weights
        q_hidden_size: total Q dimension (num_q_heads * head_dim)
        kv_hidden_size: total KV dimension (num_kv_heads * head_dim)
        head_dim: dimension per attention head (must be power of 2)
        rotary_dim: number of dimensions to apply RoPE
                    (= head_dim for full RoPE, < head_dim for partial RoPE)
        eps: RMSNorm epsilon
        q_bias, k_bias: optional per-head biases [head_dim]

    Returns:
        (q_output, k_output, v_output) each as [batch_size, hidden_size]
    """
    KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert head_dim == KV_BLOCK_SIZE, (
        f"head_dim ({head_dim}) must be a power of 2"
    )
    assert q_hidden_size % kv_hidden_size == 0, (
        f"q_hidden_size ({q_hidden_size}) must be divisible by "
        f"kv_hidden_size ({kv_hidden_size})"
    )
    assert rotary_dim <= head_dim, (
        f"rotary_dim ({rotary_dim}) must be <= head_dim ({head_dim})"
    )
    assert rotary_dim % 2 == 0, (
        f"rotary_dim ({rotary_dim}) must be even"
    )

    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2

    q_output = torch.empty(
        batch_size, q_hidden_size, device=input.device, dtype=input.dtype
    )
    k_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    v_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )

    n_cols = kv_hidden_size // KV_BLOCK_SIZE
    num_vectorcore = get_vectorcore_num()
    assert num_vectorcore % n_cols == 0, (
        f"num_vectorcore ({num_vectorcore}) must be divisible by "
        f"n_cols ({n_cols})"
    )
    n_rows = num_vectorcore // n_cols
    BIAS = q_bias is not None

    split_qkv_rmsnorm_rope_kernel[(n_rows, n_cols, 1)](
        input,
        sin,
        cos,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        BIAS,
        head_dim,
        rotary_dim,
        rotary_dim // 2,
        )
    return q_output, k_output, v_output


def split_qkv_rmsnorm_rope_impl_fake(
        input: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        q_hidden_size: int,
        kv_hidden_size: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for shape inference during Dynamo/AOT tracing."""
    batch_size = input.shape[0]
    q_output = torch.empty(
        batch_size,
        q_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    k_output = torch.empty(
        batch_size,
        kv_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    v_output = torch.empty(
        batch_size,
        kv_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    return q_output, k_output, v_output


direct_register_custom_op(
    op_name="qkv_rmsnorm_rope",
    op_func=split_qkv_rmsnorm_rope_impl,
    fake_impl=split_qkv_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
