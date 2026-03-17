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

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def minimax_qkv_crosshead_norm_rope_kernel(
    input_ptr,
    cos_sin_ptr,
    pos_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    qk_var_ptr,
    q_weight_ptr,
    k_weight_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    head_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for MiniMax cross-head QKNorm + partial RoPE.

    Performs:
    1. Split qkv -> q, k, v
    2. Cross-head RMSNorm on q (normalize across ALL q dims, not per-head)
    3. Cross-head RMSNorm on k (normalize across ALL k dims, not per-head)
    4. Partial RoPE on first rotary_dim dims of each head
    5. Output local variance for TP correction

    Cross-head RMSNorm differs from per-head RMSNorm:
    - Per-head: variance computed within each head independently
    - Cross-head: variance computed across ALL heads jointly, then weight applied
    """
    row_pid = tl.program_id(0)
    row_step = tl.num_programs(0)

    # Load q weight [q_hidden_size] and k weight [kv_hidden_size]
    q_weight = tl.load(q_weight_ptr + tl.arange(0, Q_BLOCK_SIZE),
                       mask=tl.arange(0, Q_BLOCK_SIZE) < q_hidden_size,
                       other=0.0)
    k_weight = tl.load(k_weight_ptr + tl.arange(0, KV_BLOCK_SIZE),
                       mask=tl.arange(0, KV_BLOCK_SIZE) < kv_hidden_size,
                       other=0.0)

    for row_idx in tl.range(row_pid, batch_size, row_step):
        # --- Q processing: split + cross-head RMSNorm + RoPE ---
        input_offset = row_idx * total_hidden_size
        q_col_indices = tl.arange(0, Q_BLOCK_SIZE)
        q_valid_mask = q_col_indices < q_hidden_size

        q_vals = tl.load(input_ptr + input_offset + q_col_indices,
                         mask=q_valid_mask, other=0.0).to(tl.float32)

        # Cross-head variance: mean of squares across ALL q dimensions
        q_squares_sum = tl.sum(q_vals * q_vals)
        q_var = q_squares_sum / q_hidden_size
        q_inv_rms = 1.0 / tl.sqrt(q_var + eps)

        # Normalize and apply weight
        q_normed = (q_vals * q_inv_rms * q_weight).to(tl.bfloat16)

        # Apply partial RoPE per head
        # cos_sin_cache layout: [max_pos, rotary_dim] with first half cos,
        # second half sin
        pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
        cos_offsets = pos_idx * rotary_dim + tl.arange(0, half_rotary_dim)
        sin_offsets = (pos_idx * rotary_dim + half_rotary_dim
                       + tl.arange(0, half_rotary_dim))
        cos = tl.load(cos_sin_ptr + cos_offsets).to(tl.float32)
        sin = tl.load(cos_sin_ptr + sin_offsets).to(tl.float32)

        # Reshape q to [num_q_heads, head_dim] for per-head RoPE
        num_q_heads: tl.constexpr = Q_BLOCK_SIZE // head_dim
        q_2d = q_normed.reshape(num_q_heads, head_dim)

        # Extract first half_rotary_dim and second half_rotary_dim
        q_x1 = tl.extract_slice(q_2d,
                                offsets=(0, 0),
                                sizes=(num_q_heads, half_rotary_dim),
                                strides=(1, 1))
        q_x2 = tl.extract_slice(q_2d,
                                offsets=(0, half_rotary_dim),
                                sizes=(num_q_heads, half_rotary_dim),
                                strides=(1, 1))

        # Neox-style RoPE: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        q_r1 = q_x1.to(tl.float32) * cos - q_x2.to(tl.float32) * sin
        q_r2 = q_x2.to(tl.float32) * cos + q_x1.to(tl.float32) * sin

        # Reconstruct: insert roped values back
        q_out = tl.zeros((num_q_heads, head_dim), dtype=tl.bfloat16)
        q_out = tl.insert_slice(q_out, q_r1.to(tl.bfloat16),
                                offsets=(0, 0),
                                sizes=(num_q_heads, half_rotary_dim),
                                strides=(1, 1))
        q_out = tl.insert_slice(q_out, q_r2.to(tl.bfloat16),
                                offsets=(0, half_rotary_dim),
                                sizes=(num_q_heads, half_rotary_dim),
                                strides=(1, 1))

        # Copy non-rotated dims (rotary_dim to head_dim) from normalized q
        non_rot_start: tl.constexpr = half_rotary_dim * 2
        if non_rot_start < head_dim:
            non_rot_size: tl.constexpr = head_dim - non_rot_start
            q_pass = tl.extract_slice(q_2d,
                                      offsets=(0, non_rot_start),
                                      sizes=(num_q_heads, non_rot_size),
                                      strides=(1, 1))
            q_out = tl.insert_slice(q_out, q_pass,
                                    offsets=(0, non_rot_start),
                                    sizes=(num_q_heads, non_rot_size),
                                    strides=(1, 1))

        # Store q output
        q_output_offset = row_idx * q_hidden_size
        tl.store(q_ptr + q_output_offset + q_col_indices,
                 q_out.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
                 mask=q_valid_mask)
        # Store q local variance (packed into qk_var[row, 0])
        tl.store(qk_var_ptr + row_idx * 2, q_var)

        # --- K processing: split + cross-head RMSNorm + RoPE ---
        k_col_indices = tl.arange(0, KV_BLOCK_SIZE)
        k_valid_mask = k_col_indices < kv_hidden_size

        k_vals = tl.load(
            input_ptr + input_offset + q_hidden_size + k_col_indices,
            mask=k_valid_mask, other=0.0).to(tl.float32)

        # Cross-head variance
        k_squares_sum = tl.sum(k_vals * k_vals)
        k_var = k_squares_sum / kv_hidden_size
        k_inv_rms = 1.0 / tl.sqrt(k_var + eps)

        # Normalize and apply weight
        k_normed = (k_vals * k_inv_rms * k_weight).to(tl.bfloat16)

        # Apply partial RoPE per head
        num_kv_heads: tl.constexpr = KV_BLOCK_SIZE // head_dim
        k_2d = k_normed.reshape(num_kv_heads, head_dim)

        k_x1 = tl.extract_slice(k_2d,
                                offsets=(0, 0),
                                sizes=(num_kv_heads, half_rotary_dim),
                                strides=(1, 1))
        k_x2 = tl.extract_slice(k_2d,
                                offsets=(0, half_rotary_dim),
                                sizes=(num_kv_heads, half_rotary_dim),
                                strides=(1, 1))

        k_r1 = k_x1.to(tl.float32) * cos - k_x2.to(tl.float32) * sin
        k_r2 = k_x2.to(tl.float32) * cos + k_x1.to(tl.float32) * sin

        k_out = tl.zeros((num_kv_heads, head_dim), dtype=tl.bfloat16)
        k_out = tl.insert_slice(k_out, k_r1.to(tl.bfloat16),
                                offsets=(0, 0),
                                sizes=(num_kv_heads, half_rotary_dim),
                                strides=(1, 1))
        k_out = tl.insert_slice(k_out, k_r2.to(tl.bfloat16),
                                offsets=(0, half_rotary_dim),
                                sizes=(num_kv_heads, half_rotary_dim),
                                strides=(1, 1))

        if non_rot_start < head_dim:
            k_pass = tl.extract_slice(k_2d,
                                      offsets=(0, non_rot_start),
                                      sizes=(num_kv_heads, non_rot_size),
                                      strides=(1, 1))
            k_out = tl.insert_slice(k_out, k_pass,
                                    offsets=(0, non_rot_start),
                                    sizes=(num_kv_heads, non_rot_size),
                                    strides=(1, 1))

        # Store k output
        k_output_offset = row_idx * kv_hidden_size
        tl.store(k_ptr + k_output_offset + k_col_indices,
                 k_out.reshape(KV_BLOCK_SIZE).to(k_ptr.dtype.element_ty),
                 mask=k_valid_mask)
        # Store k local variance (packed into qk_var[row, 1])
        tl.store(qk_var_ptr + row_idx * 2 + 1, k_var)

        # --- V processing: just copy ---
        v_col_indices = tl.arange(0, KV_BLOCK_SIZE)
        v_valid_mask = v_col_indices < kv_hidden_size
        v_vals = tl.load(
            input_ptr + input_offset + q_hidden_size + kv_hidden_size
            + v_col_indices,
            mask=v_valid_mask, other=0.0)
        v_output_offset = row_idx * kv_hidden_size
        tl.store(v_ptr + v_output_offset + v_col_indices,
                 v_vals, mask=v_valid_mask)


def minimax_qkv_crosshead_norm_rope_impl(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = qkv.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2

    Q_BLOCK_SIZE = triton.next_power_of_2(q_hidden_size)
    KV_BLOCK_SIZE = triton.next_power_of_2(kv_hidden_size)

    q_output = torch.empty(batch_size, q_hidden_size,
                           device=qkv.device, dtype=qkv.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size,
                           device=qkv.device, dtype=qkv.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size,
                           device=qkv.device, dtype=qkv.dtype)
    # Packed [batch, 2]: column 0 = q_var, column 1 = k_var
    qk_var_output = torch.empty(batch_size, 2,
                                device=qkv.device, dtype=torch.float32)

    num_vectorcore = get_vectorcore_num()
    n_rows = min(batch_size, num_vectorcore)
    half_rotary_dim = rotary_dim // 2

    minimax_qkv_crosshead_norm_rope_kernel[(n_rows,)](
        qkv,
        cos_sin_cache,
        positions,
        q_output,
        k_output,
        v_output,
        qk_var_output,
        q_weight,
        k_weight,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        head_dim,
        half_rotary_dim,
        rotary_dim,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )
    return q_output, k_output, v_output, qk_var_output


def minimax_qkv_crosshead_norm_rope_impl_fake(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = qkv.shape[0]
    q_output = torch.empty(batch_size, q_hidden_size,
                           device=qkv.device, dtype=qkv.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size,
                           device=qkv.device, dtype=qkv.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size,
                           device=qkv.device, dtype=qkv.dtype)
    qk_var_output = torch.empty(batch_size, 2,
                                device=qkv.device, dtype=torch.float32)
    return q_output, k_output, v_output, qk_var_output


direct_register_custom_op(
    op_name="minimax_qkv_crosshead_norm_rope",
    op_func=minimax_qkv_crosshead_norm_rope_impl,
    fake_impl=minimax_qkv_crosshead_norm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


@triton.jit
def minimax_tp_correction_kernel(
    q_in_ptr, k_in_ptr,
    q_out_ptr, k_out_ptr,
    qk_var_ptr,
    qk_reduced_ptr,
    batch_size,
    tp_world,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    """
    Fused TP correction for MiniMax cross-head QKNorm.

    Replaces 8 small PyTorch ops (RealDiv, Add, Rsqrt, Add, Rsqrt, Cast,
    Mul, Mul) with a single kernel launch.

    qk_var layout: [batch, 2] where col 0 = q_var, col 1 = k_var
    qk_reduced layout: same as qk_var after all_reduce

    Per row:
      global_var = all_reduced_var / tp_world
      correction = sqrt(local_var + eps) / sqrt(global_var + eps)
      q_out = q_in * q_correction
      k_out = k_in * k_correction
    """
    row_pid = tl.program_id(0)
    row_step = tl.num_programs(0)

    inv_tp = 1.0 / tp_world

    for row_idx in tl.range(row_pid, batch_size, row_step):
        # Load local variances from packed [batch, 2]
        q_local_var = tl.load(qk_var_ptr + row_idx * 2)
        k_local_var = tl.load(qk_var_ptr + row_idx * 2 + 1)

        # Load all-reduced sum, divide by tp_world for global mean variance
        q_global_var = tl.load(qk_reduced_ptr + row_idx * 2) * inv_tp
        k_global_var = tl.load(qk_reduced_ptr + row_idx * 2 + 1) * inv_tp

        # correction = rsqrt(global+eps) / rsqrt(local+eps)
        #            = sqrt(local+eps) / sqrt(global+eps)
        q_corr = tl.sqrt(q_local_var + eps) / tl.sqrt(q_global_var + eps)
        k_corr = tl.sqrt(k_local_var + eps) / tl.sqrt(k_global_var + eps)

        # Apply correction to q
        q_col = tl.arange(0, Q_BLOCK_SIZE)
        q_mask = q_col < q_hidden_size
        q_offset = row_idx * q_hidden_size
        q_vals = tl.load(q_in_ptr + q_offset + q_col,
                         mask=q_mask, other=0.0).to(tl.float32)
        tl.store(q_out_ptr + q_offset + q_col,
                 (q_vals * q_corr).to(q_out_ptr.dtype.element_ty),
                 mask=q_mask)

        # Apply correction to k
        k_col = tl.arange(0, KV_BLOCK_SIZE)
        k_mask = k_col < kv_hidden_size
        k_offset = row_idx * kv_hidden_size
        k_vals = tl.load(k_in_ptr + k_offset + k_col,
                         mask=k_mask, other=0.0).to(tl.float32)
        tl.store(k_out_ptr + k_offset + k_col,
                 (k_vals * k_corr).to(k_out_ptr.dtype.element_ty),
                 mask=k_mask)


def minimax_tp_correction_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    qk_var: torch.Tensor,
    qk_reduced: torch.Tensor,
    tp_world: int,
    eps: float,
    q_hidden_size: int,
    kv_hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = q.shape[0]

    Q_BLOCK_SIZE = triton.next_power_of_2(q_hidden_size)
    KV_BLOCK_SIZE = triton.next_power_of_2(kv_hidden_size)

    q_output = torch.empty_like(q)
    k_output = torch.empty_like(k)

    num_vectorcore = get_vectorcore_num()
    n_rows = min(batch_size, num_vectorcore)

    minimax_tp_correction_kernel[(n_rows,)](
        q, k,
        q_output, k_output,
        qk_var,
        qk_reduced,
        batch_size,
        tp_world,
        q_hidden_size, kv_hidden_size,
        eps,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )
    return q_output, k_output


def minimax_tp_correction_impl_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    qk_var: torch.Tensor,
    qk_reduced: torch.Tensor,
    tp_world: int,
    eps: float,
    q_hidden_size: int,
    kv_hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k)


direct_register_custom_op(
    op_name="minimax_tp_correction",
    op_func=minimax_tp_correction_impl,
    fake_impl=minimax_tp_correction_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
