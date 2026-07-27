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

from __future__ import annotations

import torch
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_vectorcore_num, insert_slice


# TODO: UB size differs across chips; consider whether BLOCK_SIZE can
# be dynamically computed with a formula instead of autotuning {1,2,4}.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1}),
        triton.Config({"BLOCK_SIZE": 2}),
        triton.Config({"BLOCK_SIZE": 4}),
    ],
    key=["q_cols", "k_cols"],
)
@triton.jit
def _split_qkv_and_compute_local_qk_var_kernel(
    input_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    qk_var_ptr,
    num_tokens,
    q_cols: tl.constexpr,
    k_cols: tl.constexpr,
    q_cols_pow2: tl.constexpr,
    k_cols_pow2: tl.constexpr,
    qkv_stride: tl.constexpr,
    q_inv_size: tl.constexpr,
    k_inv_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid Stride Loop + batch loading + precomputed reciprocal.
    (BLOCK_SIZE is limited to 1-4 to prevent UB overflow for large hidden_size)
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    block_range = tl.arange(0, BLOCK_SIZE)

    # Grid Stride Loop: each program processes BLOCK_SIZE tokens at a time
    stride = num_pids * BLOCK_SIZE
    start_token_idx = pid * BLOCK_SIZE

    for block_start in tl.range(start_token_idx, num_tokens, stride):
        token_indices = block_start + block_range
        token_mask = (token_indices < num_tokens)[:, None]

        # === Batch load QKV data ===
        # Q: [BLOCK_SIZE, q_cols]
        q_offset = tl.arange(0, q_cols_pow2)[None, :]
        q_mask = token_mask & (q_offset < q_cols)
        q_batch = tl.load(
            input_ptr + token_indices[:, None] * qkv_stride + q_offset,
            mask=q_mask,
            other=0.0,
        )
        q_batch_f32 = q_batch.to(tl.float32)

        # K: [BLOCK_SIZE, k_cols], K follows immediately after Q
        k_offset = tl.arange(0, k_cols_pow2)[None, :]
        k_mask = token_mask & (k_offset < k_cols)
        k_batch = tl.load(
            input_ptr + token_indices[:, None] * qkv_stride + q_cols + k_offset,
            mask=k_mask,
            other=0.0,
        )
        k_batch_f32 = k_batch.to(tl.float32)

        # V: [BLOCK_SIZE, k_cols], V is at offset Q + 2*K
        v_offset = tl.arange(0, k_cols_pow2)[None, :]
        v_mask = token_mask & (v_offset < k_cols)
        v_batch = tl.load(
            input_ptr + token_indices[:, None] * qkv_stride + q_cols + k_cols + v_offset,
            mask=v_mask,
            other=0.0,
        )

        # === Batch compute sum of squares ===
        q_squaresum = tl.sum(q_batch_f32 * q_batch_f32, axis=-1) * q_inv_size
        k_squaresum = tl.sum(k_batch_f32 * k_batch_f32, axis=-1) * k_inv_size

        # === Batch store QKV output ===
        # Store Q
        q_out_offset = token_indices[:, None] * q_cols + q_offset
        q_out_mask = token_mask & (q_offset < q_cols)
        tl.store(q_out_ptr + q_out_offset, q_batch, mask=q_out_mask)

        # Store K
        k_out_offset = token_indices[:, None] * k_cols + k_offset
        k_out_mask = token_mask & (k_offset < k_cols)
        tl.store(k_out_ptr + k_out_offset, k_batch, mask=k_out_mask)

        # Store V
        v_out_offset = token_indices[:, None] * k_cols + v_offset
        v_out_mask = token_mask & (v_offset < k_cols)
        tl.store(v_out_ptr + v_out_offset, v_batch, mask=v_out_mask)

        # === Store variance ===
        var_offset = token_indices * 2
        var_mask = token_indices < num_tokens
        tl.store(qk_var_ptr + var_offset, q_squaresum, mask=var_mask)
        tl.store(qk_var_ptr + var_offset + 1, k_squaresum, mask=var_mask)


@triton.jit
def _apply_global_rmsnorm_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    cs_row_stride,
    q_weight_ptr,
    k_weight_ptr,
    qk_global_var_ptr,
    eps: tl.constexpr,
    inv_tp_world: tl.constexpr,
    num_tokens,
    q_cols: tl.constexpr,
    k_cols: tl.constexpr,
    q_num_heads: tl.constexpr,
    k_num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    HALF: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0)
    tokens_per_program = tl.cdiv(num_tokens, num_programs)
    iter_num_per_program = tokens_per_program
    program_token_offset = pid * tokens_per_program
    program_token_end = min(program_token_offset + tokens_per_program, num_tokens)

    token_tile_offsets = tl.arange(0, 1)
    q_head_offsets = tl.arange(0, q_num_heads)[:, None]
    k_head_offsets = tl.arange(0, k_num_heads)[:, None]
    hd_offsets = tl.arange(0, head_dim)[None, :]

    q_row_offsets = q_head_offsets * head_dim + hd_offsets
    k_row_offsets = k_head_offsets * head_dim + hd_offsets

    q_weight = tl.load(q_weight_ptr + q_row_offsets).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + k_row_offsets).to(tl.float32)

    half_offsets = tl.arange(0, HALF)
    base_token_offsets = program_token_offset + token_tile_offsets

    for iter in tl.range(iter_num_per_program):
        token_offsets = base_token_offsets + iter
        token_mask = token_offsets < program_token_end

        q_gv = tl.load(qk_global_var_ptr + token_offsets * 2, mask=token_mask, other=0.0).to(tl.float32)
        q_gv = q_gv * inv_tp_world
        k_gv = tl.load(qk_global_var_ptr + token_offsets * 2 + 1, mask=token_mask, other=0.0).to(tl.float32)
        k_gv = k_gv * inv_tp_world
        q_scale = 1.0 / tl.sqrt(q_gv + eps)
        k_scale = 1.0 / tl.sqrt(k_gv + eps)

        q_offsets = token_offsets[:, None, None] * q_cols + q_row_offsets[None, :, :]
        q_mask = token_mask[:, None, None]
        q_vals_raw = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
        q_vals = q_vals_raw.to(tl.float32) * q_scale[:, None, None] * q_weight[None, :, :]

        k_offsets = token_offsets[:, None, None] * k_cols + k_row_offsets[None, :, :]
        k_mask = token_mask[:, None, None]
        k_vals_raw = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
        k_vals = k_vals_raw.to(tl.float32) * k_scale[:, None, None] * k_weight[None, :, :]

        # Neox-style RoPE on the first rotary_dim dimensions of each head
        cs_offsets = token_offsets[:, None] * cs_row_stride + half_offsets[None, :]
        cs_mask = token_mask[:, None]
        cos_row = tl.load(cos_ptr + cs_offsets, mask=cs_mask, other=0.0).to(tl.float32)
        sin_row = tl.load(sin_ptr + cs_offsets, mask=cs_mask, other=0.0).to(tl.float32)

        q1 = extract_slice(
            q_vals,
            offsets=(0, 0, 0),
            sizes=(1, q_num_heads, HALF),
            strides=(1, 1, 1),
        )
        q2 = extract_slice(
            q_vals,
            offsets=(0, 0, HALF),
            sizes=(1, q_num_heads, HALF),
            strides=(1, 1, 1),
        )
        q_vals = insert_slice(
            q_vals,
            q1 * cos_row[:, None, :] - q2 * sin_row[:, None, :],
            offsets=(0, 0, 0),
            sizes=(1, q_num_heads, HALF),
            strides=(1, 1, 1),
        )
        q_vals = insert_slice(
            q_vals,
            q2 * cos_row[:, None, :] + q1 * sin_row[:, None, :],
            offsets=(0, 0, HALF),
            sizes=(1, q_num_heads, HALF),
            strides=(1, 1, 1),
        )
        tl.store(q_ptr + q_offsets, q_vals.to(q_vals_raw.dtype), mask=q_mask)

        k1 = extract_slice(
            k_vals,
            offsets=(0, 0, 0),
            sizes=(1, k_num_heads, HALF),
            strides=(1, 1, 1),
        )
        k2 = extract_slice(
            k_vals,
            offsets=(0, 0, HALF),
            sizes=(1, k_num_heads, HALF),
            strides=(1, 1, 1),
        )
        k_vals = insert_slice(
            k_vals,
            k1 * cos_row[:, None, :] - k2 * sin_row[:, None, :],
            offsets=(0, 0, 0),
            sizes=(1, k_num_heads, HALF),
            strides=(1, 1, 1),
        )
        k_vals = insert_slice(
            k_vals,
            k2 * cos_row[:, None, :] + k1 * sin_row[:, None, :],
            offsets=(0, 0, HALF),
            sizes=(1, k_num_heads, HALF),
            strides=(1, 1, 1),
        )
        tl.store(k_ptr + k_offsets, k_vals.to(k_vals_raw.dtype), mask=k_mask)


def split_qkv_tp_rmsnorm_rope_impl(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    tp_world: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = input.shape[0]
    input_2d = input.view(num_tokens, -1)
    q = torch.empty(num_tokens, q_hidden_size, device=input.device, dtype=input.dtype)
    k = torch.empty(num_tokens, kv_hidden_size, device=input.device, dtype=input.dtype)
    v = torch.empty(num_tokens, kv_hidden_size, device=input.device, dtype=input.dtype)
    if num_tokens == 0:
        return q, k, v

    num_vectorcore = get_vectorcore_num()
    grid = (min(num_tokens, num_vectorcore),)
    q_cols = q_hidden_size
    k_cols = kv_hidden_size
    q_num_heads = q_hidden_size // head_dim
    k_num_heads = kv_hidden_size // head_dim

    qk_var = torch.empty(num_tokens, 2, dtype=torch.float32, device=q.device)
    # Precompute reciprocal to avoid division inside kernel
    q_inv_size = 1.0 / q_cols
    k_inv_size = 1.0 / k_cols
    # Pad to power-of-2 for tl.arange (required by Ascend NPU Triton backend)
    q_cols_pow2 = 1 << (q_cols - 1).bit_length()
    k_cols_pow2 = 1 << (k_cols - 1).bit_length()
    _split_qkv_and_compute_local_qk_var_kernel[grid](
        input_2d,
        q,
        k,
        v,
        qk_var,
        num_tokens,
        q_cols,
        k_cols,
        q_cols_pow2,
        k_cols_pow2,
        q_cols + 2 * k_cols,
        q_inv_size,
        k_inv_size,
    )
    if tp_world > 1:
        qk_var = tensor_model_parallel_all_reduce(qk_var)

    cos_2d = cos.view(num_tokens, -1)
    sin_2d = sin.view(num_tokens, -1)
    q_2d = q.view(num_tokens, -1)
    k_2d = k.view(num_tokens, -1)
    _apply_global_rmsnorm_kernel[grid](
        q_2d,
        k_2d,
        cos_2d,
        sin_2d,
        cos_2d.stride(0),
        q_weight,
        k_weight,
        qk_var,
        eps,
        1.0 / tp_world,
        num_tokens,
        q_cols,
        k_cols,
        q_num_heads,
        k_num_heads,
        head_dim,
        rotary_dim,
        rotary_dim // 2,
    )

    return q, k, v


def split_qkv_tp_rmsnorm_rope_impl_fake(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    tp_world: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = input.shape[0]
    q_out = torch.empty(
        num_tokens,
        q_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    k_out = torch.empty(
        num_tokens,
        kv_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    v_out = torch.empty(
        num_tokens,
        kv_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    return q_out, k_out, v_out


direct_register_custom_op(
    op_name="split_qkv_tp_rmsnorm_rope",
    op_func=split_qkv_tp_rmsnorm_rope_impl,
    fake_impl=split_qkv_tp_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
