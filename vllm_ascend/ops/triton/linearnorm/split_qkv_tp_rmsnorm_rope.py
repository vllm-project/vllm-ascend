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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}),
        triton.Config({"BLOCK": 128}),
        triton.Config({"BLOCK": 256}),
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
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0)
    tokens_per_program = tl.cdiv(num_tokens, num_programs)
    iter_num_per_program = tokens_per_program
    program_token_offset = pid * tokens_per_program
    program_token_end = min(program_token_offset + tokens_per_program, num_tokens)
    input_row_stride = q_cols + 2 * k_cols

    for iter in tl.range(iter_num_per_program):
        idx = program_token_offset + iter
        token_mask = idx < program_token_end
        input_base = input_ptr + idx * input_row_stride

        q_in_base = input_base
        q_out_base = q_out_ptr + idx * q_cols
        q_sum = tl.zeros((), dtype=tl.float32)
        q_comp = tl.zeros((), dtype=tl.float32)
        for q_off in tl.static_range(0, q_cols, BLOCK):
            q_offsets = q_off + tl.arange(0, BLOCK)
            q_mask = token_mask & (q_offsets < q_cols)
            q_vals = tl.load(q_in_base + q_offsets, mask=q_mask, other=0.0)
            q_vals_f32 = q_vals.to(tl.float32)
            q_chunk = tl.sum(q_vals_f32 * q_vals_f32, axis=0)
            y = q_chunk - q_comp
            t = q_sum + y
            q_comp = (t - q_sum) - y
            q_sum = t
            tl.store(q_out_base + q_offsets, q_vals, mask=q_mask)
        q_var = q_sum / q_cols

        k_in_base = input_base + q_cols
        k_out_base = k_out_ptr + idx * k_cols
        k_sum = tl.zeros((), dtype=tl.float32)
        k_comp = tl.zeros((), dtype=tl.float32)
        for k_off in tl.static_range(0, k_cols, BLOCK):
            k_offsets = k_off + tl.arange(0, BLOCK)
            k_mask = token_mask & (k_offsets < k_cols)
            k_vals = tl.load(k_in_base + k_offsets, mask=k_mask, other=0.0)
            k_vals_f32 = k_vals.to(tl.float32)
            k_chunk = tl.sum(k_vals_f32 * k_vals_f32, axis=0)
            y = k_chunk - k_comp
            t = k_sum + y
            k_comp = (t - k_sum) - y
            k_sum = t
            tl.store(k_out_base + k_offsets, k_vals, mask=k_mask)
        k_var = k_sum / k_cols

        v_in_base = input_base + q_cols + k_cols
        v_out_base = v_out_ptr + idx * k_cols
        for v_off in tl.static_range(0, k_cols, BLOCK):
            v_offsets = v_off + tl.arange(0, BLOCK)
            v_mask = token_mask & (v_offsets < k_cols)
            v_vals = tl.load(v_in_base + v_offsets, mask=v_mask, other=0.0)
            tl.store(v_out_base + v_offsets, v_vals, mask=v_mask)

        tl.store(qk_var_ptr + idx * 2, q_var, mask=token_mask)
        tl.store(qk_var_ptr + idx * 2 + 1, k_var, mask=token_mask)


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

    cos_2d = cos.view(num_tokens, -1)
    sin_2d = sin.view(num_tokens, -1)
    q_2d = q.view(num_tokens, -1)
    k_2d = k.view(num_tokens, -1)
    qk_var = torch.empty(num_tokens, 2, dtype=torch.float32, device=q.device)
    _split_qkv_and_compute_local_qk_var_kernel[grid](
        input_2d,
        q,
        k,
        v,
        qk_var,
        num_tokens,
        q_cols,
        k_cols,
    )
    if tp_world > 1:
        qk_var = tensor_model_parallel_all_reduce(qk_var)

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
