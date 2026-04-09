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

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


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
    PAD_Q: tl.constexpr,
    PAD_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0)
    input_row_stride = q_cols + 2 * k_cols

    for idx in tl.range(pid, num_tokens, num_programs):
        input_base = input_ptr + idx * input_row_stride

        q_in_base = input_base
        q_out_base = q_out_ptr + idx * q_cols
        q_sum = tl.zeros((), dtype=tl.float32)
        q_comp = tl.zeros((), dtype=tl.float32)
        for q_off in tl.static_range(0, PAD_Q, BLOCK):
            q_offsets = q_off + tl.arange(0, BLOCK)
            q_mask = q_offsets < q_cols
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
        for k_off in tl.static_range(0, PAD_K, BLOCK):
            k_offsets = k_off + tl.arange(0, BLOCK)
            k_mask = k_offsets < k_cols
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
        for v_off in tl.static_range(0, PAD_K, BLOCK):
            v_offsets = v_off + tl.arange(0, BLOCK)
            v_mask = v_offsets < k_cols
            v_vals = tl.load(v_in_base + v_offsets, mask=v_mask, other=0.0)
            tl.store(v_out_base + v_offsets, v_vals, mask=v_mask)

        tl.store(qk_var_ptr + idx * 2, q_var)
        tl.store(qk_var_ptr + idx * 2 + 1, k_var)


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
    q_num_heads,
    k_num_heads,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    PAD_Q: tl.constexpr,
    PAD_K: tl.constexpr,
    PAD_QH: tl.constexpr,
    PAD_KH: tl.constexpr,
    PAD_HALF: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0)

    for idx in tl.range(pid, num_tokens, num_programs):
        q_gv = tl.load(qk_global_var_ptr + idx * 2).to(tl.float32) * inv_tp_world
        k_gv = tl.load(qk_global_var_ptr + idx * 2 + 1).to(tl.float32) * inv_tp_world
        q_scale = 1.0 / tl.sqrt(q_gv + eps)
        k_scale = 1.0 / tl.sqrt(k_gv + eps)

        q_base = q_ptr + idx * q_cols
        q_offsets = tl.arange(0, PAD_Q)
        q_mask = q_offsets < q_cols
        q_vals = tl.load(q_base + q_offsets, mask=q_mask, other=0.0)
        q_weight = tl.load(q_weight_ptr + q_offsets, mask=q_mask, other=1.0).to(tl.float32)
        q_vals = (q_vals.to(tl.float32) * q_scale * q_weight).to(q_vals.dtype)
        tl.store(q_base + q_offsets, q_vals, mask=q_mask)

        k_base = k_ptr + idx * k_cols
        k_offsets = tl.arange(0, PAD_K)
        k_mask = k_offsets < k_cols
        k_vals = tl.load(k_base + k_offsets, mask=k_mask, other=0.0)
        k_weight = tl.load(k_weight_ptr + k_offsets, mask=k_mask, other=1.0).to(tl.float32)
        k_vals = (k_vals.to(tl.float32) * k_scale * k_weight).to(k_vals.dtype)
        tl.store(k_base + k_offsets, k_vals, mask=k_mask)

        # Neox-style RoPE on the first rotary_dim dimensions of each head
        half = rotary_dim // 2
        half_offsets = tl.arange(0, PAD_HALF)
        half_mask = half_offsets < half
        cos_row = tl.load(
            cos_ptr + idx * cs_row_stride + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        sin_row = tl.load(
            sin_ptr + idx * cs_row_stride + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)

        qh_offsets = tl.arange(0, PAD_QH)[:, None] * head_dim + half_offsets[None, :]
        qh_mask = (tl.arange(0, PAD_QH)[:, None] < q_num_heads) & half_mask[None, :]
        qh_offsets_2 = qh_offsets + half
        q1_raw = tl.load(q_base + qh_offsets, mask=qh_mask, other=0.0)
        q2_raw = tl.load(q_base + qh_offsets_2, mask=qh_mask, other=0.0)
        q1 = q1_raw.to(tl.float32)
        q2 = q2_raw.to(tl.float32)
        qn1 = q1 * cos_row[None, :] - q2 * sin_row[None, :]
        qn2 = q2 * cos_row[None, :] + q1 * sin_row[None, :]
        tl.store(q_base + qh_offsets, qn1.to(q1_raw.dtype), mask=qh_mask)
        tl.store(q_base + qh_offsets_2, qn2.to(q2_raw.dtype), mask=qh_mask)

        kh_offsets = tl.arange(0, PAD_KH)[:, None] * head_dim + half_offsets[None, :]
        kh_mask = (tl.arange(0, PAD_KH)[:, None] < k_num_heads) & half_mask[None, :]
        kh_offsets_2 = kh_offsets + half
        k1_raw = tl.load(k_base + kh_offsets, mask=kh_mask, other=0.0)
        k2_raw = tl.load(k_base + kh_offsets_2, mask=kh_mask, other=0.0)
        k1 = k1_raw.to(tl.float32)
        k2 = k2_raw.to(tl.float32)
        kn1 = k1 * cos_row[None, :] - k2 * sin_row[None, :]
        kn2 = k2 * cos_row[None, :] + k1 * sin_row[None, :]
        tl.store(k_base + kh_offsets, kn1.to(k1_raw.dtype), mask=kh_mask)
        tl.store(k_base + kh_offsets_2, kn2.to(k2_raw.dtype), mask=kh_mask)


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
    _split_qkv_and_compute_local_qk_var_kernel[grid](
        input_2d,
        q,
        k,
        v,
        qk_var,
        num_tokens,
        q_cols,
        k_cols,
        triton.next_power_of_2(q_cols),
        triton.next_power_of_2(k_cols),
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
        triton.next_power_of_2(q_cols),
        triton.next_power_of_2(k_cols),
        triton.next_power_of_2(q_num_heads),
        triton.next_power_of_2(k_num_heads),
        triton.next_power_of_2(rotary_dim // 2),
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
