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
def split_qkv_rmsnorm_rope_kernel(
    input_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
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
    IS_PARTIAL_ROPE: tl.constexpr,
    positions_gm_ptr,
    cos_sin_cache_gm_ptr,
    ele_sin_cos_per_batch: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)
    # q
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
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(Q_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = input_values * reciprocal_std  # (Q_BLOCK_SIZE//HEAD_DIM, HEAD_DIM)
        if BIAS:
            normalized_values = (normalized_values * weight_values + bias_values).to(tl.bfloat16)
        else:
            normalized_values = (normalized_values * weight_values).to(tl.bfloat16)

        pos_values = tl.load(positions_gm_ptr + row_idx)
        sin_cos_indices = (pos_values * ele_sin_cos_per_batch + tl.arange(0, ele_sin_cos_per_batch)).reshape(
            2, ROPE_DIM
        )
        input_values = tl.load(cos_sin_cache_gm_ptr + sin_cos_indices)
        cos = tl.extract_slice(
            input_values,
            offsets=(0, 0),
            sizes=(1, ROPE_DIM),
            strides=(1, 1),
        )
        sin = tl.extract_slice(
            input_values,
            offsets=(1, 0),
            sizes=(1, ROPE_DIM),
            strides=(1, 1),
        )

        x1 = tl.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            normalized_values,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.bfloat16)
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
        roped_q = cat_x * sin
        if IS_PARTIAL_ROPE:
            cat_x = tl.extract_slice(
                normalized_values,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            roped_q += cat_x * cos
            normalized_values = tl.insert_slice(
                normalized_values,
                roped_q,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            tl.store(
                q_ptr + output_offset + col_indices,
                normalized_values.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
                mask=valid_mask,
            )
        else:
            roped_q += normalized_values * cos
            tl.store(
                q_ptr + output_offset + col_indices,
                roped_q.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
                mask=valid_mask,
            )
        input_offset += input_offset_step
        output_offset += output_offset_step

    weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))
    input_offset = row_pid * total_hidden_size + q_hidden_size
    output_offset = row_pid * kv_hidden_size
    output_offset_step = row_step * kv_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(KV_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = input_values * reciprocal_std  # (KV_BLOCK_SIZE/HEAD_DIM, HEAD_DIM)
        if BIAS:
            normalized_values = (normalized_values * weight_values + bias_values).to(tl.bfloat16)
        else:
            normalized_values = (normalized_values * weight_values).to(tl.bfloat16)

        pos_values = tl.load(positions_gm_ptr + row_idx)
        sin_cos_indices = (pos_values * ele_sin_cos_per_batch + tl.arange(0, ele_sin_cos_per_batch)).reshape(
            2, ROPE_DIM
        )

        input_values = tl.load(cos_sin_cache_gm_ptr + sin_cos_indices)
        cos = tl.extract_slice(
            input_values,
            offsets=(0, 0),
            sizes=(1, ROPE_DIM),
            strides=(1, 1),
        )
        sin = tl.extract_slice(
            input_values,
            offsets=(1, 0),
            sizes=(1, ROPE_DIM),
            strides=(1, 1),
        )

        x1 = tl.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            normalized_values,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.bfloat16)
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
        roped_k = cat_x * sin
        if IS_PARTIAL_ROPE:
            cat_x = tl.extract_slice(
                normalized_values,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            roped_k += cat_x * cos
            normalized_values = tl.insert_slice(
                normalized_values,
                roped_k,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            tl.store(
                k_ptr + output_offset + col_indices,
                normalized_values.reshape(KV_BLOCK_SIZE).to(k_ptr.dtype.element_ty),
                mask=valid_mask,
            )
        else:
            roped_k += normalized_values * cos
            tl.store(
                k_ptr + output_offset + col_indices,
                roped_k.to(tl.bfloat16).reshape(KV_BLOCK_SIZE),
                mask=valid_mask,
            )
        input_offset += input_offset_step
        output_offset += output_offset_step

    input_offset = row_pid * total_hidden_size + q_hidden_size + kv_hidden_size
    output_offset = row_pid * kv_hidden_size
    for _ in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
        tl.store(v_ptr + output_offset + col_indices, input_values, mask=valid_mask)
        input_offset += input_offset_step
        output_offset += output_offset_step


@triton.jit
def split_qkv_rmsnorm_rope_prefill_kernel(
    input_gm_ptr,
    q_gm_ptr,
    k_gm_ptr,
    v_gm_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size: tl.constexpr,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    batch_size_per_vec: tl.constexpr,
    iter_num_per_vec: tl.constexpr,
    batch_size_per_iter_per_vec: tl.constexpr,
    qk_head_nums_per_iter_per_vec: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    qk_head_num_sum: tl.constexpr,
    v_batch_size_per_iter_per_vec: tl.constexpr,
    v_iter_num_per_vec: tl.constexpr,
    positions_gm_ptr,
    cos_sin_cache_gm_ptr,
    ele_sin_cos_per_batch: tl.constexpr,
):
    row_pid = tl.program_id(0)

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

    input_batch_offset = row_pid * batch_size_per_vec
    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(0, q_hidden_size + kv_hidden_size)
    nmask = nblk_idx < total_hidden_size

    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    pos_indices = input_batch_offset + tl.arange(0, batch_size_per_iter_per_vec)
    output_q_nblk_idx = tl.arange(0, q_hidden_size)
    output_q_nmask = output_q_nblk_idx < q_hidden_size
    output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
    output_kv_nmask = output_kv_nblk_idx < kv_hidden_size
    sin_cos_range = tl.arange(0, ele_sin_cos_per_batch)
    cos_sin_cache_offset = cos_sin_cache_gm_ptr + sin_cos_range

    for iter in tl.range(iter_num_per_vec):
        pos_offset = iter * batch_size_per_iter_per_vec
        x = tl.load(
            positions_gm_ptr + pos_indices + pos_offset, mask=(pos_indices + pos_offset) < input_batch_offset_end
        )
        mmask = (mblk_idx + pos_offset) < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = (mblk_idx + pos_offset)[:, None] * total_hidden_size + nblk_idx[None, :]
        values_tmp1 = tl.load(input_gm_ptr + idx, mask=mask).reshape(qk_head_nums_per_iter_per_vec, HEAD_DIM)
        if BIAS:
            q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
            k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

        values_tmp3 = tl.zeros((batch_size_per_iter_per_vec, ele_sin_cos_per_batch), dtype=tl.bfloat16)
        for i in tl.range(batch_size_per_iter_per_vec):
            pos = tl.get_element(x, (i,))
            values_tmp3 = tl.insert_slice(
                values_tmp3.reshape(batch_size_per_iter_per_vec, ele_sin_cos_per_batch),
                tl.load(pos * ele_sin_cos_per_batch + cos_sin_cache_offset[:, None]).reshape(1, ele_sin_cos_per_batch),
                offsets=(i, 0),
                sizes=(1, ele_sin_cos_per_batch),
                strides=(1, 1),
            )
        values_tmp3 = values_tmp3.reshape(batch_size_per_iter_per_vec, 2, ROPE_DIM)
        cos = tl.extract_slice(
            values_tmp3,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, 1, ROPE_DIM),
            strides=(1, 1, 1),
        )
        sin = tl.extract_slice(
            values_tmp3,
            offsets=(0, 1, 0),
            sizes=(batch_size_per_iter_per_vec, 1, ROPE_DIM),
            strides=(1, 1, 1),
        )

        normalized_values = values_tmp1.to(tl.float32)
        normalized_values = normalized_values * normalized_values
        normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
        normalized_values = values_tmp1 / tl.sqrt(normalized_values + eps).reshape(qk_head_nums_per_iter_per_vec, 1)

        normalized_values_tmp = tl.extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )

        if BIAS:
            normalized_values_tmp = (normalized_values_tmp * q_weight_values + q_bias_values).to(tl.bfloat16)
        else:
            normalized_values_tmp = (normalized_values_tmp * q_weight_values).to(tl.bfloat16)

        # q rope
        values_tmp = tl.zeros((batch_size_per_iter_per_vec, q_head_num, ROPE_DIM), dtype=tl.bfloat16)
        y = tl.extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        values_tmp = tl.insert_slice(
            values_tmp,
            y,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        y = tl.extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = tl.insert_slice(
            values_tmp,
            -y,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = values_tmp * sin
        q_output_idx = output_q_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * q_hidden_size
        mask = (mmask[:, None]) & (output_q_nmask[None, :])
        if IS_PARTIAL_ROPE:
            y = tl.extract_slice(
                normalized_values_tmp,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            values_tmp += y * cos
            normalized_values_tmp = tl.insert_slice(
                normalized_values_tmp,
                values_tmp,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                q_gm_ptr + q_output_idx,
                normalized_values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=mask,
            )
        else:
            values_tmp += normalized_values_tmp * cos
            tl.store(
                q_gm_ptr + q_output_idx,
                values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=mask,
            )

        # k rope
        normalized_values_tmp1 = tl.extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, q_head_num, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )

        if BIAS:
            normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values + k_bias_values).to(tl.bfloat16)
        else:
            normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values).to(tl.bfloat16)

        values_tmp2 = tl.zeros((batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM), dtype=tl.bfloat16)

        y = tl.extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        values_tmp2 = tl.insert_slice(
            values_tmp2,
            y,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        y = tl.extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = tl.insert_slice(
            values_tmp2,
            -y,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = values_tmp2 * sin
        kv_output_idx = output_kv_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * kv_hidden_size
        mask = (mmask[:, None]) & (output_kv_nmask[None, :])
        if IS_PARTIAL_ROPE:
            y = tl.extract_slice(
                normalized_values_tmp1,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            values_tmp2 += y * cos
            normalized_values_tmp1 = tl.insert_slice(
                normalized_values_tmp1,
                values_tmp2,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                k_gm_ptr + kv_output_idx,
                normalized_values_tmp1.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=mask,
            )
        else:
            values_tmp2 += normalized_values_tmp1 * cos
            tl.store(
                k_gm_ptr + kv_output_idx,
                values_tmp2.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=mask,
            )

    mblk_idx = tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(q_hidden_size + kv_hidden_size, total_hidden_size)
    nmask = nblk_idx < total_hidden_size
    out_nblk_idx = tl.arange(0, kv_hidden_size)
    out_nmask = out_nblk_idx < kv_hidden_size

    for _ in tl.range(v_iter_num_per_vec):
        mmask = mblk_idx < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = mblk_idx[:, None] * total_hidden_size + nblk_idx[None, :]
        values = tl.load(input_gm_ptr + idx, mask=mask)
        out_idx = mblk_idx[:, None] * kv_hidden_size + out_nblk_idx[None, :]
        out_mask = (mmask[:, None]) & (out_nmask[None, :])
        tl.store(v_gm_ptr + out_idx, values, mask=out_mask)
        mblk_idx += v_batch_size_per_iter_per_vec


def split_qkv_rmsnorm_rope_impl(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # get available vector core
    num_vectorcore = get_vectorcore_num()
    if rope_dim is None:
        rope_dim = head_dim

    batch_size = input.shape[0]
    BIAS = q_bias is not None
    IS_PARTIAL_ROPE = rope_dim != head_dim
    # Q + K + V
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    ele_sin_cos_per_batch = rope_dim * 2

    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    # prefill or decode
    if batch_size < num_vectorcore:  # decode
        KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
        assert head_dim == KV_BLOCK_SIZE
        assert q_hidden_size % kv_hidden_size == 0
        Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
        n_cols = kv_hidden_size // KV_BLOCK_SIZE
        n_rows = num_vectorcore // n_cols

        grid = (n_rows, n_cols, 1)

        split_qkv_rmsnorm_rope_kernel[grid](
            input,
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
            rope_dim,
            rope_dim // 2,
            IS_PARTIAL_ROPE,
            positions,
            cos_sin_cache,
            ele_sin_cos_per_batch,
        )

    else:  # prefill
        KV_BLOCK_SIZE = triton.next_power_of_2(head_dim) * batch_size
        Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * KV_BLOCK_SIZE
        q_head_num = q_hidden_size // head_dim
        kv_head_num = kv_hidden_size // head_dim
        batch_size_per_vec = triton.cdiv(batch_size, num_vectorcore)

        # set number of line loading from GM data is x
        # x*(q_head_num + kv_head_num)*HEAD_DIM: values_tmp
        # 2x*(q_head_num + kv_head_num)*HEAD_DIM: normalized_values(float32)
        # x*ROPE_DIM*2 : cos/sin
        # x*q_head_num*HEAD_DIM*2： normalized_values_tmp
        # x*q_head_num*ROPE_DIM*(0.5) (not IS_PARTIAL_ROPE) x*q_head_num*ROPE_DIM*(0.5): y
        UB_SIZE = 85 * 1024
        # the factor is the sum of elements number
        if IS_PARTIAL_ROPE:
            factor = 5 * q_head_num * head_dim + 3 * kv_head_num * head_dim + rope_dim * 4 + q_head_num * rope_dim
            batch_size_per_iter_per_vec = int(UB_SIZE / input.element_size()) // factor
        else:
            factor = 5 * q_head_num * head_dim + 3 * kv_head_num * head_dim + rope_dim * 2 + q_head_num * rope_dim // 2
            batch_size_per_iter_per_vec = int(UB_SIZE / input.element_size()) // factor
        batch_size_per_iter_per_vec = min(batch_size_per_iter_per_vec, batch_size_per_vec)
        qk_head_num_sum = int(q_head_num + kv_head_num)
        qk_head_nums_per_iter_per_vec = batch_size_per_iter_per_vec * qk_head_num_sum

        iter_num_per_vec = triton.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)

        grid = (min(num_vectorcore, batch_size), 1, 1)

        # v tiling
        v_batch_size_per_iter_per_vec = UB_SIZE / torch.bfloat16.itemsize // (kv_hidden_size + 1)
        v_batch_size_per_iter_per_vec = min(v_batch_size_per_iter_per_vec, batch_size_per_vec)
        v_iter_num_per_vec = triton.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)

        split_qkv_rmsnorm_rope_prefill_kernel[grid](
            input,
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
            BIAS,
            head_dim,
            rope_dim,
            rope_dim // 2,
            IS_PARTIAL_ROPE,
            int(batch_size_per_vec),
            int(iter_num_per_vec),
            int(batch_size_per_iter_per_vec),
            int(qk_head_nums_per_iter_per_vec),
            q_head_num,
            kv_head_num,
            qk_head_num_sum,
            int(v_batch_size_per_iter_per_vec),
            int(v_iter_num_per_vec),
            positions,
            cos_sin_cache,
            int(ele_sin_cos_per_batch),
        )
    return q_output, k_output, v_output


def split_qkv_rmsnorm_rope_impl_fake(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Fake implementation for shape inference during Dynamo/AOT tracing.
    # Note: sin and cos are not used in shape computation, but must be present in signature.
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
