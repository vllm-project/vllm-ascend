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
from typing import Optional

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def split_qkv_rmsnorm_rope_kernel(
    input_ptr,
    sin_ptr,
    cos_ptr,
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
    HALF_HEAD_DIM: tl.constexpr,
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
        input_values = (tl.load(input_ptr + input_offset + col_indices,
                                mask=valid_mask,
                                other=0.0).to(tl.float32).reshape(
                                    Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM))
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(
            Q_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = (input_values * reciprocal_std
                             )  # (Q_BLOCK_SIZE//HEAD_DIM, HEAD_DIM)
        if BIAS:
            normalized_values = (normalized_values * weight_values +
                                 bias_values).to(tl.bfloat16)
        else:
            normalized_values = (normalized_values * weight_values).to(
                tl.bfloat16)

        sc_offsets = row_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        x1 = tl.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            normalized_values,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM),
                         dtype=tl.bfloat16)
        cat_x = tl.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        roped_q = cat_x * sin + normalized_values * cos
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
        input_values = (tl.load(input_ptr + input_offset + col_indices,
                                mask=valid_mask,
                                other=0.0).to(tl.float32).reshape(
                                    KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM))
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(
            KV_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = (input_values * reciprocal_std
                             )  # (KV_BLOCK_SIZE/HEAD_DIM, HEAD_DIM)
        if BIAS:
            normalized_values = (normalized_values * weight_values +
                                 bias_values).to(tl.bfloat16)
        else:
            normalized_values = (normalized_values * weight_values).to(
                tl.bfloat16)
        sc_offsets = row_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        x1 = tl.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            normalized_values,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM),
                         dtype=tl.bfloat16)
        cat_x = tl.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        roped_k = cat_x * sin + normalized_values * cos

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
        input_values = tl.load(input_ptr + input_offset + col_indices,
                               mask=valid_mask,
                               other=0.0)
        tl.store(v_ptr + output_offset + col_indices,
                 input_values,
                 mask=valid_mask)
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
    eps: float,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert KV_BLOCK_SIZE == head_dim
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    q_output = torch.empty(batch_size,
                           q_hidden_size,
                           device=input.device,
                           dtype=input.dtype)
    k_output = torch.empty(batch_size,
                           kv_hidden_size,
                           device=input.device,
                           dtype=input.dtype)
    v_output = torch.empty(batch_size,
                           kv_hidden_size,
                           device=input.device,
                           dtype=input.dtype)
    n_cols = kv_hidden_size // KV_BLOCK_SIZE
    num_vectorcore = get_vectorcore_num()
    assert num_vectorcore % n_cols == 0
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
        head_dim // 2,
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
    eps: float,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
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


direct_register_custom_op(op_name="qkv_rmsnorm_rope",
                          op_func=split_qkv_rmsnorm_rope_impl,
                          fake_impl=split_qkv_rmsnorm_rope_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")
