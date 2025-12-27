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
    q_bias: Optional[torch.Tensor],
    k_bias: Optional[torch.Tensor],
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


# currently, only applicable to the qwen3-next model.
@triton.jit
def split_qkv_gated_rmsnorm_rope_kernel(
    input_ptr,
    sin_ptr,
    cos_ptr,
    q_ptr,
    gate_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    k_weight_ptr,
    batch_size,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    q_gate_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PASS_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
):
    pid, step = tl.program_id(0), tl.num_programs(0)

    # base cols & masks
    q_gate_cols = tl.arange(0, HEAD_DIM)
    q_gate_mask = q_gate_cols < HEAD_DIM
    kv_cols = tl.arange(0, kv_hidden_size)
    kv_mask = kv_cols < kv_hidden_size
    rot_cols = tl.arange(0, ROPE_DIM)
    rot_mask = rot_cols < ROPE_DIM
    pass_cols = tl.arange(0, PASS_DIM)
    pass_mask = pass_cols < PASS_DIM

    q_norm_weight = tl.load(q_weight_ptr + q_gate_cols,
                            mask=q_gate_mask,
                            other=0.0).to(tl.float32) + 1.0
    k_norm_weight = tl.load(k_weight_ptr + q_gate_cols,
                            mask=q_gate_mask,
                            other=0.0).to(tl.float32) + 1.0

    for x in tl.range(pid, batch_size, step):
        sc_offset = x * ROPE_DIM
        input_offset = x * total_hidden_size
        output_offset = x * q_gate_hidden_size

        sin = (tl.load(sin_ptr + sc_offset + rot_cols,
                       mask=rot_mask,
                       other=0.0)).to(tl.float32)
        cos = (tl.load(cos_ptr + sc_offset + rot_cols,
                       mask=rot_mask,
                       other=0.0)).to(tl.float32)

        # [HEAD0_q, HEAD0_gate, HEAD1_q, HEAD1_gate, ..., KV_HEAD0_k, KV_HEAD1_k, ..., KV_HEAD0_v, KV_HEAD1_v, ...]
        # q norm & rope + gate copy
        for _ in tl.range(num_heads):
            q = tl.load(input_ptr + input_offset + q_gate_cols,
                        mask=q_gate_mask,
                        other=0.0).to(tl.float32)
            # q norm
            squa = q * q
            var = tl.sum(squa) / HEAD_DIM
            rstd = tl.rsqrt(var + eps)
            norm = q * rstd
            q_norm = (norm * q_norm_weight).reshape(1, HEAD_DIM)
            # q rope
            q_rot = tl.extract_slice(
                q_norm,
                offsets=(0, 0),
                sizes=(1, ROPE_DIM),
                strides=(1, 1),
            )
            q_pass = tl.extract_slice(
                q_norm,
                offsets=(0, ROPE_DIM),
                sizes=(1, PASS_DIM),
                strides=(1, 1),
            )
            x1 = tl.extract_slice(
                q_rot,
                offsets=(0, 0),
                sizes=(1, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            x2 = tl.extract_slice(
                q_rot,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(1, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            cat_x = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
            cat_x = tl.insert_slice(
                cat_x,
                -x2,
                offsets=(0, 0),
                sizes=(1, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            cat_x = tl.insert_slice(
                cat_x,
                x1,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(1, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            q_rot_rope = cat_x * sin + q_rot * cos
            tl.store(q_ptr + output_offset + rot_cols,
                     q_rot_rope.reshape(ROPE_DIM),
                     mask=rot_mask)
            tl.store(q_ptr + output_offset + ROPE_DIM + pass_cols,
                     q_pass.reshape(PASS_DIM),
                     mask=pass_mask)
            input_offset += HEAD_DIM
            # gate copy
            i_gate = tl.load(input_ptr + input_offset + q_gate_cols,
                             mask=q_gate_mask,
                             other=0.0)
            tl.store(gate_ptr + output_offset + q_gate_cols,
                     i_gate,
                     mask=q_gate_mask)

            input_offset += HEAD_DIM
            output_offset += HEAD_DIM

        output_offset = x * kv_hidden_size
        # k norm & rope
        for _ in tl.range(num_kv_heads):
            k = tl.load(input_ptr + input_offset + q_gate_cols,
                        mask=q_gate_mask,
                        other=0.0).to(tl.float32)
            # k norm
            squa = k * k
            var = tl.sum(squa) / HEAD_DIM
            rstd = tl.rsqrt(var + eps)
            norm = k * rstd
            k_norm = (norm * k_norm_weight).reshape(1, HEAD_DIM)
            # k rope
            k_rot = tl.extract_slice(
                k_norm,
                offsets=(0, 0),
                sizes=(1, ROPE_DIM),
                strides=(1, 1),
            )
            k_pass = tl.extract_slice(
                k_norm,
                offsets=(0, ROPE_DIM),
                sizes=(1, PASS_DIM),
                strides=(1, 1),
            )
            x1 = tl.extract_slice(
                k_rot,
                offsets=(0, 0),
                sizes=(1, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            x2 = tl.extract_slice(
                k_rot,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(1, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            cat_x = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
            cat_x = tl.insert_slice(
                cat_x,
                -x2,
                offsets=(0, 0),
                sizes=(1, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            cat_x = tl.insert_slice(
                cat_x,
                x1,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(1, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            k_rot_rope = cat_x * sin + k_rot * cos
            tl.store(k_ptr + output_offset + rot_cols,
                     k_rot_rope.reshape(ROPE_DIM),
                     mask=rot_mask)
            tl.store(k_ptr + output_offset + ROPE_DIM + pass_cols,
                     k_pass.reshape(PASS_DIM),
                     mask=pass_mask)
            input_offset += HEAD_DIM
            output_offset += HEAD_DIM

        output_offset = x * kv_hidden_size
        # v copy
        i_v = tl.load(input_ptr + input_offset + kv_cols,
                      mask=kv_mask,
                      other=0.0)
        tl.store(v_ptr + output_offset + kv_cols, i_v, mask=kv_mask)


def split_qkv_gated_rmsnorm_rope_impl(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_gate_hidden_size: int,
    kv_hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert triton.next_power_of_2(
        head_dim
    ) == head_dim, "head_dim must be a power of 2 for triton kernel"
    bs, total_hidden_size = input.shape
    assert total_hidden_size == 2 * kv_hidden_size + 2 * q_gate_hidden_size
    cos = cos.view(-1, rope_dim)
    sin = sin.view(-1, rope_dim)
    q_output = torch.empty(bs,
                           q_gate_hidden_size,
                           device=input.device,
                           dtype=input.dtype)
    gate_output = torch.empty(bs,
                              q_gate_hidden_size,
                              device=input.device,
                              dtype=input.dtype)
    k_output = torch.empty(bs,
                           kv_hidden_size,
                           device=input.device,
                           dtype=input.dtype)
    v_output = torch.empty(bs,
                           kv_hidden_size,
                           device=input.device,
                           dtype=input.dtype)

    num_vectorcore = get_vectorcore_num()
    grid = (min(num_vectorcore, bs), )
    split_qkv_gated_rmsnorm_rope_kernel[grid](
        input,
        sin,
        cos,
        q_output,
        gate_output,
        k_output,
        v_output,
        q_weight,
        k_weight,
        bs,
        total_hidden_size,
        eps,
        num_heads,
        num_kv_heads,
        q_gate_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        head_dim - rope_dim,
        rope_dim // 2,
    )
    return q_output, gate_output, k_output, v_output


def split_qkv_gated_rmsnorm_rope_impl_fake(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_gate_hidden_size: int,
    kv_hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = input.shape[0]
    q_output = torch.empty(
        batch_size,
        q_gate_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    gate_output = torch.empty(
        batch_size,
        q_gate_hidden_size,
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

    return q_output, gate_output, k_output, v_output


direct_register_custom_op(op_name="qkv_gated_rmsnorm_rope",
                          op_func=split_qkv_gated_rmsnorm_rope_impl,
                          fake_impl=split_qkv_gated_rmsnorm_rope_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")
