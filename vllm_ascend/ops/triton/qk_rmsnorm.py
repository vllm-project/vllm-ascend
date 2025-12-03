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
import triton
import triton.language as tl

MAX_CORES = 65535


@triton.jit
def qk_rmsnorm_triton_kernel(
    input_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    k_weight_ptr,
    total_hidden_size,
    eps,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    x = tl.program_id(0)

    q_norm_weight = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(
        tl.float32)
    k_norm_weight = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(
        tl.float32)

    input_offset = x * total_hidden_size
    output_offset = x * q_hidden_size
    q_cols = tl.arange(0, q_hidden_size)
    q_mask = q_cols < q_hidden_size
    kv_cols = tl.arange(0, kv_hidden_size)
    kv_mask = kv_cols < kv_hidden_size

    # q norm
    i_q = tl.load(input_ptr + input_offset + q_cols, mask=q_mask,
                  other=0.0).reshape(num_heads, HEAD_DIM).to(tl.float32)
    squa = i_q * i_q
    var = tl.sum(squa, axis=1) / HEAD_DIM
    rstd = tl.rsqrt(var + eps)
    norm = i_q * rstd[:, None]
    o_q = norm * (q_norm_weight + 1.0)
    tl.store(q_ptr + output_offset + q_cols,
             o_q.reshape(q_hidden_size).to(tl.bfloat16),
             mask=q_mask)

    input_offset += q_hidden_size
    output_offset = x * kv_hidden_size

    # k norm
    i_k = tl.load(input_ptr + input_offset + kv_cols, mask=kv_mask,
                  other=0.0).reshape(num_kv_heads, HEAD_DIM).to(tl.float32)
    squa = i_k * i_k
    var = tl.sum(squa, axis=1) / HEAD_DIM
    rstd = tl.rsqrt(var + eps)
    norm = i_k * rstd[:, None]
    o_k = norm * (k_norm_weight + 1.0)
    tl.store(k_ptr + output_offset + kv_cols,
             o_k.reshape(kv_hidden_size).to(tl.bfloat16),
             mask=kv_mask)

    input_offset += kv_hidden_size

    # v copy
    i_v = tl.load(input_ptr + input_offset + kv_cols, mask=kv_mask, other=0.0)
    tl.store(v_ptr + output_offset + kv_cols, i_v, mask=kv_mask)


@triton.jit
def qk_rmsnorm_gate_triton_kernel(
    input_ptr,
    q_ptr,
    gate_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    k_weight_ptr,
    total_hidden_size,
    eps,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    q_gate_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    x = tl.program_id(0)

    q_norm_weight = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(
        tl.float32)
    k_norm_weight = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(
        tl.float32)

    input_offset = x * total_hidden_size
    output_offset = x * q_gate_hidden_size
    q_gate_cols = tl.arange(0, HEAD_DIM)
    q_gate_mask = q_gate_cols < HEAD_DIM
    kv_cols = tl.arange(0, kv_hidden_size)
    kv_mask = kv_cols < kv_hidden_size

    # [HEAD0_q, HEAD0_gate, HEAD1_q, HEAD1_gate, ..., KV_HEAD0_k, KV_HEAD1_k, ..., KV_HEAD0_v, KV_HEAD1_v, ...]
    # q_gate = q norm + gate copy
    for _ in tl.range(num_heads):
        # q norm
        i_q = tl.load(input_ptr + input_offset + q_gate_cols,
                      mask=q_gate_mask,
                      other=0.0).to(tl.float32)
        squa = i_q * i_q
        var = tl.sum(squa) / HEAD_DIM
        rstd = tl.rsqrt(var + eps)
        norm = i_q * rstd
        o_q = norm * (q_norm_weight + 1.0)
        tl.store(q_ptr + output_offset + q_gate_cols,
                 o_q.to(tl.bfloat16),
                 mask=q_gate_mask)

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
    # k norm
    i_k = tl.load(input_ptr + input_offset + kv_cols, mask=kv_mask,
                  other=0.0).reshape(num_kv_heads, HEAD_DIM).to(tl.float32)
    squa = i_k * i_k
    var = tl.sum(squa, axis=1) / HEAD_DIM
    rstd = tl.rsqrt(var + eps)
    norm = i_k * rstd[:, None]
    o_k = norm * (k_norm_weight + 1.0)
    tl.store(k_ptr + output_offset + kv_cols,
             o_k.reshape(kv_hidden_size).to(tl.bfloat16),
             mask=kv_mask)

    input_offset += kv_hidden_size

    # v copy
    i_v = tl.load(input_ptr + input_offset + kv_cols, mask=kv_mask, other=0.0)
    tl.store(v_ptr + output_offset + kv_cols, i_v, mask=kv_mask)


def qk_rmsnorm_triton(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    has_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert triton.next_power_of_2(
        head_dim
    ) == head_dim, "head_dim must be a power of 2 for triton kernel"
    bs, total_hidden_size = input.shape
    assert bs <= MAX_CORES
    q_gate_hidden_size = num_heads * head_dim
    kv_hidden_size = num_kv_heads * head_dim
    assert total_hidden_size == 2 * kv_hidden_size + (
        1 + has_gate) * q_gate_hidden_size

    input = input.contiguous()
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()

    q_output = torch.empty(bs,
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

    grid = (bs, )
    if has_gate:
        gate_output = torch.empty(bs,
                                  q_gate_hidden_size,
                                  device=input.device,
                                  dtype=input.dtype)
        qk_rmsnorm_gate_triton_kernel[grid](
            input,
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
        )
        return q_output, k_output, v_output, gate_output
    else:
        qk_rmsnorm_triton_kernel[grid](
            input,
            q_output,
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
        )
        return q_output, k_output, v_output, None
