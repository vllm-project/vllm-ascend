#
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
# Variant of `split_qkv_rmsnorm_rope` that also normalizes V.
#
# Models such as Gemma4 normalize V per head with a weight-less RMSNorm
# (`RMSNorm(head_dim, has_weight=False)`) before attention, so V cannot simply
# be copied out of the fused QKV tensor as `split_qkv_rmsnorm_rope` does.
#

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_element, get_vectorcore_num, insert_slice

# Unified buffer budget of a single vector core.
UB_SIZE = 87040  # 85K = 85 * 1024

# UB bytes one token occupies in the V stage: the bfloat16 row that is loaded,
# its float32 copy used for the variance reduction and the bfloat16 row that is
# stored.
V_UB_BYTES_PER_ELEMENT = 8

# UB bytes one token occupies in the Q/K stage. Every element of the Q/K slice
# is held as the bfloat16 value that was loaded plus its float32 copy used for
# the RMSNorm reduction (6 bytes), and every Q element is additionally held as
# the bfloat16 norm output and the bfloat16 RoPE result (4 bytes). The K tiles
# reuse the buffers freed by the Q tiles. The cos/sin row is bfloat16 and is
# shared by all heads of the token.
QK_UB_BYTES_PER_QKV_ELEMENT = 6
QK_UB_BYTES_PER_Q_ELEMENT = 4
QK_UB_BYTES_PER_ROPE_ELEMENT = 2


@triton.jit
def split_qkv_rmsnorm_rope_vnorm_kernel(
    input_gm_ptr,
    q_gm_ptr,
    k_gm_ptr,
    v_gm_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    num_vectorcore: tl.constexpr,
    batch_size_per_iter_per_vec: tl.constexpr,
    qk_head_nums_per_iter_per_vec: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    qk_head_num_sum: tl.constexpr,
    v_batch_size_per_iter_per_vec: tl.constexpr,
    v_head_nums_per_iter_per_vec: tl.constexpr,
    positions_gm_ptr,
    cos_sin_cache_gm_ptr,
):
    row_pid = tl.program_id(0)

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

    batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
    iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
    v_iter_num_per_vec = tl.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)
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
    sin_cos_range = tl.arange(0, ROPE_DIM)
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

        values_tmp3 = tl.zeros((batch_size_per_iter_per_vec, ROPE_DIM), dtype=tl.bfloat16)
        for i in tl.range(batch_size_per_iter_per_vec):
            pos = get_element(x, (i,))
            values_tmp3 = insert_slice(
                values_tmp3.reshape(batch_size_per_iter_per_vec, ROPE_DIM),
                tl.load(pos * ROPE_DIM + cos_sin_cache_offset[:, None]).reshape(1, ROPE_DIM),
                offsets=(i, 0),
                sizes=(1, ROPE_DIM),
                strides=(1, 1),
            )
        values_tmp3 = values_tmp3.reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)
        cos = extract_slice(
            values_tmp3,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        sin = extract_slice(
            values_tmp3,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        normalized_values = values_tmp1.to(tl.float32)
        normalized_values = normalized_values * normalized_values
        normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
        normalized_values = 1 / tl.sqrt(normalized_values + eps).reshape(qk_head_nums_per_iter_per_vec, 1)
        normalized_values = values_tmp1 * normalized_values

        normalized_values_tmp = extract_slice(
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
        x1 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        q_output_idx = output_q_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * q_hidden_size
        mask = (mmask[:, None]) & (output_q_nmask[None, :])
        if IS_PARTIAL_ROPE:
            normalized_values_tmp = insert_slice(
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
            tl.store(
                q_gm_ptr + q_output_idx,
                values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=mask,
            )

        # k rope
        normalized_values_tmp1 = extract_slice(
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

        x1 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        kv_output_idx = output_kv_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * kv_hidden_size
        mask = (mmask[:, None]) & (output_kv_nmask[None, :])
        if IS_PARTIAL_ROPE:
            normalized_values_tmp1 = insert_slice(
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
        values = tl.load(input_gm_ptr + idx, mask=mask).reshape(v_head_nums_per_iter_per_vec, HEAD_DIM)

        # v rmsnorm, per head and without a learnable scale. The reduction runs
        # in float32 for the same reason as the q/k one above.
        v_reciprocal_std = values.to(tl.float32)
        v_reciprocal_std = v_reciprocal_std * v_reciprocal_std
        v_reciprocal_std = tl.sum(v_reciprocal_std, axis=1) / HEAD_DIM
        v_reciprocal_std = 1 / tl.sqrt(v_reciprocal_std + eps).reshape(v_head_nums_per_iter_per_vec, 1)
        normalized_v = (values * v_reciprocal_std).to(tl.bfloat16)

        out_idx = mblk_idx[:, None] * kv_hidden_size + out_nblk_idx[None, :]
        out_mask = (mmask[:, None]) & (out_nmask[None, :])
        tl.store(
            v_gm_ptr + out_idx,
            normalized_v.reshape(v_batch_size_per_iter_per_vec, kv_hidden_size),
            mask=out_mask,
        )
        mblk_idx += v_batch_size_per_iter_per_vec


def qk_batch_size_per_iter_per_vec(
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rope_dim: int,
    element_size: int,
) -> int:
    """Number of tokens the q/k stage processes per iteration and vector core.

    set number of line loading from GM data is x
    x*(q_head_num + kv_head_num)*HEAD_DIM: values_tmp
    2x*(q_head_num + kv_head_num)*HEAD_DIM: normalized_values(float32)
    x*ROPE_DIM*2 : cos/sin
    x*q_head_num*HEAD_DIM*2： normalized_values_tmp
    x*q_head_num*ROPE_DIM*(0.5) (not IS_PARTIAL_ROPE) x*q_head_num*ROPE_DIM*(0.5): y

    The factor is the sum of elements number. It also counts buffers that are
    not live at the same time, so it is a tiling heuristic rather than a UB
    feasibility check; see :func:`qkv_rmsnorm_rope_vnorm_fits_ub` for the
    latter.
    """
    q_head_num = q_hidden_size // head_dim
    if rope_dim != head_dim:
        factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 4 + q_head_num * rope_dim
    else:
        factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 2 + q_head_num * rope_dim // 2
    return int(UB_SIZE / element_size) // factor


def v_batch_size_per_iter_per_vec(kv_hidden_size: int) -> int:
    """Number of tokens the v stage processes per iteration and vector core."""
    return UB_SIZE // (V_UB_BYTES_PER_ELEMENT * kv_hidden_size)


def qkv_rmsnorm_rope_vnorm_fits_ub(
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rope_dim: int,
) -> bool:
    """Whether a single token fits into one vector core's unified buffer.

    Both stages of the kernel always process at least one token per iteration,
    so a layer whose per-token tiles exceed the UB budget cannot be served by
    this kernel at all. Callers are expected to fall back to the unfused
    implementation in that case.
    """
    qk_ub_bytes_per_token = (
        QK_UB_BYTES_PER_QKV_ELEMENT * (q_hidden_size + kv_hidden_size)
        + QK_UB_BYTES_PER_Q_ELEMENT * q_hidden_size
        + QK_UB_BYTES_PER_ROPE_ELEMENT * rope_dim
    )
    v_ub_bytes_per_token = V_UB_BYTES_PER_ELEMENT * kv_hidden_size
    return (
        head_dim > 0
        and q_hidden_size % head_dim == 0
        and kv_hidden_size % head_dim == 0
        and qk_ub_bytes_per_token <= UB_SIZE
        and v_ub_bytes_per_token <= UB_SIZE
    )


def split_qkv_rmsnorm_rope_vnorm_impl(
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
    """Split a fused QKV tensor, RMSNorm and RoPE q/k, and RMSNorm v.

    V is normalized per head without a learnable scale, matching a
    `RMSNorm(head_dim, has_weight=False)` layer.
    """
    # get available vector core
    num_vectorcore = get_vectorcore_num()
    rope_dim = cos_sin_cache.shape[-1]
    batch_size = input.shape[0]
    BIAS = q_bias is not None
    IS_PARTIAL_ROPE = rope_dim != head_dim
    # Q + K + V
    total_hidden_size = q_hidden_size + kv_hidden_size * 2

    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    q_head_num = q_hidden_size // head_dim
    kv_head_num = kv_hidden_size // head_dim

    qk_batch_size = max(
        1,
        qk_batch_size_per_iter_per_vec(
            q_hidden_size=q_hidden_size,
            kv_hidden_size=kv_hidden_size,
            head_dim=head_dim,
            rope_dim=rope_dim,
            element_size=input.element_size(),
        ),
    )
    qk_head_num_sum = int(q_head_num + kv_head_num)
    qk_head_nums_per_iter_per_vec = qk_batch_size * qk_head_num_sum

    # v tiling
    v_batch_size = max(1, v_batch_size_per_iter_per_vec(kv_hidden_size))
    v_head_nums_per_iter_per_vec = v_batch_size * kv_head_num

    grid = (num_vectorcore, 1, 1)

    split_qkv_rmsnorm_rope_vnorm_kernel[grid](
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
        num_vectorcore,
        int(qk_batch_size),
        int(qk_head_nums_per_iter_per_vec),
        q_head_num,
        kv_head_num,
        qk_head_num_sum,
        int(v_batch_size),
        int(v_head_nums_per_iter_per_vec),
        positions,
        cos_sin_cache,
    )
    return q_output, k_output, v_output


def split_qkv_rmsnorm_rope_vnorm_impl_fake(
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
        int(q_hidden_size),
        device=input.device,
        dtype=input.dtype,
    )
    k_output = torch.empty(
        batch_size,
        int(kv_hidden_size),
        device=input.device,
        dtype=input.dtype,
    )
    v_output = torch.empty(
        batch_size,
        int(kv_hidden_size),
        device=input.device,
        dtype=input.dtype,
    )
    return q_output, k_output, v_output


direct_register_custom_op(
    op_name="qkv_rmsnorm_rope_vnorm",
    op_func=split_qkv_rmsnorm_rope_vnorm_impl,
    fake_impl=split_qkv_rmsnorm_rope_vnorm_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

