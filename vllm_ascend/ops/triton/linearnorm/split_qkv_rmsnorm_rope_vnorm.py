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
# `split_qkv_rmsnorm_rope` gives V a second loop with a much wider tile because
# a plain copy needs no scratch. Once V has to go through the same float32
# reduction as Q and K that no longer holds, so this kernel folds V into the
# q/k loop: one load covers the whole fused row, and one RMSNorm reduction
# covers every q, k and v head of the tile.
#

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_element, get_vectorcore_num, insert_slice

# Unified buffer budget of a single vector core.
UB_SIZE = 87040  # 85K = 85 * 1024

# UB bytes one token occupies. Every element of the fused qkv row is held as the
# bfloat16 value that was loaded plus its float32 copy used for the RMSNorm
# reduction (6 bytes). Every q element is additionally held as the bfloat16 norm
# output and the bfloat16 RoPE result (4 bytes); the k tiles reuse the buffers
# freed by the q tiles. Every v element is held once more as the bfloat16 store
# buffer. The cos/sin row is bfloat16 and is shared by all heads of the token.
UB_BYTES_PER_QKV_ELEMENT = 6
UB_BYTES_PER_Q_ELEMENT = 4
UB_BYTES_PER_V_ELEMENT = 2
UB_BYTES_PER_ROPE_ELEMENT = 2


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
    qkv_head_nums_per_iter_per_vec: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    qk_head_num_sum: tl.constexpr,
    qkv_head_num_sum: tl.constexpr,
    positions_gm_ptr,
    cos_sin_cache_gm_ptr,
):
    row_pid = tl.program_id(0)

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

    batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
    iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
    input_batch_offset = row_pid * batch_size_per_vec
    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(0, total_hidden_size)
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
        values_tmp1 = tl.load(input_gm_ptr + idx, mask=mask).reshape(qkv_head_nums_per_iter_per_vec, HEAD_DIM)
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

        # One reduction for every q, k and v head of the tile. V shares it
        # because its norm is the same per-head RMSNorm with the same eps, it
        # only skips the learnable scale that q and k apply below.
        normalized_values = values_tmp1.to(tl.float32)
        normalized_values = normalized_values * normalized_values
        normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
        normalized_values = 1 / tl.sqrt(normalized_values + eps).reshape(qkv_head_nums_per_iter_per_vec, 1)
        normalized_values = values_tmp1 * normalized_values

        normalized_values_tmp = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qkv_head_num_sum, HEAD_DIM),
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
            normalized_values.reshape(batch_size_per_iter_per_vec, qkv_head_num_sum, HEAD_DIM),
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

        # v norm. The v heads sit behind the q and k ones in the same tile and
        # were normalized by the reduction above, so all that is left is to cut
        # them out and store them. No weight and no rope, and the k output index
        # and mask apply unchanged.
        normalized_v = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qkv_head_num_sum, HEAD_DIM),
            offsets=(0, qk_head_num_sum, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        ).to(tl.bfloat16)
        tl.store(
            v_gm_ptr + kv_output_idx,
            normalized_v.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
            mask=mask,
        )


def qkv_batch_size_per_iter_per_vec(
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rope_dim: int,
    element_size: int,
) -> int:
    """Number of tokens the kernel processes per iteration and vector core.

    set number of line loading from GM data is x
    x*(q_head_num + 2*kv_head_num)*HEAD_DIM: values_tmp
    2x*(q_head_num + 2*kv_head_num)*HEAD_DIM: normalized_values(float32)
    x*ROPE_DIM*2 : cos/sin
    x*q_head_num*HEAD_DIM*2: normalized_values_tmp
    x*kv_head_num*HEAD_DIM: normalized_v
    x*q_head_num*ROPE_DIM*(0.5) (not IS_PARTIAL_ROPE) x*q_head_num*ROPE_DIM*(0.5): y

    This is the `split_qkv_rmsnorm_rope` accounting with the v columns moved
    into the tile: the loaded slice and its float32 copy grow from q+kv to
    q+2*kv (3*kv_hidden_size more elements) and the normalized v output adds one
    more bfloat16 copy of the v columns, so the factor gains 4*kv_hidden_size.

    The factor is the sum of elements number. It also counts buffers that are
    not live at the same time, so it is a tiling heuristic rather than a UB
    feasibility check; see :func:`qkv_rmsnorm_rope_vnorm_fits_ub` for the
    latter.
    """
    q_head_num = q_hidden_size // head_dim
    if rope_dim != head_dim:
        factor = 5 * q_hidden_size + 7 * kv_hidden_size + rope_dim * 4 + q_head_num * rope_dim
    else:
        factor = 5 * q_hidden_size + 7 * kv_hidden_size + rope_dim * 2 + q_head_num * rope_dim // 2
    return int(UB_SIZE / element_size) // factor


def qkv_rmsnorm_rope_vnorm_fits_ub(
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rope_dim: int,
) -> bool:
    """Whether a single token fits into one vector core's unified buffer.

    The kernel always processes at least one token per iteration, so a layer
    whose per-token tile exceeds the UB budget cannot be served by this kernel
    at all. Callers are expected to fall back to the unfused implementation in
    that case.
    """
    ub_bytes_per_token = (
        UB_BYTES_PER_QKV_ELEMENT * (q_hidden_size + 2 * kv_hidden_size)
        + UB_BYTES_PER_Q_ELEMENT * q_hidden_size
        + UB_BYTES_PER_V_ELEMENT * kv_hidden_size
        + UB_BYTES_PER_ROPE_ELEMENT * rope_dim
    )
    return (
        head_dim > 0
        and q_hidden_size % head_dim == 0
        and kv_hidden_size % head_dim == 0
        and ub_bytes_per_token <= UB_SIZE
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

    qkv_batch_size = max(
        1,
        qkv_batch_size_per_iter_per_vec(
            q_hidden_size=q_hidden_size,
            kv_hidden_size=kv_hidden_size,
            head_dim=head_dim,
            rope_dim=rope_dim,
            element_size=input.element_size(),
        ),
    )
    qk_head_num_sum = int(q_head_num + kv_head_num)
    qkv_head_num_sum = int(q_head_num + 2 * kv_head_num)
    qkv_head_nums_per_iter_per_vec = qkv_batch_size * qkv_head_num_sum

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
        int(qkv_batch_size),
        int(qkv_head_nums_per_iter_per_vec),
        q_head_num,
        kv_head_num,
        qk_head_num_sum,
        qkv_head_num_sum,
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
