# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MiniMax-M3 QK normalization, NeoX RoPE and paged cache insertion.

Strided main/index projection views produce Q and index Q. Programs
process tokens cyclically and reuse FP32 Gemma normalization scales. All
input rows produce queries; only actual tokens with nonnegative slots
write K/V or index K directly into their independent paged caches.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    extract_slice,
    get_vectorcore_num,
    init_device_properties_triton,
    insert_slice,
)

FP8_E4M3_MAX = 448.0


@triton.jit
def _norm_rope_with_scale(
    input_ptr,
    scale,
    cos,
    sin,
    HEADS: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    cols = tl.arange(0, HEAD_DIM)
    heads = tl.arange(0, HEADS)
    offsets = heads[:, None] * HEAD_DIM + cols[None, :]
    x = tl.load(input_ptr + offsets * COL_STRIDE).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 1) / HEAD_DIM + EPS)[:, None] * scale
    x = x.to(input_ptr.dtype.element_ty).to(tl.float32)
    half: tl.constexpr = ROTARY_DIM // 2
    a = extract_slice(x, (0, 0), (HEADS, half), (1, 1))
    b = extract_slice(x, (0, half), (HEADS, half), (1, 1))
    x = insert_slice(x, a * cos[None, :] - b * sin[None, :], (0, 0), (HEADS, half), (1, 1))
    x = insert_slice(x, b * cos[None, :] + a * sin[None, :], (0, half), (HEADS, half), (1, 1))
    return x


# Request-dependent bounds must not specialize on their value or alignment.
# Head dimensions and layout strides below are fixed for a warmed deployment.
@triton.jit(do_not_specialize=["NUM_SLOTS", "NUM_TOKENS"])
def _minimax_qknorm_rope_cache_kernel(
    q_input,
    k_input,
    v_input,
    iq_input,
    ik_input,
    cos_sin,
    positions,
    q_weight,
    k_weight,
    iq_weight,
    ik_weight,
    query,
    index_query,
    key_cache,
    value_cache,
    index_cache,
    slots,
    index_slots,
    NUM_SLOTS,
    NUM_TOKENS,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    IQ_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    Q_ROW: tl.constexpr,
    Q_COL: tl.constexpr,
    K_ROW: tl.constexpr,
    K_COL: tl.constexpr,
    V_ROW: tl.constexpr,
    V_COL: tl.constexpr,
    IQ_ROW: tl.constexpr,
    IQ_COL: tl.constexpr,
    IK_ROW: tl.constexpr,
    IK_COL: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    COS_SIN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    K_BLOCK: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_BLOCK: tl.constexpr,
    V_TOKEN: tl.constexpr,
    V_HEAD: tl.constexpr,
    I_BLOCK: tl.constexpr,
    I_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INDEX_BLOCK_SIZE: tl.constexpr,
    CACHE_FP8: tl.constexpr,
    INDEX_FP8: tl.constexpr,
    FP8_E4M3_MAX: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    programs = tl.num_programs(0).to(tl.int64)
    weight_dim = tl.arange(0, HEAD_DIM)
    q_scale = 1.0 + tl.load(q_weight + weight_dim).to(tl.float32)
    k_scale = 1.0 + tl.load(k_weight + weight_dim).to(tl.float32)
    iq_scale = 1.0 + tl.load(iq_weight + weight_dim).to(tl.float32)
    ik_scale = 1.0 + tl.load(ik_weight + weight_dim).to(tl.float32)
    half: tl.constexpr = ROTARY_DIM // 2
    rotary_offsets = tl.arange(0, half)
    for token in range(program, NUM_TOKENS, programs):
        position = tl.load(positions + token * POSITION_STRIDE).to(tl.int64)
        cs = cos_sin + position * COS_SIN_STRIDE + rotary_offsets
        cos = tl.load(cs).to(tl.float32)
        sin = tl.load(cs + half).to(tl.float32)
        dim = tl.arange(0, HEAD_DIM)
        q = _norm_rope_with_scale(q_input + token * Q_ROW, q_scale, cos, sin, Q_HEADS, Q_COL, HEAD_DIM, ROTARY_DIM, EPS)
        offsets = token * Q_HEADS * HEAD_DIM + tl.arange(0, Q_HEADS)[:, None] * HEAD_DIM + dim[None, :]
        tl.store(query + offsets, q)
        slot = tl.load(slots + token, token < NUM_SLOTS, other=-1).to(tl.int64)
        if slot >= 0:
            k = _norm_rope_with_scale(
                k_input + token * K_ROW, k_scale, cos, sin, KV_HEADS, K_COL, HEAD_DIM, ROTARY_DIM, EPS
            )
            k = k.to(k_input.dtype.element_ty).to(tl.float32)
            v_offsets = tl.arange(0, KV_HEADS)[:, None] * HEAD_DIM + dim[None, :]
            v = tl.load(v_input + token * V_ROW + v_offsets * V_COL)
            if CACHE_FP8:
                k = tl.minimum(tl.maximum(k, -FP8_E4M3_MAX), FP8_E4M3_MAX)
                v = tl.minimum(tl.maximum(v.to(tl.float32), -FP8_E4M3_MAX), FP8_E4M3_MAX)
            k_offset = slot // BLOCK_SIZE * K_BLOCK + slot % BLOCK_SIZE * K_TOKEN
            v_offset = slot // BLOCK_SIZE * V_BLOCK + slot % BLOCK_SIZE * V_TOKEN
            tl.store(key_cache + k_offset + tl.arange(0, KV_HEADS)[:, None] * K_HEAD + dim[None, :], k)
            tl.store(value_cache + v_offset + tl.arange(0, KV_HEADS)[:, None] * V_HEAD + dim[None, :], v)
        iq = _norm_rope_with_scale(
            iq_input + token * IQ_ROW, iq_scale, cos, sin, IQ_HEADS, IQ_COL, HEAD_DIM, ROTARY_DIM, EPS
        )
        if INDEX_FP8:
            iq = tl.minimum(tl.maximum(iq, -FP8_E4M3_MAX), FP8_E4M3_MAX)
        offsets = token * IQ_HEADS * HEAD_DIM + tl.arange(0, IQ_HEADS)[:, None] * HEAD_DIM + dim[None, :]
        tl.store(index_query + offsets, iq)
        slot = tl.load(index_slots + token, token < NUM_SLOTS, other=-1).to(tl.int64)
        if slot >= 0:
            ik = _norm_rope_with_scale(
                ik_input + token * IK_ROW, ik_scale, cos, sin, 1, IK_COL, HEAD_DIM, ROTARY_DIM, EPS
            )
            if INDEX_FP8:
                ik = tl.minimum(tl.maximum(ik, -FP8_E4M3_MAX), FP8_E4M3_MAX)
            offset = slot // INDEX_BLOCK_SIZE * I_BLOCK + slot % INDEX_BLOCK_SIZE * I_TOKEN
            tl.store(index_cache + offset + dim, ik.reshape(HEAD_DIM))


def minimax_qknorm_rope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    num_actual_tokens: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Q/index Q and insert K/V/index K, with no intermediate KV tensors.

    Projections are strided [token, head * dim] views.
    Caches use [block, token, head, dim] and [block, token, dim] layouts.
    Head dimensions must match (the MiniMax-M3 128-wide main/index heads).
    The caller owns cache lifetime and orders cache readers after this call.
    """
    head_dim = q_weight.numel()
    rotary_dim = cos_sin_cache.shape[-1]
    assert head_dim == k_weight.numel() == index_q_weight.numel() == index_k_weight.numel()
    assert head_dim in (64, 128, 256) and 0 < rotary_dim <= head_dim and rotary_dim % 2 == 0
    projections = (q, k, v, index_q, index_k)
    assert q.ndim == 2 and q.dtype == torch.bfloat16
    num_tokens = q.shape[0]
    assert all(
        x.ndim == 2 and x.shape[0] == num_tokens and x.dtype == q.dtype and x.device == q.device for x in projections
    )
    assert all(x.shape[1] > 0 and x.shape[1] % head_dim == 0 for x in projections)
    assert v.shape == k.shape and index_k.shape[1] == head_dim
    num_q_heads = q.shape[1] // head_dim
    num_kv_heads = k.shape[1] // head_dim
    num_index_heads = index_q.shape[1] // head_dim
    assert key_cache.ndim == value_cache.ndim == 4 and index_cache.ndim == 3
    assert key_cache.shape == value_cache.shape and key_cache.shape[2:] == (num_kv_heads, head_dim)
    assert index_cache.shape[-1] == head_dim
    assert key_cache.stride(-1) == value_cache.stride(-1) == index_cache.stride(-1) == 1
    assert 0 <= num_actual_tokens <= num_tokens
    assert slot_mapping.numel() >= num_actual_tokens and index_slot_mapping.numel() >= num_actual_tokens
    assert positions.ndim == 1 and positions.numel() == num_tokens
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.stride(1) == 1
    assert key_cache.dtype == value_cache.dtype and key_cache.dtype in (q.dtype, torch.float8_e4m3fn)
    assert index_cache.dtype in (q.dtype, torch.float8_e4m3fn)
    assert all(w.ndim == 1 and w.is_contiguous() for w in (q_weight, k_weight, index_q_weight, index_k_weight))
    assert slot_mapping.ndim == index_slot_mapping.ndim == 1
    assert slot_mapping.is_contiguous() and index_slot_mapping.is_contiguous()
    query = torch.empty((num_tokens, num_q_heads * head_dim), device=q.device, dtype=q.dtype)
    index_query = torch.empty((num_tokens, num_index_heads * head_dim), device=q.device, dtype=index_cache.dtype)
    if num_tokens == 0:
        return query, index_query
    init_device_properties_triton()
    # Each program owns tokens congruent to its id modulo this grid.
    num_programs = min(num_tokens, get_vectorcore_num())
    _minimax_qknorm_rope_cache_kernel[(num_programs,)](
        *projections,
        cos_sin_cache,
        positions,
        q_weight,
        k_weight,
        index_q_weight,
        index_k_weight,
        query,
        index_query,
        key_cache,
        value_cache,
        index_cache,
        slot_mapping,
        index_slot_mapping,
        num_actual_tokens,
        num_tokens,
        num_q_heads,
        num_kv_heads,
        num_index_heads,
        head_dim,
        rotary_dim,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *index_q.stride(),
        *index_k.stride(),
        positions.stride(0),
        cos_sin_cache.stride(0),
        eps,
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        *index_cache.stride()[:2],
        key_cache.shape[1],
        index_cache.shape[1],
        key_cache.dtype == torch.float8_e4m3fn,
        index_cache.dtype == torch.float8_e4m3fn,
        FP8_E4M3_MAX=FP8_E4M3_MAX,
        enable_fp_fusion=False,
    )
    return query, index_query
