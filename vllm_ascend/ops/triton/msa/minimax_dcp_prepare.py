# SPDX-License-Identifier: Apache-2.0
"""Separate fused prepare branches for MiniMax-M3 BF16 decode overlap."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache import _norm_rope_with_scale
from vllm_ascend.ops.triton.triton_utils import extract_slice


@triton.jit(do_not_specialize=["tokens", "actual"])
def _index_prepare(
    packed,
    cos_sin,
    positions,
    qw,
    kw,
    query,
    cache,
    slots,
    tokens,
    actual,
    ROW: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    CS: tl.constexpr,
    CP: tl.constexpr,
    CT: tl.constexpr,
    HEADS: tl.constexpr,
    ROTARY: tl.constexpr,
    EPS: tl.constexpr,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 128)
    weight = tl.where(tl.arange(0, HEADS + 1)[:, None] < HEADS, tl.load(qw + d)[None, :], tl.load(kw + d)[None, :]).to(
        tl.float32
    )
    # Cast weights before adding one, matching the combined prepare kernel.
    position = tl.load(positions + token * POSITION_STRIDE).to(tl.int64)
    qk = _norm_rope_with_scale(packed + token * ROW, 1.0 + weight, cos_sin + position * CS, HEADS + 1, 128, ROTARY, EPS)
    q = extract_slice(qk, (0, 0), (HEADS, 128), (1, 1))
    tl.store(query + token * HEADS * 128 + tl.arange(0, HEADS)[:, None] * 128 + d[None, :], q)
    slot = tl.load(slots + token, token < actual, other=-1).to(tl.int64)
    if slot >= 0 and slot % SIZE == RANK:
        k = extract_slice(qk, (HEADS, 0), (1, 128), (1, 1))
        tl.store(cache + slot // 128 * CP + slot % 128 // SIZE * CT + d, k.reshape(128))


@triton.jit(do_not_specialize=["tokens", "actual"])
def _main_prepare(
    packed,
    cos_sin,
    positions,
    qw,
    kw,
    query,
    key,
    value,
    slots,
    tokens,
    actual,
    ROW: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    CS: tl.constexpr,
    KP: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VT: tl.constexpr,
    VH: tl.constexpr,
    BLOCK: tl.constexpr,
    Q: tl.constexpr,
    KV: tl.constexpr,
    ROTARY: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 128)
    weight = tl.where(tl.arange(0, Q + KV)[:, None] < Q, tl.load(qw + d)[None, :], tl.load(kw + d)[None, :]).to(
        tl.float32
    )
    position = tl.load(positions + token * POSITION_STRIDE).to(tl.int64)
    qk = _norm_rope_with_scale(packed + token * ROW, 1.0 + weight, cos_sin + position * CS, Q + KV, 128, ROTARY, EPS)
    q = extract_slice(qk, (0, 0), (Q, 128), (1, 1))
    tl.store(query + token * Q * 128 + tl.arange(0, Q)[:, None] * 128 + d[None, :], q)
    slot = tl.load(slots + token, token < actual, other=-1).to(tl.int64)
    if slot >= 0:
        k = extract_slice(qk, (Q, 0), (KV, 128), (1, 1))
        v = tl.load(packed + token * ROW + (Q + KV) * 128 + tl.arange(0, KV)[:, None] * 128 + d[None, :])
        tl.store(key + slot // BLOCK * KP + slot % BLOCK * KT + tl.arange(0, KV)[:, None] * KH + d[None, :], k)
        tl.store(value + slot // BLOCK * VP + slot % BLOCK * VT + tl.arange(0, KV)[:, None] * VH + d[None, :], v)


def _validate_prepare_inputs(packed, cos_sin, positions, qw, kw, slots, actual, width):
    assert packed.dtype == torch.bfloat16 and packed.ndim == 2 and packed.stride(1) == 1
    assert packed.shape[1] == width
    assert positions.ndim == 1 and positions.numel() == packed.shape[0]
    assert positions.dtype in (torch.int32, torch.int64)
    assert cos_sin.ndim == 2 and cos_sin.stride(1) == 1
    assert 0 < cos_sin.shape[1] <= 128 and cos_sin.shape[1] % 2 == 0
    assert slots.ndim == 1 and slots.is_contiguous() and slots.numel() >= actual
    assert slots.dtype in (torch.int32, torch.int64) and 0 <= actual <= packed.shape[0]
    assert all(w.shape == (128,) and w.is_contiguous() for w in (qw, kw))
    assert all(t.device == packed.device for t in (cos_sin, positions, qw, kw, slots))


def minimax_dcp_index_prepare(packed, cos_sin, positions, qw, kw, cache, slots, actual, heads, eps, size, rank):
    """Prepare index Q and write this rank's compact index K cache."""
    _validate_prepare_inputs(packed, cos_sin, positions, qw, kw, slots, actual, (heads + 1) * 128)
    assert size in (2, 4) and 0 <= rank < size and heads == 4 // size
    assert cache.ndim == 3 and cache.shape[1:] == (128 // size, 128)
    assert cache.dtype == packed.dtype and cache.device == packed.device and cache.stride(-1) == 1
    query = packed.new_empty((packed.shape[0], heads * 128))
    if packed.shape[0] == 0:
        return query
    _index_prepare[(packed.shape[0],)](
        packed,
        cos_sin,
        positions,
        qw,
        kw,
        query,
        cache,
        slots,
        packed.shape[0],
        actual,
        packed.stride(0),
        positions.stride(0),
        cos_sin.stride(0),
        *cache.stride()[:2],
        heads,
        cos_sin.shape[1],
        eps,
        size,
        rank,
        enable_fp_fusion=False,
    )
    return query


def minimax_dcp_main_prepare(packed, cos_sin, positions, qw, kw, key, value, slots, actual, heads, kv_heads, eps):
    """Prepare main Q and write head-sharded BF16 K/V caches."""
    _validate_prepare_inputs(packed, cos_sin, positions, qw, kw, slots, actual, (heads + 2 * kv_heads) * 128)
    assert key.ndim == value.ndim == 4 and key.shape == value.shape
    assert key.shape[1:] == (128, kv_heads, 128)
    assert key.dtype == value.dtype == packed.dtype and key.device == value.device == packed.device
    assert key.stride(-1) == value.stride(-1) == 1
    query = packed.new_empty((packed.shape[0], heads * 128))
    if packed.shape[0] == 0:
        return query
    _main_prepare[(packed.shape[0],)](
        packed,
        cos_sin,
        positions,
        qw,
        kw,
        query,
        key,
        value,
        slots,
        packed.shape[0],
        actual,
        packed.stride(0),
        positions.stride(0),
        cos_sin.stride(0),
        *key.stride()[:3],
        *value.stride()[:3],
        key.shape[1],
        heads,
        kv_heads,
        cos_sin.shape[1],
        eps,
        enable_fp_fusion=False,
    )
    return query
