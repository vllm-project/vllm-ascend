# SPDX-License-Identifier: Apache-2.0
"""Single-node validation: accounted FP8 and FP4 caches in one padded page.

The sparse model reserves physical head size256 for logical head size128.
Each physical K/V plane contains [FP8 page][FP4 page][BF16 scales][padding].
Prefill reads FP8; decode reads FP4. Both are produced in the same kernel.
"""

import torch
from vllm.triton_utils import tl, triton

A8C4_PHYSICAL_HEAD_SIZE = 256


def a8c4_cache_views(caches: tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
    k, v = caches[:2]
    if k.ndim != 4 or k.shape[-1] != A8C4_PHYSICAL_HEAD_SIZE or k.dtype != torch.float8_e4m3fn:
        return None
    if k.ndim != 4 or k.shape != v.shape or v.dtype != k.dtype or k.shape[1] != 128:
        raise ValueError("A8C4 requires matching FP8 [blocks, 128, heads, 256] K/V planes")
    blocks, tokens, heads, _ = k.shape
    views = []
    for cache in (k, v):
        if cache.stride()[1:] != (heads * 256, 256, 1):
            raise ValueError("A8C4 requires contiguous token/head dimensions")
        raw = cache.view(torch.uint8)
        stride = raw.stride(0)
        start = raw.storage_offset()
        fp8_bytes = tokens * heads * 128
        fp4_bytes = tokens * heads * 64
        if stride < tokens * heads * 256 or start % 2:
            raise ValueError("A8C4 requires non-overlapping blocks and BF16-aligned storage")
        fp8 = torch.as_strided(raw, (blocks, tokens, heads, 128), (stride, heads * 128, 128, 1), start).view(
            torch.float8_e4m3fn
        )
        data = torch.as_strided(raw, (blocks, tokens, heads, 64), (stride, heads * 64, 64, 1), start + fp8_bytes)
        scale = torch.as_strided(
            raw.view(torch.bfloat16),
            (blocks, tokens, heads, 8),
            (stride // 2, heads * 8, 8, 1),
            (start + fp8_bytes + fp4_bytes) // 2,
        )
        # The mode8 C++ bridge explicitly wraps packed uint8 as ACL_FLOAT4_E2M1.
        views.append((fp8, data, scale, data))
    return views


@triton.jit
def _store_fp4(values, data, scales, slot, HEADS: tl.constexpr, DATA_BLOCK: tl.constexpr, SCALE_BLOCK: tl.constexpr):
    x = values.to(tl.float32).reshape(HEADS * 8, 16)
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.where(amax > 0, amax / 6.0, 1.0).to(tl.bfloat16).to(tl.float32)
    y = tl.minimum(tl.abs(x / scale[:, None]), 6.0)
    code = (
        (y >= 0.25).to(tl.int32)
        + (y >= 0.75).to(tl.int32)
        + (y >= 1.25).to(tl.int32)
        + (y >= 1.75).to(tl.int32)
        + (y >= 2.5).to(tl.int32)
        + (y >= 3.5).to(tl.int32)
        + (y >= 5.0).to(tl.int32)
    )
    even_lower_tie = (y == 0.25) | (y == 1.25) | (y == 2.5) | (y == 5.0)
    code = code - even_lower_tie.to(tl.int32)
    code = code | ((x < 0).to(tl.int32) << 3)
    pair = code.reshape(HEADS, 64, 2)
    packed = tl.sum(pair << (tl.arange(0, 2)[None, None, :] * 4), axis=2).to(tl.uint8)
    row = slot % 128
    block = slot // 128
    h = tl.arange(0, HEADS)
    tl.store(data + block * DATA_BLOCK + row * HEADS * 64 + h[:, None] * 64 + tl.arange(0, 64)[None, :], packed)
    tl.store(scales + block * SCALE_BLOCK + row * HEADS * 8 + tl.arange(0, HEADS * 8), scale)


@triton.jit
def _dual_scatter_kernel(
    k,
    v,
    k8,
    v8,
    kp,
    vp,
    ks,
    vs,
    slots,
    T: tl.constexpr,
    H: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    KB: tl.constexpr,
    VB: tl.constexpr,
    PBK: tl.constexpr,
    PBV: tl.constexpr,
    SBK: tl.constexpr,
    SBV: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token).to(tl.int64)
    if slot >= 0:
        h = tl.arange(0, H)
        d = tl.arange(0, 128)
        key = tl.load(k + token * K0 + h[:, None] * K1 + d[None, :]).to(tl.float32)
        val = tl.load(v + token * V0 + h[:, None] * V1 + d[None, :]).to(tl.float32)
        _store_fp4(key, kp, ks, slot, H, PBK, SBK)
        _store_fp4(val, vp, vs, slot, H, PBV, SBV)
        offset = slot % 128 * H * 128 + h[:, None] * 128 + d[None, :]
        tl.store(k8 + slot // 128 * KB + offset, tl.minimum(tl.maximum(key, -448.0), 448.0))
        tl.store(v8 + slot // 128 * VB + offset, tl.minimum(tl.maximum(val, -448.0), 448.0))


def dual_cache_scatter(key, value, caches, slots):
    views = a8c4_cache_views(caches)
    if views is None:
        raise ValueError("dual_cache_scatter requires A8C4 dual-format pages")
    if (
        key.ndim != 3
        or key.shape != value.shape
        or key.dtype != torch.bfloat16
        or value.dtype != key.dtype
        or key.stride(-1) != 1
        or value.stride(-1) != 1
        or key.device != caches[0].device
        or value.device != key.device
        or slots.device != key.device
        or slots.dtype != torch.int64
        or not slots.is_contiguous()
        or key.shape[1:] != (caches[0].shape[2], 128)
        or slots.ndim != 1
        or slots.numel() > key.shape[0]
    ):
        raise ValueError("A8C4 scatter requires [tokens, heads, 128] K/V and one slot per active token")
    (k8, kp, ks, _), (v8, vp, vs, _) = views
    count = slots.numel()
    if count:
        _dual_scatter_kernel[(count,)](
            key,
            value,
            k8,
            v8,
            kp,
            vp,
            ks,
            vs,
            slots,
            count,
            key.shape[1],
            *key.stride()[:2],
            *value.stride()[:2],
            k8.stride(0),
            v8.stride(0),
            kp.stride(0),
            vp.stride(0),
            ks.stride(0),
            vs.stride(0),
            enable_fp_fusion=False,
        )
