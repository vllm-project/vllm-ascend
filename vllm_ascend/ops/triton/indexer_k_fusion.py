# SPDX-License-Identifier: Apache-2.0
"""Experimental DeepSeek V4.1 Ascend A2/A3 INT8 indexer K epilogue.

Not hardware-validated. Requires vLLM + Triton Ascend on an NPU host.
Input is wk(latent), BEFORE k_norm; wk remains a separate GEMM.
cos/sin are already selected for each source position, repeated interleaved
as in vllm_ascend/ops/rope_dsv4.py: [T, 1, 1, R] or [T, R].
slots: physical [block, offset] coordinates, [T, 2]; negatives skip.
Caches: separate [blocks, page_size, 1, D] INT8 and
[blocks, page_size, 1, 1] FP16 scale tensors; physical strides are respected.
Valid output slots must be unique; caller owns metadata and stream ordering.
Finite inputs only. Zero rows produce q=0, scale=0 (check against your CANN).
Do not use for A5 packed/MXFP4 caches or folded-cache publication.
"""
import torch
from vllm.triton_utils import triton, tl


@triton.jit
def _round_even(x):
    # Explicit nearest-even; integer casts alone would truncate.
    lo = tl.floor(x)
    frac = x - lo
    odd = (lo.to(tl.int32) & 1) != 0
    return lo + ((frac > 0.5) | ((frac == 0.5) & odd)).to(tl.float32)


@triton.jit
def _indexer_k_norm_rope_int8_store(
    X, W, COS, SIN, SLOTS, KC, SC,
    XS: tl.constexpr, WS: tl.constexpr,
    CS0: tl.constexpr, CS1: tl.constexpr,
    SS0: tl.constexpr, SS1: tl.constexpr,
    IS0: tl.constexpr, IS1: tl.constexpr,
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr,
    NB: tl.constexpr, PAGE: tl.constexpr,
    D: tl.constexpr, R: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.load(SLOTS + row * IS0).to(tl.int64)
    slot = tl.load(SLOTS + row * IS0 + IS1).to(tl.int64)
    if (block >= 0) & (block < NB) & (slot >= 0) & (slot < PAGE):
        pair = tl.arange(0, D // 2)
        even = 2 * pair
        odd = even + 1
        a = tl.load(X + row * XS + even).to(tl.float32)
        b = tl.load(X + row * XS + odd).to(tl.float32)
        inv = tl.rsqrt((tl.sum(a * a, 0) + tl.sum(b * b, 0)) / D + EPS)
        wa = tl.load(W + even * WS).to(tl.float32)
        wb = tl.load(W + odd * WS).to(tl.float32)
        # Preserve the BF16 boundary between RMSNorm and RoPE.
        a = (a * inv * wa).to(tl.bfloat16).to(tl.float32)
        b = (b * inv * wb).to(tl.bfloat16).to(tl.float32)
        local = even - (D - R)
        rotary = local >= 0
        safe_local = tl.maximum(local, 0)
        c = tl.load(COS + row * CS0 + safe_local * CS1,
                    mask=rotary, other=1.0).to(tl.float32)
        s = tl.load(SIN + row * SS0 + safe_local * SS1,
                    mask=rotary, other=0.0).to(tl.float32)
        ar = (a * c - b * s).to(tl.bfloat16).to(tl.float32)
        br = (b * c + a * s).to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(ar), 0), tl.max(tl.abs(br), 0))
        scale = amax / 127.0
        # Quantize with FP32 scale; only round the stored scale to FP16.
        divisor = tl.where(amax > 0.0, scale, 1.0)
        qa = tl.minimum(tl.maximum(_round_even(ar / divisor), -127.0), 127.0)
        qb = tl.minimum(tl.maximum(_round_even(br / divisor), -127.0), 127.0)
        base = KC + block * K0 + slot * K1
        tl.store(base + even * K2, qa.to(tl.int8))
        tl.store(base + odd * K2, qb.to(tl.int8))
        tl.store(SC + block * S0 + slot * S1,scale.to(tl.float16))

def indexer_k_norm_rope_int8_store(k_pre, weight, eps, cos, sin,
                                   slots, k_cache, scale_cache):
    """No host tensor reads, allocation, cache copies, or synchronization."""
    if k_pre.ndim != 2 or k_pre.dtype != torch.bfloat16:
        raise ValueError("k_pre must be [T,D] BF16")
    t, d = k_pre.shape
    if d < 2 or d & (d - 1) or k_pre.stride(1) != 1:
        raise ValueError("D must be an even power of two with stride(1)=1")
    if weight.shape != (d,) or weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("weight must be BF16/FP32 [D]")
    if not 0 < eps < float("inf"):
        raise ValueError("Require finite positive model RMSNorm epsilon")
    if slots.shape != (t, 2) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("slots must be INT32/INT64 [T,2]")
    if cos.ndim == 4 and cos.shape[1:3] == (1, 1):
        cos = cos[:, 0, 0, :]
    if sin.ndim == 4 and sin.shape[1:3] == (1, 1):
        sin = sin[:, 0, 0, :]
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != t:
        raise ValueError("cos/sin must be [T,R] or [T,1,1,R]")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("Require source-selected FP32 RoPE tables")
    r = cos.shape[1]
    if not (0 < r <= d and r % 2 == 0):
        raise ValueError("R must be positive, even, and <= D")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, d):
        raise ValueError("k_cache must be [blocks,page,1,D]")
    nb, page = k_cache.shape[:2]
    if scale_cache.shape != (nb, page, 1, 1):
        raise ValueError("scale_cache must be [blocks,page,1,1]")
    if k_cache.dtype != torch.int8 or scale_cache.dtype != torch.float16:
        raise ValueError("Require INT8 K and FP16 dequant scales")
    # Permit gaps between pages, e.g. views in a shared multi-layer allocation.
    if (k_cache.stride(3) != 1 or k_cache.stride(1) < d
            or k_cache.stride(0) < page * k_cache.stride(1)
            or scale_cache.stride(1) < 1
            or scale_cache.stride(0) < page * scale_cache.stride(1)):
        raise ValueError("Require internally contiguous, nonoverlapping cache rows")
    tensors = (weight, cos, sin, slots, k_cache, scale_cache)
    if k_pre.device.type != "npu" or any(x.device != k_pre.device for x in tensors):
        raise ValueError("All tensors must share one NPU")
    if t == 0:
        return
    _indexer_k_norm_rope_int8_store[(t,)](
        k_pre, weight, cos, sin, slots, k_cache, scale_cache,
        k_pre.stride(0), weight.stride(0),
        *cos.stride(), *sin.stride(), *slots.stride(),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(3),
        scale_cache.stride(0), scale_cache.stride(1),
        nb, page, d, r, eps,
        enable_fp_fusion=False,
    )
