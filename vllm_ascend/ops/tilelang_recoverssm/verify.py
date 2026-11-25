# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch interface for A5 RecoverSSM Verify. Offsets and page IDs are device-side."""

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=512)
def _layout_from_metadata(shape, strides, dtype):
    if any(s < 0 for s in strides):
        raise ValueError("negative strides are unsupported")
    span = 0 if 0 in shape else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    return span, strides, str(dtype).removeprefix("torch.")


def layout(t):
    # Cache immutable metadata only. Base address and storage offset remain per-call.
    return _layout_from_metadata(t.shape, t.stride(), t.dtype)


def prepare(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    checkpoint_state,
    replay_cache,
    query_start_loc,
    state_indices,
    spec_query_len,
    scale=None,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    use_qk_l2norm_in_kernel=True,
    out=None,
    tile_v=8,
    block_cap=64,
    shaped_qk=False,
):
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128 or k.shape != q.shape:
        raise ValueError("Q/K must have shape [1,T,HK,128]")
    if v.ndim != 4 or v.shape[:2] != q.shape[:2] or v.shape[-1] != 128:
        raise ValueError("V must have shape [1,T,HV,128]")
    _, tokens, hk, _ = q.shape
    hv = v.shape[2]
    pages = checkpoint_state.shape[0]
    if not (hk and hv and hv % hk == 0 and hv // hk <= 16 and 1 <= spec_query_len <= 32):
        raise ValueError("invalid GQA ratio or speculative window")
    if checkpoint_state.shape != (pages, hv, 128, 128) or replay_cache.shape != (
        pages,
        hv,
        spec_query_len,
        257,
    ):
        raise ValueError("invalid checkpoint or replay shape")
    if query_start_loc.shape != (state_indices.numel() + 1,) or state_indices.ndim != 1:
        raise ValueError("expected one page ID per request and B+1 offsets")
    if tokens > state_indices.numel() * spec_query_len or tile_v not in (4, 8, 16) or block_cap < 0:
        raise ValueError("invalid window capacity or tile_v")
    if a.shape not in ((tokens, hv), (1, tokens, hv)) or b.shape not in ((tokens, hv), (1, tokens, hv)):
        raise ValueError("invalid gate shape")
    if A_log.shape != (hv,) or dt_bias.shape != (hv,):
        raise ValueError("invalid gate parameter shape")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, a, b)):
        raise TypeError("BF16 Q/K/V and gate activations required")
    if (
        checkpoint_state.dtype not in (torch.bfloat16, torch.float32)
        or replay_cache.dtype != torch.float32
        or any(t.dtype not in (torch.bfloat16, torch.float32) for t in (A_log, dt_bias))
    ):
        raise TypeError("invalid checkpoint, replay or gate parameter dtype")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (query_start_loc, state_indices)):
        raise TypeError("metadata must use INT32 or INT64")
    scale = 128**-0.5 if scale is None else float(scale)
    if softplus_beta <= 0 or not all(map(math.isfinite, (softplus_beta, softplus_threshold, scale))):
        raise ValueError("invalid softplus or scale")
    if out is None:
        out = torch.empty_like(v, memory_format=torch.contiguous_format)
    if out.shape != v.shape or out.dtype != v.dtype:
        raise ValueError("invalid output buffer")
    args = (
        A_log,
        a.view(tokens, hv),
        b.view(tokens, hv),
        dt_bias,
        q[0],
        k[0],
        v[0],
        checkpoint_state,
        replay_cache,
        query_start_loc,
        state_indices,
        out[0],
    )
    if any(t.device != q.device for t in args):
        raise ValueError("all inputs must share a device")
    desc = tuple(layout(t) for t in args)
    config = (
        desc,
        state_indices.numel(),
        hk,
        hv,
        pages,
        spec_query_len,
        tile_v,
        block_cap,
        bool(use_qk_l2norm_in_kernel),
        scale,
        float(softplus_beta),
        float(softplus_threshold),
    )
    flat = tuple(
        t if ((shaped_qk and i in (4, 5)) or (t.ndim == 1 and s[1] == (1,))) else t.as_strided((s[0],), (1,))
        for i, (t, s) in enumerate(zip(args, desc))
    )
    return out, config, flat


def gdn_recoverssm_verify(*args, **kwargs):
    out, config, inputs = prepare(*args, **kwargs)
    if out.device.type != "npu":
        raise ValueError("TileLang Verify requires Ascend NPU tensors")
    if out.numel():
        from .kernel import compiled

        compiled(*config)(*inputs)
    return out
