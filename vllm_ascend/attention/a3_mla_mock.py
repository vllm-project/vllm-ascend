# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical development mock of the PR #338 compressed MLA API.

This is not an A3 performance kernel. Its metadata is private to this mock;
the framework must treat both mock and native metadata as opaque tensors.
"""

import math

import torch

ROPE_DIM = 64
LATENT_DIM = 512
MAX_CAPACITY = 16384
MAX_MOCK_ELEMENTS = 16 * 1024 * 1024
MIN_PAGE_SIZE = 16
MAX_PAGE_SIZE = 1024


def _assert_device(condition: torch.Tensor, message: str) -> None:
    # Do not turn a device condition into a Python bool / item().
    torch._assert_async(condition.all(), message)


def _lengths(cache_seqlens, cu_seqlens_q, batch_size):
    if cache_seqlens.shape != (batch_size,) or cache_seqlens.dtype != torch.int32:
        raise ValueError("cache_seqlens must be int32 [batch_size]")
    if cu_seqlens_q is None or cu_seqlens_q.shape != (batch_size + 1,) or cu_seqlens_q.dtype != torch.int32:
        raise ValueError("cu_seqlens_q must be int32 [batch_size + 1]")
    if cache_seqlens.device != cu_seqlens_q.device:
        raise ValueError("lengths must share one device")
    if not cache_seqlens.is_contiguous() or not cu_seqlens_q.is_contiguous():
        raise ValueError("lengths must be contiguous")
    _assert_device(cu_seqlens_q[:1] == 0, "query prefixes must start at zero")
    _assert_device(cu_seqlens_q[1:] >= cu_seqlens_q[:-1], "query prefixes must be nondecreasing")


def get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    headdim,
    cache_seqlens,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    cache_leftpad=None,
    page_size=None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    has_softcap=False,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
):
    """Match the published API; create records without reading device lengths."""
    if (headdim, headdim_v, num_heads_kv) != (ROPE_DIM, LATENT_DIM, 1) or num_heads_q <= 0:
        raise ValueError("mock supports compressed MLA 64/512 with one KV head")
    if qkv_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("mock supports FP16/BF16 only")
    if page_size is None or not MIN_PAGE_SIZE <= page_size <= MAX_PAGE_SIZE or page_size % MIN_PAGE_SIZE:
        raise ValueError("page_size must be a multiple of 16 in [16, 1024]")
    if not 0 < max_seqlen_k <= MAX_CAPACITY or max_seqlen_k % page_size or max_seqlen_q < 0:
        raise ValueError("invalid static query/KV capacity")
    if (
        cu_seqlens_k_new is not None
        or cache_leftpad is not None
        or max_seqlen_k_new
        or window_size != (-1, -1)
        or attention_chunk
        or has_softcap
        or num_splits not in (0, 1)
        or pack_gqa is not None
        or sm_margin
    ):
        raise NotImplementedError("mock does not implement append/window/softcap/split tuning")
    _lengths(cache_seqlens, cu_seqlens_q, batch_size)
    _assert_device((cache_seqlens >= 0) & (cache_seqlens <= max_seqlen_k), "KV length exceeds capacity")
    _assert_device(cu_seqlens_q[1:] - cu_seqlens_q[:-1] <= max_seqlen_q, "query length exceeds capacity")
    # Shape/static configuration and device contents are bound to this record.
    # new_full creates device constants; no CPU tensor containing lengths exists.
    attrs = (max_seqlen_q, max_seqlen_k, num_heads_q, page_size, int(causal), int(qkv_dtype == torch.bfloat16))
    header = [cache_seqlens.new_full((1,), value) for value in attrs]
    return torch.cat((*header, cache_seqlens, cu_seqlens_q))


@torch.no_grad()
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    rotary_seqlens=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
):
    """Dense tensor reference with the same split-Q, read-only cache contract."""
    del rotary_interleaved
    if any(
        value is not None
        for value in (
            k,
            v,
            rotary_cos,
            rotary_sin,
            cache_batch_idx,
            cache_leftpad,
            cu_seqlens_k_new,
            rotary_seqlens,
            q_descale,
            k_descale,
            v_descale,
        )
    ):
        raise NotImplementedError("mock supports read-only, unquantized paged MLA")
    if qv is None or page_table is None or cache_seqlens is None or max_seqlen_q is None:
        raise ValueError("qv, page_table, cache_seqlens and max_seqlen_q are required")
    if q.ndim != 3 or q.shape[-1] != ROPE_DIM or qv.shape != (*q.shape[:2], LATENT_DIM):
        raise ValueError("expected TND q(64), qv(512)")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, ROPE_DIM):
        raise ValueError("expected paged k_cache [pages, page_size, 1, 64]")
    if v_cache.shape != (*k_cache.shape[:3], LATENT_DIM):
        raise ValueError("expected matching paged v_cache with width 512")
    if page_table.ndim != 2 or page_table.dtype != torch.int32:
        raise ValueError("page_table must be int32 [batch, pages_per_request]")
    if any(t.device != q.device for t in (qv, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q)):
        raise ValueError("all inputs must share one device")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(t.dtype != q.dtype for t in (qv, k_cache, v_cache)):
        raise ValueError("Q and KV must share FP16/BF16 dtype")
    if any(t.requires_grad for t in (q, qv, k_cache, v_cache)):
        raise ValueError("MLA inference does not accept gradients")
    if softcap != 0:
        raise NotImplementedError("softcap is not implemented by the mock")
    scale = (ROPE_DIM + LATENT_DIM) ** -0.5 if softmax_scale is None else softmax_scale
    if not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("softmax_scale must be a finite host scalar")
    batch, columns = page_table.shape
    page_size = k_cache.shape[1]
    capacity = columns * page_size
    expected = get_scheduler_metadata(
        batch,
        max_seqlen_q,
        capacity,
        q.shape[1],
        1,
        ROPE_DIM,
        cache_seqlens,
        qkv_dtype=q.dtype,
        headdim_v=LATENT_DIM,
        cu_seqlens_q=cu_seqlens_q,
        page_size=page_size,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    if scheduler_metadata is not None:
        if scheduler_metadata.shape != expected.shape or scheduler_metadata.dtype != torch.int32:
            raise ValueError("invalid scheduler metadata shape or dtype")
        if scheduler_metadata.device != q.device:
            raise ValueError("scheduler metadata must share the input device")
        _assert_device(scheduler_metadata == expected, "stale or incompatible scheduler metadata")
    tokens, heads = q.shape[:2]
    _assert_device(cu_seqlens_q[-1:] == tokens, "query prefixes must end at T")
    if tokens * capacity * LATENT_DIM > MAX_MOCK_ELEMENTS:
        raise RuntimeError("mock workload limit exceeded; use small correctness cases or a native provider")
    used_pages = torch.arange(columns, device=q.device)[None, :] * page_size < cache_seqlens[:, None]
    valid_pages = (page_table >= 0) & (page_table < k_cache.shape[0])
    _assert_device(~used_pages | valid_pages, "used page index outside physical cache")
    if not tokens or not k_cache.shape[0]:
        out = q.new_zeros((tokens, heads, LATENT_DIM))
        lse = torch.full((heads, tokens), -torch.inf, device=q.device, dtype=torch.float32)
    else:
        # Use static capacity, masking invalid/unused entries on the device.
        token = torch.arange(tokens, device=q.device, dtype=torch.int32)
        requests = torch.searchsorted(cu_seqlens_q[1:].contiguous(), token, right=True).long()
        offset = torch.arange(capacity, device=q.device)
        pages = page_table[:, offset // page_size].long().clamp(0, k_cache.shape[0] - 1)
        keys = k_cache[pages, offset % page_size, 0][requests].float()
        values = v_cache[pages, offset % page_size, 0][requests].float()
        scores = (
            torch.einsum("thd,tkd->thk", q.float(), keys) + torch.einsum("thd,tkd->thk", qv.float(), values)
        ) * scale
        lengths = cache_seqlens[requests]
        mask = offset[None, :] < lengths[:, None]
        # Unused physical cache may contain NaNs/uninitialized values. Zero it
        # before the value reduction: IEEE 0 * NaN must not pollute valid rows.
        values = values.masked_fill(~mask[:, :, None], 0)
        if causal:
            allocated_q = cu_seqlens_q[requests + 1] - cu_seqlens_q[requests]
            local_q = token - cu_seqlens_q[requests]
            mask &= offset[None, :] <= (lengths - allocated_q + local_q)[:, None]
        mask = mask[:, None, :]
        scores = scores.masked_fill(~mask, -torch.inf)
        live = mask.any(dim=-1)
        maxima = scores.amax(dim=-1)
        safe_max = torch.where(live, maxima, torch.zeros_like(maxima))
        weights = torch.exp(scores - safe_max.unsqueeze(-1))
        denom = weights.sum(dim=-1)
        probs = weights / denom.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
        out = torch.einsum("thk,tkd->thd", probs, values).to(q.dtype)
        lse = torch.where(live, safe_max + denom.log(), torch.full_like(safe_max, -torch.inf)).T.contiguous()
    if return_softmax_lse:
        empty = torch.empty((0,), device=q.device, dtype=torch.float32)
        return out, lse, empty, empty.clone()
    return out
