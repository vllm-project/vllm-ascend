"""A3 Cube paged MLA for small BF16 decode batches with four local heads."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _attention_block(
    Q,
    CACHE,
    TABLE,
    OUT,
    PART,
    STATS,
    batch,
    split,
    start,
    used,
    kv_len,
    query_offset,
    live,
    SCALE: tl.constexpr,
    TOKENS: tl.constexpr,
    PAGES: tl.constexpr,
    WIDTH: tl.constexpr,
    NS: tl.constexpr,
    BN: tl.constexpr,
    CAUSAL: tl.constexpr,
    NTD: tl.constexpr,
):
    # 4 MTP positions x 4 local query heads share one cache tile on the Cube.
    row = tl.arange(0, 16)
    col = tl.arange(0, BN)
    latent_dim = tl.arange(0, 512)
    rotary_dim = tl.arange(0, 64)
    local = query_offset + row // 4
    first = tl.maximum(start, 0) + query_offset
    remaining = tl.minimum(live - query_offset, TOKENS - first)
    store = row < remaining * 4
    # Keep token/head flattening affine for GM access. Expressing this address
    # as 4*(row//4) + row%4 can lower to scalar scattered stores on A3.
    q_base = Q + first.to(tl.int64) * 4 * 576
    q = tl.load(q_base + row[:, None] * 576 + latent_dim[None, :], store[:, None], other=0.0)
    q_rope = tl.load(q_base + row[:, None] * 576 + 512 + rotary_dim[None, :], store[:, None], other=0.0)
    end = tl.full((16,), 0, tl.int32) + kv_len
    if CAUSAL:
        end = tl.minimum(end, kv_len - used + local + 1)
    end = tl.where((local < used) & (kv_len >= 0) & (kv_len <= WIDTH * 128), end, 0)
    tile_idx = split
    first_pos = tile_idx * BN
    logical = tile_idx // (128 // BN)
    page = tl.load(TABLE + batch * WIDTH + logical, logical < WIDTH, other=-1)
    page_valid = (page >= 0) & (page < PAGES)
    kv_base = CACHE + (tl.where(page_valid, page, 0).to(tl.int64) * 128 + tile_idx % (128 // BN) * BN) * 576
    # Mask while loading so unwritten padded V slots cannot inject NaNs.
    valid_key = page_valid & (first_pos + col < kv_len)
    k = tl.load(kv_base + col[:, None] * 576 + latent_dim[None, :], valid_key[:, None], other=0.0)
    k_rope = tl.load(kv_base + col[:, None] * 576 + 512 + rotary_dim[None, :], valid_key[:, None], other=0.0)
    score = (tl.dot(q, tl.trans(k)) + tl.dot(q_rope, tl.trans(k_rope))) * SCALE
    score = tl.where((first_pos + col[None, :] < end[:, None]) & page_valid, score, -float("inf"))
    maximum = tl.max(score, axis=1)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(score - safe[:, None])
    total = tl.sum(weights, axis=1)
    numerator = tl.dot(weights.to(tl.bfloat16), k)
    if page < 0 or page >= PAGES:
        total = tl.where(end > first_pos, -1.0, total)
    rows = TOKENS * 4
    stride = tl.cdiv(rows, 8) * 8
    packed_row = first.to(tl.int64) * 4 + row
    metric = split.to(tl.int64) * 2 * stride + packed_row
    tl.store(STATS + metric, maximum, store)
    tl.store(STATS + metric + stride, total, store)
    slot = split.to(tl.int64) * rows + packed_row
    tl.store(PART + slot[:, None] * 512 + latent_dim[None, :], numerator, store[:, None])


@triton.jit
def _paged(
    Q,
    CACHE,
    TABLE,
    LENGTHS,
    CU,
    USED,
    PLAN,
    OUT,
    PART,
    STATS,
    SCALE: tl.constexpr,
    TOKENS: tl.constexpr,
    B: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGES: tl.constexpr,
    NS: tl.constexpr,
    BN: tl.constexpr,
    PACKED: tl.constexpr,
    HAS_PLAN: tl.constexpr,
    HAS_USED: tl.constexpr,
    CAUSAL: tl.constexpr,
    NTD: tl.constexpr,
):
    split = tl.program_id(0)
    b = tl.program_id(1)
    if HAS_PLAN:
        record = PLAN + b * 4
        used = tl.load(record + 2)
        length = tl.load(record + 3)
    else:
        length = tl.load(LENGTHS + b)
    if PACKED:
        if HAS_PLAN:
            start = tl.load(record)
            end = tl.load(record + 1)
        else:
            start = tl.load(CU + b)
            end = tl.load(CU + b + 1)
        physical = tl.maximum(0, tl.minimum(TOKENS, end) - tl.maximum(0, start))
    else:
        start = b * (TOKENS // B)
        physical = TOKENS // B
    if not HAS_PLAN:
        if HAS_USED:
            used = tl.load(USED + b)
        else:
            used = physical
    used = tl.where((used >= 0) & (used <= physical), used, 0)
    _attention_block(
        Q,
        CACHE,
        TABLE,
        OUT,
        PART,
        STATS,
        b,
        split,
        start,
        used,
        length,
        0,
        physical,
        SCALE,
        TOKENS,
        PAGES,
        WIDTH,
        NS,
        BN,
        CAUSAL,
        NTD,
    )
    # Most decode batches fit the first 4 query positions. A skewed TND batch
    # can have more in a single sequence even when T / B is small.
    if PACKED and TOKENS > 4:
        for group in range(1, tl.cdiv(physical, 4)):
            _attention_block(
                Q,
                CACHE,
                TABLE,
                OUT,
                PART,
                STATS,
                b,
                split,
                start,
                used,
                length,
                group * 4,
                physical,
                SCALE,
                TOKENS,
                PAGES,
                WIDTH,
                NS,
                BN,
                CAUSAL,
                NTD,
            )


@triton.jit
def _combine(
    PART,
    STATS,
    OUT,
    NEXT_STATS,
    TOKENS: tl.constexpr,
    NS: tl.constexpr,
    BS: tl.constexpr,
    NTD: tl.constexpr,
    BD: tl.constexpr,
    FINAL: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    subdim = tl.program_id(2)
    split = group * BS + tl.arange(0, BS)
    dim = subdim * BD + tl.arange(0, BD)
    rows = TOKENS * 4
    stride = tl.cdiv(rows, 8) * 8
    metric = split.to(tl.int64) * 2 * stride + row
    maximum = tl.load(STATS + metric, split < NS, other=-float("inf"))
    total = tl.load(STATS + metric + stride, split < NS, other=0.0)
    m = tl.max(maximum, 0)
    factor = tl.exp(maximum - tl.where(m == -float("inf"), 0.0, m))
    denom = tl.sum(factor * total, axis=0)
    slot = split.to(tl.int64) * rows + row
    partial = tl.load(PART + slot[:, None] * 512 + dim[None, :], split[:, None] < NS, other=0.0)
    num = tl.sum(partial * factor[:, None], axis=0)
    invalid = tl.min(total, 0) < 0.0
    if FINAL:
        dst = row % 4 * TOKENS + row // 4 if NTD else row
        valid = (denom > 0.0) & (~invalid)
        tl.store(OUT + dst.to(tl.int64) * 512 + dim, tl.where(valid, num / tl.where(denom > 0.0, denom, 1.0), 0.0))
    else:
        dst = group.to(tl.int64) * rows + row
        tl.store(OUT + dst * 512 + dim, num)
        if subdim == 0:
            # These adjacent scalar rows have separate producers; use one
            # aligned DMA block per result and read its two stats in _finish.
            lanes = tl.arange(0, 8)
            block = tl.where(lanes == 0, m, tl.where(lanes == 1, tl.where(invalid, -1.0, denom), 0.0))
            tl.store(NEXT_STATS + dst * 8 + lanes, block)


@triton.jit
def _finish(PART, STATS, OUT, TOKENS: tl.constexpr, GROUPS: tl.constexpr, BS: tl.constexpr, NTD: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.program_id(1) * 128 + tl.arange(0, 128)
    g = tl.arange(0, BS)
    slot = g.to(tl.int64) * (TOKENS * 4) + row
    m = tl.load(STATS + slot * 8, g < GROUPS, other=-float("inf"))
    den = tl.load(STATS + slot * 8 + 1, g < GROUPS, other=0.0)
    peak = tl.max(m, axis=0)
    a = tl.exp(m - tl.where(peak == -float("inf"), 0.0, peak))
    denom = tl.sum(den * a, axis=0)
    num = tl.load(PART + slot[:, None] * 512 + dim[None, :], g[:, None] < GROUPS, other=0.0)
    y = tl.sum(num * a[:, None], axis=0) / tl.where(denom > 0.0, denom, 1.0)
    dst = row % 4 * TOKENS + row // 4 if NTD else row
    tl.store(OUT + dst.to(tl.int64) * 512 + dim, tl.where((tl.min(den, axis=0) >= 0.0) & (denom > 0.0), y, 0.0))


def cube_decode(q, cache, table, lengths, cu, used, plan, scale, causal, ntd):
    if table.size(1) == 1:
        from .one_page import batch_one_page_decode

        return batch_one_page_decode(q, cache, table, lengths, cu, used, plan, scale, causal, ntd)

    tokens = q.shape[0]
    batch, width = table.shape
    tile = 64 if width == 1 or (cu is not None and tokens > 4) else 128
    splits = width * (128 // tile)
    rows = tokens * 4
    stride = triton.cdiv(rows, 8) * 8
    out = q.new_empty((4, tokens, 512) if ntd else (tokens, 4, 512))
    partial = q.new_empty((splits, rows, 512), dtype=torch.float32)
    stats = q.new_empty((splits, 2, stride), dtype=torch.float32)
    _paged[(splits, batch)](
        q,
        cache,
        table,
        lengths,
        cu if cu is not None else lengths,
        used if used is not None else lengths,
        plan if plan is not None else lengths,
        out,
        partial,
        stats,
        scale,
        tokens,
        batch,
        width,
        cache.size(0),
        splits,
        tile,
        cu is not None,
        plan is not None,
        used is not None,
        causal,
        ntd,
    )
    if splits <= 32:
        dim = 512 if splits <= 16 else 256
        _combine[(rows, 1, 512 // dim)](
            partial, stats, out, stats, tokens, splits, triton.next_power_of_2(splits), ntd, dim, True
        )
    else:
        span = 64 if splits > 1024 else 32
        groups = triton.cdiv(splits, span)
        mid = q.new_empty((groups, rows, 512), dtype=torch.float32)
        mid_stats = q.new_empty((groups, rows, 8), dtype=torch.float32)
        _combine[(rows, groups, 8)](partial, stats, mid, mid_stats, tokens, splits, span, ntd, 64, False)
        _finish[(rows, 4)](mid, mid_stats, out, tokens, groups, triton.next_power_of_2(groups), ntd)
    return out
