# SPDX-License-Identifier: Apache-2.0
"""GPU scheduling of packed paged MLA for the A3 CATLASS kernel."""

import math
import struct

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _find_batch(t, CU, PLAN, B: tl.constexpr, T: tl.constexpr, PACKED: tl.constexpr, HAS_PLAN: tl.constexpr):
    batch = tl.minimum(t // triton.cdiv(T, B), B - 1)
    if PACKED and B > 1:
        if HAS_PLAN:
            lo = tl.load(PLAN + 4 * batch)
            hi = tl.load(PLAN + 4 * batch + 1)
        else:
            lo = tl.load(CU + batch)
            hi = tl.load(CU + batch + 1)
        if (t < lo) | (t >= hi):
            b = tl.arange(0, triton.next_power_of_2(B))
            if HAS_PLAN:
                ends = tl.load(PLAN + b * 4 + 1, b < B, other=T)
            else:
                ends = tl.load(CU + b + 1, b < B, other=T)
            batch = tl.minimum(B - 1, tl.sum((ends <= t).to(tl.int32), 0))
    return batch


@triton.jit
def _header(
    TILING,
    B,
    H: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    SCALE_BITS: tl.constexpr,
    CAUSAL: tl.constexpr,
    SPAN: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_T: tl.constexpr = 0,
    DIRECT: tl.constexpr = False,
):
    w = tl.arange(0, 32)
    data = tl.where((w == 0) | (w == 18), B, 0)
    data += tl.where((w == 1) | (w == 10), H, 0)
    data += tl.where(w == 2, 512, 0) + tl.where(w == 3, P, 0)
    data += tl.where(w == 4, 128, 0) + tl.where(w == 5, M, 0)
    data += tl.where(w == 6, SCALE_BITS, 0)
    data += ((w == 7) | (w == 11) | (w == 17)).to(tl.int32)
    data += tl.where((w == 8) | (w == 9), 32, 0)
    if CAUSAL and H == 16:
        data += tl.where(w == 12, 3, 0)
    data += tl.where(w == 13, 64, 0) + tl.where(w == 14, M * 128, 0)
    data += tl.where(w == 15, SPAN, 0) + tl.where(w == 16, SPLITS, 0)
    data += tl.where(w == 21, 576, 0)
    data += tl.where(w == 22, OUT_T, 0) + tl.where((w == 23) & DIRECT, 1, 0)
    tl.store(TILING + w, data)


@triton.jit
def _record(TILING, v, kv, H: tl.constexpr, SPLITS: tl.constexpr):
    w = tl.arange(0, 32)
    base = v.to(tl.int64) * H
    latent = base * 512
    rope = base * 64
    reduction = base * SPLITS
    data = tl.where(w == 0, 1, 0) + tl.where(w == 1, kv, 0) + tl.where(w == 3, 128, 0)
    data += tl.where((w == 4) | (w == 13), (latent >> 32).to(tl.int32), 0)
    data += tl.where((w == 5) | (w == 14), latent.to(tl.int32), 0)
    data += tl.where(w == 6, (rope >> 32).to(tl.int32), 0) + tl.where(w == 7, rope.to(tl.int32), 0)
    data += tl.where(w == 11, (reduction >> 32).to(tl.int32), 0)
    data += tl.where(w == 12, reduction.to(tl.int32), 0)
    tl.store(TILING + (v + 1) * 32 + w, data)


@triton.jit
def _prepare(
    Q,
    TABLE,
    LEN,
    CU,
    USED,
    PLAN,
    QPACK,
    VTAB,
    TILING,
    ROWS,
    B: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    PACKED: tl.constexpr,
    HAS_USED: tl.constexpr,
    HAS_PLAN: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE_BITS: tl.constexpr,
    SPAN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    v = tl.program_id(0)
    if v == 0:
        _header(TILING, G, H, M, P, SCALE_BITS, CAUSAL, SPAN, SPLITS)
    if H == 4:
        batch = _find_batch(v, CU, PLAN, B, T, PACKED, HAS_PLAN)
    else:
        batch = tl.minimum(v, B - 1)
        if G > B:
            if v >= B:
                batch = _find_batch((v - B) * 4, CU, PLAN, B, T, PACKED, HAS_PLAN)
    if HAS_PLAN:
        rec = PLAN + 4 * batch
        lo = tl.load(rec) if PACKED else batch * (T // B)
        hi = tl.load(rec + 1) if PACKED else lo + T // B
        used = tl.load(rec + 2)
        kv = tl.load(rec + 3)
        valid_layout = (tl.load(PLAN + 4 * B) == 0x4D4C4133) & (tl.load(PLAN + 4 * B + 1) == PACKED)
    else:
        lo = tl.load(CU + batch) if PACKED else batch * (T // B)
        hi = tl.load(CU + batch + 1) if PACKED else lo + T // B
        used = tl.load(USED + batch) if HAS_USED else hi - lo
        kv = tl.load(LEN + batch)
        valid_layout = True
    if PACKED:
        if HAS_PLAN:
            valid_layout = valid_layout & (tl.load(PLAN) == 0) & (tl.load(PLAN + 4 * B - 3) == T)
        else:
            valid_layout = valid_layout & (tl.load(CU) == 0) & (tl.load(CU + B) == T)
        # The four-position route has several programs publishing disjoint
        # output maps. Validate that the schedule actually partitions [0,T).
        if H == 16 and B > 1:
            i = tl.arange(0, triton.next_power_of_2(B))
            if HAS_PLAN:
                begin = tl.load(PLAN + i * 4, i < B, other=0)
                end = tl.load(PLAN + i * 4 + 1, i < B, other=0)
                next_begin = tl.load(PLAN + (i + 1) * 4, i + 1 < B, other=T)
                invalid = (i < B) & ((begin > end) | (end != next_begin))
            else:
                begin = tl.load(CU + i, i < B, other=0)
                end = tl.load(CU + i + 1, i < B, other=0)
                invalid = (i < B) & (begin > end)
            valid_layout = valid_layout & (tl.sum(invalid.to(tl.int32), 0) == 0)
    start = tl.maximum(0, tl.minimum(T, lo))
    stop = tl.maximum(0, tl.minimum(T, hi))
    valid = valid_layout & (lo == start) & (hi == stop) & (used >= 0) & (used <= stop - start)
    valid = valid & (kv >= 0) & (kv <= M * 128)
    # All native reads only visit valid pages. A bad page invalidates precisely
    # those query rows whose attention interval would include it.
    p = tl.arange(0, triton.next_power_of_2(M))
    pages = tl.load(TABLE + batch.to(tl.int64) * M + p, p < M, other=0)
    bad = (pages < 0) | (pages >= P)
    first_bad = tl.min(tl.where((p < M) & bad, p * 128, M * 128), 0)
    tl.store(VTAB + v.to(tl.int64) * M + p, tl.where(bad, 0, pages), p < M)
    if H == 4:
        local = v - start
        end = tl.minimum(kv, kv - used + local + 1) if CAUSAL else kv
        active = valid & (local >= 0) & (local < used) & (end > 0) & (end <= first_bad)
        _record(TILING, v, tl.where(active, end, 0), H, SPLITS)
        tl.store(ROWS + v, tl.where(active, v * 4, -1))
    else:
        local = 0
        launch = v < B
        if G > B:
            if v >= B:
                anchor = (v - B) * 4
                local = tl.maximum(1, tl.cdiv(anchor - start, 4)) * 4
                launch = (start + local >= anchor) & (start + local < anchor + 4) & (start + local < stop)
        first = start + local
        count = tl.maximum(0, tl.minimum(4, used - local))
        if CAUSAL:
            # Native causal mode aligns four rows to the right of record_kv.
            # Right-align partially occupied groups to retain the original causal diagonal.
            base = kv - used + local
            count = tl.minimum(count, tl.maximum(0, first_bad - base))
            record_kv = tl.maximum(0, base + count)
        else:
            count = tl.where(first_bad < kv, 0, count)
            record_kv = kv
        count = tl.where(valid & launch, count, 0)
        shift = 4 - count
        _record(TILING, v, tl.where(count > 0, record_kv, 0), H, SPLITS)
        row = tl.arange(0, 16)
        if count > 0:
            d = tl.arange(0, 512)
            r = tl.arange(0, 64)
            j = row // 4 - shift
            # The A3 vector backend can issue the aligned DMA for a masked
            # lane. Keep its source inside allocated storage for partial groups.
            token = tl.minimum(T - 1, tl.maximum(0, first + j))
            src = (token.to(tl.int64) * 4 + row % 4) * 576
            dst = (v.to(tl.int64) * 16 + row) * 576
            q0 = tl.load(Q + src[:, None] + d[None, :])
            qr = tl.load(Q + src[:, None] + 512 + r[None, :])
            tl.store(QPACK + dst[:, None] + d[None, :], q0)
            tl.store(QPACK + dst[:, None] + 512 + r[None, :], qr)
        idx = tl.arange(0, 4)
        mapped = idx < count
        if CAUSAL:
            mapped = mapped & (base + idx + 1 > 0)
        else:
            mapped = mapped & (kv > 0)
        # Even malformed global endpoints must initialize all output row maps.
        store_t = tl.where(valid_layout, first + idx, v * 4 + idx)
        store_ok = tl.where(valid_layout, launch & (first + idx < stop), v * 4 + idx < T)
        if tl.sum(store_ok.to(tl.int32), 0) > 0:
            tl.store(ROWS + store_t, tl.where(mapped, v * 16 + (idx + shift) * 4, -1), store_ok)


@triton.jit
def _unpack(OUT, ROWS, Y, T: tl.constexpr, NTD: tl.constexpr, GROUPED: tl.constexpr):
    pid = tl.program_id(0)
    head = tl.arange(0, 4)
    d = tl.arange(0, 512)
    if GROUPED:
        pos = tl.arange(0, 4)
        t = pid * 4 + pos
        off = tl.load(ROWS + t, t < T, other=-1)
        src = tl.maximum(off, 0).to(tl.int64)[:, None] + head[None, :]
        value = tl.load(OUT + src[:, :, None] * 512 + d[None, None, :], (off >= 0)[:, None, None], other=0.0)
        if NTD:
            value = tl.trans(value, (1, 0, 2))
            dst = head[:, None] * T + t[None, :]
            tl.store(Y + dst.to(tl.int64)[:, :, None] * 512 + d[None, None, :], value, (t < T)[None, :, None])
        else:
            dst = t[:, None] * 4 + head[None, :]
            tl.store(Y + dst.to(tl.int64)[:, :, None] * 512 + d[None, None, :], value, (t < T)[:, None, None])
    else:
        off = tl.load(ROWS + pid)
        src = (tl.maximum(off, 0).to(tl.int64) + head) * 512
        value = tl.load(OUT + src[:, None] + d[None, :], off >= 0, other=0.0)
        dst = head * T + pid if NTD else pid * 4 + head
        tl.store(Y + dst.to(tl.int64)[:, None] * 512 + d[None, :], value)


def catlass_decode(q, cache, table, lengths, cu, used, plan, scale, causal, ntd, *, split_hint=None):
    """Run dense BF16 short-query MLA after the public operator has normalized inputs."""
    b, width = table.shape
    t, heads, dim = q.shape
    if not (
        0 < t <= b * 4
        and heads == 4
        and dim == 576
        and q.dtype == torch.bfloat16
        and cache.shape[1:] == (128, 576)
        and 0 < width <= 1024
    ):
        raise ValueError("CATLASS short decode requires 4 BF16 heads, 128-token pages and mean query length <= 4")
    packed = cu is not None
    if not packed and t % b:
        raise ValueError("dense query count must be divisible by batch size")
    h = 4 if t <= b else 16
    g = t if h == 4 else b
    if h == 16 and packed and t > 4 and b > 1:
        g += triton.cdiv(t, 4)
    if g >= 65536:
        raise ValueError("CATLASS supports fewer than 65536 virtual query groups")
    ops = torch.ops.flash_mla_a3
    cores = ops.catlass_cores(q)
    target = split_hint if split_hint is not None else cores // math.gcd(cores, b)
    if not 1 <= target <= 64:
        raise ValueError("the split target must be in [1,64]")
    span_pages = triton.cdiv(width, min(target, width))
    splits = triton.cdiv(width, span_pages)
    if h == 16 and t >= 4 and splits >= 2:
        from .catlass_direct import direct_decode

        return direct_decode(q, cache, table, lengths, cu, used, plan, scale, causal, ntd, span_pages * 128, splits)
    qi = q.view(t, 4, 576) if h == 4 else q.new_empty((g, h, 576))
    ti = table.new_empty((g, width))
    tiling = table.new_empty((g + 1) * 32)
    rows = table.new_empty(t)
    _prepare[(g,)](
        q,
        table,
        lengths,
        cu if packed else lengths,
        used if used is not None else lengths,
        plan if plan is not None else lengths,
        qi,
        ti,
        tiling,
        rows,
        b,
        t,
        g,
        h,
        width,
        cache.size(0),
        packed,
        used is not None,
        plan is not None,
        causal,
        struct.unpack("i", struct.pack("f", scale))[0],
        span_pages * 128,
        splits,
        num_stages=1,
    )
    out = ops.catlass_run(qi, cache, ti, tiling, splits, h)
    y = q.new_empty((4, t, 512) if ntd else (t, 4, 512))
    grouped = t >= 32
    _unpack[(triton.cdiv(t, 4) if grouped else t,)](out, rows, y, t, ntd, grouped, num_stages=1)
    return y
