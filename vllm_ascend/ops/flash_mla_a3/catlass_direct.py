# SPDX-License-Identifier: Apache-2.0
"""Device-only scheduling of four-position CATLASS groups over the original query and page table."""

import struct

import torch
from vllm.triton_utils import tl, triton

from .catlass_decode import _find_batch, _header


@triton.jit
def _check_segments(CU, PLAN, B: tl.constexpr, T: tl.constexpr, PACKED: tl.constexpr, HAS_PLAN: tl.constexpr):
    valid = True
    longest = T // B
    if HAS_PLAN:
        valid = (tl.load(PLAN + 4 * B) == 0x4D4C4133) & (tl.load(PLAN + 4 * B + 1) == PACKED)
    if PACKED:
        i = tl.arange(0, triton.next_power_of_2(B))
        if HAS_PLAN:
            starts = tl.load(PLAN + 4 * i, i < B, other=0)
            ends = tl.load(PLAN + 4 * i + 1, i < B, other=0)
            following = tl.load(PLAN + 4 * (i + 1), i + 1 < B, other=T)
            valid = valid & (tl.load(PLAN) == 0)
            bad = (i < B) & ((starts > ends) | (ends != following))
        else:
            starts = tl.load(CU + i, i < B, other=0)
            ends = tl.load(CU + i + 1, i < B, other=0)
            valid = (tl.load(CU) == 0) & (tl.load(CU + B) == T)
            bad = (i < B) & (starts > ends)
        valid = valid & (tl.sum(bad.to(tl.int32), 0) == 0)
        longest = tl.max(ends - starts, 0)
    return valid, longest


@triton.jit
def _write_record(TILING, v, batch, kv, qbase, first_slot, first_good, count, SPLITS: tl.constexpr):
    w = tl.arange(0, 32)
    q_offset = qbase.to(tl.int64) * 4 * 512
    rope_offset = qbase.to(tl.int64) * 4 * 64
    lse_offset = v.to(tl.int64) * 16 * SPLITS
    partial_offset = v.to(tl.int64) * 16 * 512
    rec = (w == 0).to(tl.int32) + tl.where(w == 1, kv, 0) + tl.where(w == 3, 128, 0)
    rec += tl.where(w == 4, (q_offset >> 32).to(tl.int32), 0) + tl.where(w == 5, q_offset.to(tl.int32), 0)
    rec += tl.where(w == 6, (rope_offset >> 32).to(tl.int32), 0) + tl.where(w == 7, rope_offset.to(tl.int32), 0)
    rec += tl.where(w == 11, (lse_offset >> 32).to(tl.int32), 0) + tl.where(w == 12, lse_offset.to(tl.int32), 0)
    rec += tl.where(w == 13, (partial_offset >> 32).to(tl.int32), 0) + tl.where(w == 14, partial_offset.to(tl.int32), 0)
    rec += tl.where(w == 17, batch, 0) + tl.where(w == 18, 4 - count - first_slot, 0)
    rec += tl.where(w == 19, (first_slot + first_good) * 4, 0) + tl.where(w == 20, (first_slot + count) * 4, 0)
    tl.store(TILING + (v + 1) * 32 + w, rec)


@triton.jit
def _group(
    v,
    TABLE,
    LEN,
    CU,
    USED,
    PLAN,
    TILING,
    Y,
    B: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    PACKED: tl.constexpr,
    HAS_PLAN: tl.constexpr,
    HAS_USED: tl.constexpr,
    CAUSAL: tl.constexpr,
    NTD: tl.constexpr,
    SCALE: tl.constexpr,
    SPAN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    if v == 0:
        valid, longest = _check_segments(CU, PLAN, B, T, PACKED, HAS_PLAN)
        groups = tl.where(valid, tl.where(longest > 4, G, B), 0)
        _header(TILING, groups, 16, M, P, SCALE, CAUSAL, SPAN, SPLITS, T if NTD else 0, True)
        if not valid:
            d = tl.arange(0, 2048)
            for t in range(T):
                tl.store(Y + t * 2048 + d, tl.full((2048,), 0, tl.bfloat16))
    batch = tl.minimum(B - 1, v)
    anchor = tl.maximum(0, v - B) * 4
    if PACKED and G > B:
        if v >= B:
            batch = _find_batch(anchor, CU, PLAN, B, T, PACKED, HAS_PLAN)
    if HAS_PLAN:
        lo = tl.load(PLAN + 4 * batch) if PACKED else batch * (T // B)
        hi = tl.load(PLAN + 4 * batch + 1) if PACKED else lo + T // B
        used = tl.load(PLAN + 4 * batch + 2)
        kv = tl.load(PLAN + 4 * batch + 3)
    else:
        lo = tl.load(CU + batch) if PACKED else batch * (T // B)
        hi = tl.load(CU + batch + 1) if PACKED else lo + T // B
        used = tl.load(USED + batch) if HAS_USED else hi - lo
        kv = tl.load(LEN + batch)
    lo = tl.maximum(0, tl.minimum(T, lo))
    hi = tl.maximum(0, tl.minimum(T, hi))
    local = 0
    if PACKED and G > B:
        if v >= B:
            local = 4 * tl.maximum(1, tl.cdiv(anchor - lo, 4))
    first = lo + local
    owns_group = (v < B) | ((first >= anchor) & (first < anchor + 4))
    occupied = tl.where(owns_group, tl.maximum(0, tl.minimum(4, hi - first)), 0)
    valid = (used >= 0) & (used <= hi - lo) & (kv >= 0) & (kv <= M * 128)
    count = tl.where(valid, tl.maximum(0, tl.minimum(occupied, used - local)), 0)
    kv_base = kv - used + local
    if count > 0:
        p = tl.arange(0, triton.next_power_of_2(M))
        pages = tl.load(TABLE + batch.to(tl.int64) * M + p, p < M, other=0)
        first_bad = tl.min(tl.where((p < M) & ((pages < 0) | (pages >= P)), p * 128, M * 128), 0)
        if CAUSAL:
            count = tl.minimum(count, tl.maximum(0, first_bad - kv_base))
        else:
            count = tl.where(kv <= first_bad, count, 0)
    end = tl.maximum(0, kv_base + count) if CAUSAL else kv
    end = tl.where(count > 0, end, 0)
    first_good = tl.minimum(count, tl.maximum(0, -kv_base)) if CAUSAL else 0
    count = tl.where(end > 0, count, 0)
    # Slide the four-token read inwards at the end of Q. The per-record
    # diagonal correction lets the native kernel consume these original rows.
    qbase = tl.minimum(T - 4, tl.maximum(0, first))
    _write_record(TILING, v, batch, end, qbase, tl.maximum(0, first - qbase), first_good, count, SPLITS)
    if (occupied > 0) & ((first_good > 0) | (count < occupied)):
        head = tl.arange(0, 4)
        d = tl.arange(0, 512)
        for j in tl.static_range(4):
            if (j < occupied) & ((j < first_good) | (j >= count)):
                t = first + j
                row = head * T + t if NTD else t * 4 + head
                tl.store(Y + row.to(tl.int64)[:, None] * 512 + d[None, :], tl.full((4, 512), 0, tl.bfloat16))


@triton.jit
def _schedule(
    TABLE,
    LEN,
    CU,
    USED,
    PLAN,
    TILING,
    Y,
    B: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    PACKED: tl.constexpr,
    HAS_PLAN: tl.constexpr,
    HAS_USED: tl.constexpr,
    CAUSAL: tl.constexpr,
    NTD: tl.constexpr,
    SCALE: tl.constexpr,
    SPAN: tl.constexpr,
    SPLITS: tl.constexpr,
    SMALL: tl.constexpr,
):
    if SMALL:
        b = tl.arange(0, triton.next_power_of_2(B))
        regular = tl.full((triton.next_power_of_2(B),), 1, tl.int1)
        if HAS_PLAN:
            used = tl.load(PLAN + 4 * b + 2, b < B, other=4)
            kv = tl.load(PLAN + 4 * b + 3, b < B, other=4)
            regular = regular & (tl.load(PLAN + 4 * B) == 0x4D4C4133) & (tl.load(PLAN + 4 * B + 1) == PACKED)
            if PACKED:
                lo = tl.load(PLAN + 4 * b, b < B, other=0)
                hi = tl.load(PLAN + 4 * b + 1, b < B, other=0)
                regular = regular & (lo == 4 * b) & (hi == 4 * b + 4)
        else:
            if PACKED:
                lo = tl.load(CU + b, b < B, other=0)
                hi = tl.load(CU + b + 1, b < B, other=0)
                regular = regular & (lo == 4 * b) & (hi == 4 * b + 4)
            used = tl.load(USED + b, b < B, other=4) if HAS_USED else 4
            kv = tl.load(LEN + b, b < B, other=4)
        regular = regular & (used == 4) & (kv >= 4) & (kv <= M * 128)
        p = tl.arange(0, triton.next_power_of_2(M))
        pages = tl.load(TABLE + b[:, None] * M + p[None, :], (b < B)[:, None] & (p < M)[None, :], other=0)
        bad = (p[None, :] * 128 < kv[:, None]) & ((pages < 0) | (pages >= P))
        failed = (b < B)[:, None] & ((~regular)[:, None] | bad)
        flat = tl.reshape(failed.to(tl.int32), (triton.next_power_of_2(B) * triton.next_power_of_2(M),))
        if tl.max(flat, 0) == 0:
            # Emit all regular records with contiguous stores from one vector core.
            _header(TILING, B, 16, M, P, SCALE, CAUSAL, SPAN, SPLITS, T if NTD else 0, False)
            w = tl.arange(0, 32)[None, :]
            q_offset = (b * 8192)[:, None]
            rec = (w == 0).to(tl.int32) + tl.where(w == 1, kv[:, None], 0) + tl.where(w == 3, 128, 0)
            rec += tl.where((w == 5) | (w == 14), q_offset, 0)
            rec += tl.where(w == 7, (b * 1024)[:, None], 0) + tl.where(w == 12, (b * 16 * SPLITS)[:, None], 0)
            tl.store(TILING + (b[:, None] + 1) * 32 + w, rec, (b < B)[:, None])
        else:
            for v in range(G):
                _group(
                    v,
                    TABLE,
                    LEN,
                    CU,
                    USED,
                    PLAN,
                    TILING,
                    Y,
                    B,
                    T,
                    G,
                    M,
                    P,
                    PACKED,
                    HAS_PLAN,
                    HAS_USED,
                    CAUSAL,
                    NTD,
                    SCALE,
                    SPAN,
                    SPLITS,
                )
    else:
        _group(
            tl.program_id(0),
            TABLE,
            LEN,
            CU,
            USED,
            PLAN,
            TILING,
            Y,
            B,
            T,
            G,
            M,
            P,
            PACKED,
            HAS_PLAN,
            HAS_USED,
            CAUSAL,
            NTD,
            SCALE,
            SPAN,
            SPLITS,
        )


@triton.jit
def _single_mtp_schedule(
    TABLE,
    LEN,
    CU,
    USED,
    PLAN,
    TILING,
    Y,
    M: tl.constexpr,
    P: tl.constexpr,
    PACKED: tl.constexpr,
    HAS_PLAN: tl.constexpr,
    HAS_USED: tl.constexpr,
    SCALE: tl.constexpr,
    NTD: tl.constexpr,
    SPAN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    # A single request owns exactly four physical query rows. Validate the
    # rows and pages while writing the 32-word header and record in one store.
    if HAS_PLAN:
        valid = (tl.load(PLAN + 4) == 0x4D4C4133) & (tl.load(PLAN + 5) == PACKED)
        if PACKED:
            valid = valid & (tl.load(PLAN) == 0) & (tl.load(PLAN + 1) == 4)
        used = tl.load(PLAN + 2)
        kv = tl.load(PLAN + 3)
    else:
        valid = (tl.load(CU) == 0) & (tl.load(CU + 1) == 4) if PACKED else True
        used = tl.load(USED) if HAS_USED else 4
        kv = tl.load(LEN)
    page = tl.arange(0, triton.next_power_of_2(M))
    physical = tl.load(TABLE + page, page < M, other=0)
    bad = (page < M) & ((physical < 0) | (physical >= P))
    first_bad = tl.min(tl.where(bad, page * 128, M * 128), 0)
    valid = valid & (used >= 0) & (used <= 4) & (kv >= 0) & (kv <= M * 128)
    base = kv - used
    count = tl.maximum(0, tl.minimum(used, first_bad - base))
    end = tl.maximum(0, base + count)
    first_good = tl.minimum(count, tl.maximum(0, -base))
    count = tl.where(valid & (end > 0), count, 0)
    end = tl.where(count > 0, end, 0)
    full = (count == 4) & (first_good == 0)

    w = tl.arange(0, 64)
    data = ((w == 0) | (w == 7) | (w == 11) | (w == 17) | (w == 18) | (w == 32)).to(tl.int32)
    data += tl.where((w == 1) | (w == 10), 16, 0) + tl.where(w == 2, 512, 0) + tl.where(w == 3, P, 0)
    data += tl.where((w == 4) | (w == 35), 128, 0) + tl.where(w == 5, M, 0) + tl.where(w == 6, SCALE, 0)
    data += tl.where((w == 8) | (w == 9), 32, 0) + tl.where(w == 12, 3, 0)
    data += tl.where(w == 13, 64, 0) + tl.where(w == 14, M * 128, 0) + tl.where(w == 15, SPAN, 0)
    data += tl.where(w == 16, SPLITS, 0) + tl.where(w == 21, 576, 0) + tl.where(w == 22, 4 if NTD else 0, 0)
    data += tl.where(w == 23, (~full).to(tl.int32), 0) + tl.where(w == 33, end, 0)
    data += tl.where(w == 50, 4 - count, 0) + tl.where(w == 51, 4 * first_good, 0) + tl.where(w == 52, 4 * count, 0)
    tl.store(TILING + w, data)
    if not full:
        head = tl.arange(0, 4)
        dim = tl.arange(0, 512)
        for token in tl.static_range(4):
            if (token < first_good) | (token >= count):
                row = head * 4 + token if NTD else token * 4 + head
                tl.store(Y + row[:, None] * 512 + dim[None, :], tl.full((4, 512), 0, tl.bfloat16))


def direct_decode(q, cache, table, lengths, cu, used, plan, scale, causal, ntd, span, splits):
    """Run multi-split attention without copying normalized queries, page tables or output."""
    from .flash_mla_a3 import _single_mtp_tail

    batches, width = table.shape
    tokens = q.size(0)
    groups = batches
    if cu is not None and batches > 1 and tokens > 4:
        groups += triton.cdiv(tokens, 4)
    small = tokens == 4 * batches and 2 <= batches <= 16 and width <= 64
    tiling = table.new_empty((groups + 1) * 32)
    out = q.new_empty((4, tokens, 512) if ntd else (tokens, 4, 512))
    scale_bits = struct.unpack("=i", struct.pack("=f", scale))[0]
    if _single_mtp_tail(batches, tokens, width, causal):
        # Five KV pages per split keep all 15–20 splits in a single Cube wave.
        span, splits = 5 * 128, triton.cdiv(width, 5)
        _single_mtp_schedule[(1,)](
            table,
            lengths,
            cu if cu is not None else lengths,
            used if used is not None else lengths,
            plan if plan is not None else lengths,
            tiling,
            out,
            width,
            cache.size(0),
            cu is not None,
            plan is not None,
            used is not None,
            scale_bits,
            ntd,
            span,
            splits,
            num_stages=1,
            enable_ubuf_saving=True,
        )
    else:
        _schedule[(1 if small else groups,)](
            table,
            lengths,
            cu if cu is not None else lengths,
            used if used is not None else lengths,
            plan if plan is not None else lengths,
            tiling,
            out,
            batches,
            tokens,
            groups,
            width,
            cache.size(0),
            cu is not None,
            plan is not None,
            used is not None,
            causal,
            ntd,
            scale_bits,
            span,
            splits,
            small,
            num_stages=1,
            enable_ubuf_saving=small,
        )
    torch.ops.flash_mla_a3.catlass_direct(q, cache, table, tiling, out, splits, groups)
    return out
