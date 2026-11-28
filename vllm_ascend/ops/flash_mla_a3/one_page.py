# SPDX-License-Identifier: Apache-2.0
"""Single-page A3 Cube implementation for groups of four query positions."""

from vllm.triton_utils import tl, triton


@triton.jit
def _four_tokens(
    Q,
    CACHE,
    OUT,
    first,
    stop,
    used,
    local,
    kv_len,
    page,
    valid,
    SCALE: tl.constexpr,
    TOKENS: tl.constexpr,
    CAUSAL: tl.constexpr,
    NTD: tl.constexpr,
    COLS: tl.constexpr = 128,
    SINGLE: tl.constexpr = False,
    QR=None,
    KR=None,
    SEPARATE: tl.constexpr = False,
    CACHE_PAGE_STRIDE: tl.constexpr = 0,
    ROPE_PAGE_STRIDE: tl.constexpr = 0,
    HEADS: tl.constexpr = 4,
):
    row = tl.arange(0, 16)
    col = tl.arange(0, COLS)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    width = 512 if SEPARATE else 576
    q_base = Q + first.to(tl.int64) * HEADS * width
    remaining = tl.minimum(stop - first, TOKENS - first)
    q_mask = row[:, None] < remaining * HEADS
    cache_page_stride = CACHE_PAGE_STRIDE if CACHE_PAGE_STRIDE else 128 * width
    kv_base = CACHE + page.to(tl.int64) * cache_page_stride
    if SEPARATE:
        qr_base = QR + first.to(tl.int64) * HEADS * 64
        rope_page_stride = ROPE_PAGE_STRIDE if ROPE_PAGE_STRIDE else 128 * 64
        kr_base = KR + page.to(tl.int64) * rope_page_stride
        qr = tl.load(qr_base + row[:, None] * 64 + r[None, :], q_mask, other=0.0)
        kr = tl.load(kr_base + col[:, None] * 64 + r[None, :])
    else:
        qr = tl.load(q_base + row[:, None] * 576 + 512 + r[None, :], q_mask, other=0.0)
        kr = tl.load(kv_base + col[:, None] * 576 + 512 + r[None, :])
    scores = tl.dot(qr, tl.trans(kr))
    q = tl.load(q_base + row[:, None] * width + d[None, :], q_mask, other=0.0)
    # Mask while loading so unwritten cache slots do not contaminate PV.
    v = tl.load(kv_base + col[:, None] * width + d[None, :], (col < kv_len)[:, None], other=0.0)
    scores += tl.dot(q, tl.trans(v))
    end = tl.full((16,), 0, tl.int32) + kv_len
    if CAUSAL:
        end = tl.minimum(end, kv_len - used + local + row // HEADS + 1)
    end = tl.where(valid & (local + row // HEADS < used) & (kv_len <= 128) & (kv_len >= 0), end, 0)
    scores = tl.where(col[None, :] < end[:, None], scores * SCALE, -float("inf"))
    maximum = tl.max(scores, axis=1)
    weights = tl.exp(scores - tl.where(maximum == -float("inf"), 0.0, maximum)[:, None])
    total = tl.sum(weights, axis=1)
    result = tl.dot(weights.to(tl.bfloat16), v) / tl.where(total == 0.0, 1.0, total)[:, None]
    result = tl.where((end > 0)[:, None], result, 0.0).to(tl.bfloat16)
    if SINGLE and NTD:
        tl.store(OUT + (row * TOKENS + first).to(tl.int64)[:, None] * 512 + d[None, :], result, row[:, None] < HEADS)
    elif NTD:
        y = tl.trans(tl.reshape(result, (16 // HEADS, HEADS, 512)), (1, 0, 2))
        head = tl.arange(0, HEADS)[:, None, None]
        s = tl.arange(0, 16 // HEADS)[None, :, None]
        tl.store(OUT + (head * TOKENS + first + s) * 512 + d[None, None, :], y, s < remaining)
    else:
        tl.store(OUT + (first.to(tl.int64) * HEADS + row[:, None]) * 512 + d[None, :], result, q_mask)


@triton.jit
def _batch_for_token(CU, PLAN, token, guess, B: tl.constexpr, TOKENS: tl.constexpr, HAS_PLAN: tl.constexpr):
    if HAS_PLAN:
        lo = tl.load(PLAN + 4 * guess)
        hi = tl.load(PLAN + 4 * guess + 1)
    else:
        lo = tl.load(CU + guess)
        hi = tl.load(CU + guess + 1)
    batch = guess
    if (token < lo) | (token >= hi):
        indices = tl.arange(0, triton.next_power_of_2(B))
        if HAS_PLAN:
            ends = tl.load(PLAN + 4 * indices + 1, indices < B, other=TOKENS)
        else:
            ends = tl.load(CU + indices + 1, indices < B, other=TOKENS)
        batch = tl.minimum(B - 1, tl.sum((ends <= token).to(tl.int32), axis=0))
    return batch


@triton.jit
def _onepage_batch(
    Q,
    CACHE,
    TABLE,
    LEN,
    CU,
    USED,
    PLAN,
    OUT,
    SCALE: tl.constexpr,
    TOKENS: tl.constexpr,
    B: tl.constexpr,
    PAGES: tl.constexpr,
    PACKED: tl.constexpr,
    HAS_PLAN: tl.constexpr,
    HAS_USED: tl.constexpr,
    CAUSAL: tl.constexpr,
    NTD: tl.constexpr,
    BY_TOKEN: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = tl.minimum(pid, B - 1)
    # The first B programs own the first four positions in each sequence.
    # At most one *additional* group can start in any four physical tokens.
    # It belongs to the same sequence as the bin's first token: a subsequent
    # sequence needs four earlier tokens of its own to begin its second group.
    if PACKED and BY_TOKEN:
        batch = _batch_for_token(CU, PLAN, pid, pid, B, TOKENS, HAS_PLAN)
    elif PACKED and TOKENS > 4:
        if pid >= B:
            anchor = (pid - B) * 4
            guess = tl.minimum(anchor // triton.cdiv(TOKENS, B), B - 1)
            batch = _batch_for_token(CU, PLAN, anchor, guess, B, TOKENS, HAS_PLAN)
    if HAS_PLAN:
        rec = PLAN + 4 * batch
        lo = tl.load(rec) if PACKED else batch * (TOKENS // B)
        hi = tl.load(rec + 1) if PACKED else lo + TOKENS // B
        used = tl.load(rec + 2)
        kv_len = tl.load(rec + 3)
        header = PLAN + 4 * B
        valid_layout = (tl.load(header) == 0x4D4C4133) & (tl.load(header + 1) == PACKED)
    else:
        lo = tl.load(CU + batch) if PACKED else batch * (TOKENS // B)
        hi = tl.load(CU + batch + 1) if PACKED else lo + TOKENS // B
        used = tl.load(USED + batch) if HAS_USED else hi - lo
        kv_len = tl.load(LEN + batch)
        valid_layout = True
    if PACKED:
        if HAS_PLAN:
            valid_layout = valid_layout & (tl.load(PLAN) == 0) & (tl.load(PLAN + 4 * (B - 1) + 1) == TOKENS)
        else:
            valid_layout = valid_layout & (tl.load(CU) == 0) & (tl.load(CU + B) == TOKENS)
    start = tl.maximum(lo, 0)
    stop = tl.minimum(hi, TOKENS)
    page = tl.load(TABLE + batch)
    page_ok = (page >= 0) & (page < PAGES)
    valid = page_ok & (used >= 0) & (used <= hi - lo) & valid_layout
    page = tl.where(page_ok, page, 0)
    if BY_TOKEN:
        # Each program owns exactly [pid, pid+1), even if the query prefix
        # moves to another batch while a captured graph is being replayed.
        local = pid - start
        valid = valid & (pid >= start) & (pid < stop)
        if kv_len <= 16:
            _four_tokens(
                Q, CACHE, OUT, pid, pid + 1, used, local, kv_len, page, valid, SCALE, TOKENS, CAUSAL, NTD, 16, True
            )
        elif kv_len <= 32:
            _four_tokens(
                Q, CACHE, OUT, pid, pid + 1, used, local, kv_len, page, valid, SCALE, TOKENS, CAUSAL, NTD, 32, True
            )
        else:
            _four_tokens(
                Q, CACHE, OUT, pid, pid + 1, used, local, kv_len, page, valid, SCALE, TOKENS, CAUSAL, NTD, 128, True
            )
    else:
        local = 0
        launch = pid < B
        if PACKED and TOKENS > 4:
            if pid >= B:
                anchor = (pid - B) * 4
                local = tl.maximum(1, tl.cdiv(anchor - start, 4)) * 4
                launch = (start + local >= anchor) & (start + local < anchor + 4) & (start + local < stop)
        first = tl.where(valid_layout, start + local, pid * 4)
        stop = tl.where(valid_layout, stop, TOKENS)
        launch = tl.where(valid_layout, launch, pid * 4 < TOKENS)
        if launch:
            if kv_len <= 32:
                remaining = tl.minimum(stop - first, TOKENS - first)
                if NTD and remaining == 1:
                    _four_tokens(
                        Q,
                        CACHE,
                        OUT,
                        first,
                        stop,
                        used,
                        local,
                        kv_len,
                        page,
                        valid,
                        SCALE,
                        TOKENS,
                        CAUSAL,
                        NTD,
                        32,
                        True,
                    )
                else:
                    _four_tokens(
                        Q, CACHE, OUT, first, stop, used, local, kv_len, page, valid, SCALE, TOKENS, CAUSAL, NTD, 32
                    )
            else:
                _four_tokens(Q, CACHE, OUT, first, stop, used, local, kv_len, page, valid, SCALE, TOKENS, CAUSAL, NTD)


def batch_one_page_decode(q, cache, table, lengths, cu, used, plan, scale, causal, ntd):
    """Four query heads, one 128-token page per sequence, average Q length up to four."""
    tokens, batches = q.size(0), table.size(0)
    out = q.new_empty((4, tokens, 512) if ntd else (tokens, 4, 512))
    by_token = batches >= 8 and tokens == batches
    programs = batches + (triton.cdiv(tokens, 4) if cu is not None and tokens > 4 and not by_token else 0)
    _onepage_batch[(programs,)](
        q,
        cache,
        table,
        lengths,
        cu if cu is not None else lengths,
        used if used is not None else lengths,
        plan if plan is not None else lengths,
        out,
        scale,
        tokens,
        batches,
        cache.size(0),
        cu is not None,
        plan is not None,
        used is not None,
        causal,
        ntd,
        by_token,
        num_stages=1,
        enable_ubuf_saving=True,
    )
    return out


@triton.jit
def _uniform_single_page(
    Q,
    CACHE,
    QR,
    KR,
    TABLE,
    LENGTHS,
    OUT,
    SCALE: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    CAUSAL: tl.constexpr,
    NTD: tl.constexpr,
    SEPARATE: tl.constexpr,
    CACHE_PAGE_STRIDE: tl.constexpr,
    ROPE_PAGE_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    TILES: tl.constexpr,
):
    batch = tl.program_id(0) // TILES
    local = (tl.program_id(0) % TILES) * (16 // HEADS)
    length = tl.load(LENGTHS + batch)
    page = tl.load(TABLE + batch)
    # The per-step plan validates the physical page map and the maximum live
    # history. With zero live keys the physical slot can contain a sentinel.
    page = tl.where(length > 0, page, 0)
    _four_tokens(
        Q,
        CACHE,
        OUT,
        batch * S + local,
        (batch + 1) * S,
        S,
        local,
        length,
        page,
        True,
        SCALE,
        B * S,
        CAUSAL,
        NTD,
        C,
        S == 1,
        QR,
        KR,
        SEPARATE,
        CACHE_PAGE_STRIDE,
        ROPE_PAGE_STRIDE,
        HEADS,
    )


def uniform_single_page(q, cache, table, lengths, query_length, maximum, scale, causal, ntd, q_rope=None, k_rope=None):
    """Use the scheduler's maximum live history to size the single-page tiles."""
    b = table.size(0)
    t, heads = q.shape[:2]
    out = q.new_empty((heads, t, 512) if ntd else (t, heads, 512))
    cols = max(16, triton.next_power_of_2(max(maximum, 1)))
    tiles = triton.cdiv(query_length * heads, 16)
    _uniform_single_page[(b * tiles,)](
        q,
        cache,
        q_rope if q_rope is not None else q,
        k_rope if k_rope is not None else cache,
        table,
        lengths,
        out,
        scale,
        b,
        query_length,
        cols,
        causal,
        ntd,
        q_rope is not None,
        cache.stride(0),
        k_rope.stride(0) if k_rope is not None else 0,
        heads,
        tiles,
        num_stages=1,
        enable_ubuf_saving=True,
    )
    return out
