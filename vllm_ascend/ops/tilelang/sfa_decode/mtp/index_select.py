# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from .long_context_topk import CHUNK_POOLS, TOP_K, make_long_rankers, make_long_selector


def pack_dense_indices(batch, requests, width):
    padded_width = (width + 63) // 64 * 64

    @T.prim_func
    def dense(
        Pos: T.Tensor((batch,), T.int64),
        Ends: T.Tensor((requests,), T.int64),
        Y: T.Tensor((batch, width), T.int32),
    ):
        with T.Kernel(batch) as bx:
            values = T.alloc_shared((padded_width,), T.int32)
            length = T.if_then_else(bx < Ends[requests - 1], T.cast(Pos[bx] + 1, T.int32), 0)
            with T.SimdVF():
                mask = S.pset(32)
                lanes = S.vci(T.int32(0), "int32")
                invalid = S.vdup(T.int32(-1), "int32", mask)
                for tile in range(padded_width // 64):
                    cols = S.vadds(lanes, tile * 64, mask)
                    valid = S.vcmps(cols, length, mask, op="lt")
                    S.vsts(values[tile * 64], S.vsel(cols, invalid, valid), mask, "NORM_B32")
            T.copy(values[:width], Y[bx, :])

    return dense


def rank_long_chunks(batch, requests, pools):
    chunks = pools // CHUNK_POOLS
    candidates = chunks * TOP_K
    rank_chunk, _ = make_long_rankers(pools)

    @T.prim_func
    def rank(
        X: T.Tensor((batch, pools), T.bfloat16),
        Pos: T.Tensor((batch,), T.int64),
        Ends: T.Tensor((requests,), T.int64),
        Scores: T.Tensor((batch, candidates), T.bfloat16),
        LocalIds: T.Tensor((batch, candidates), T.uint16),
    ):
        with T.Kernel(batch * chunks) as task:
            bx = task // chunks
            chunk = task % chunks
            visible = T.if_then_else(bx < Ends[requests - 1], T.cast(Pos[bx] + 1, T.int32) // 4, 0)
            start = chunk * CHUNK_POOLS
            count = T.min(visible - start, CHUNK_POOLS)
            if count > 0 and visible >= TOP_K:
                raw = T.alloc_shared((CHUNK_POOLS,), T.bfloat16)
                ords = T.view(raw, (CHUNK_POOLS,), dtype=T.uint16)
                hist = T.alloc_shared((256,), T.uint16)
                bounds = T.alloc_shared((8,), T.int32)
                local_ids = T.alloc_shared((TOP_K,), T.uint16)
                shortlist = T.alloc_shared((1, TOP_K), T.bfloat16)
                words = T.view(shortlist, (1, TOP_K), dtype=T.uint16)
                ids = T.alloc_shared((1, TOP_K), T.uint16)
                rank_chunk(X, bx, start, count, 0, raw, ords, hist, bounds, local_ids, shortlist, words, ids)
                T.copy(shortlist[0, :], Scores[bx, chunk * TOP_K : (chunk + 1) * TOP_K])
                T.copy(ids[0, :], LocalIds[bx, chunk * TOP_K : (chunk + 1) * TOP_K])

    return rank


def pack_program(batch, requests, context, width=2176, *, merge_shortlists=False):
    clip = True
    pools = context // 4

    @T.macro
    def expand(top, length, visible, Y, bx):
        token_ids = T.alloc_shared((2176,), T.int32)
        history = T.min(visible, 512) * 4
        with T.SimtVF(threads=128):
            for fragment in T.serial(2176 // 128):
                for lane in T.Parallel(128):
                    col = fragment * 128 + lane
                    full_id = T.cast(top[T.min(col // 4, 511)], T.int32) * 4 + col % 4
                    suffix = col - history
                    token_ids[col] = T.if_then_else(
                        col < history, full_id, T.if_then_else(suffix < length % 4, visible * 4 + suffix, -1)
                    )
        T.copy(token_ids[:width], Y[bx, :])

    if merge_shortlists:
        chunks = pools // CHUNK_POOLS
        candidates = chunks * TOP_K
        _, merge_chunks = make_long_rankers(pools)

        @T.prim_func
        def merge(
            Scores: T.Tensor((batch, candidates), T.bfloat16),
            LocalIds: T.Tensor((batch, candidates), T.uint16),
            Pos: T.Tensor((batch,), T.int64),
            Ends: T.Tensor((requests,), T.int64),
            Y: T.Tensor((batch, width), T.int32),
        ):
            with T.Kernel(batch) as bx:
                top = T.alloc_shared((TOP_K,), T.int32)
                length = T.if_then_else(bx < Ends[requests - 1], T.cast(Pos[bx] + 1, T.int32), 0)
                visible = length // 4
                if visible >= TOP_K:
                    scores = T.alloc_shared((candidates,), T.bfloat16)
                    ords = T.view(scores, (candidates,), dtype=T.uint16)
                    ids = T.alloc_shared((candidates,), T.uint16)
                    by_chunk = T.view(ids, (chunks, TOP_K), dtype=T.uint16)
                    hist = T.alloc_shared((256,), T.uint16)
                    bounds = T.alloc_shared((8,), T.int32)
                    local_ids = T.alloc_shared((TOP_K,), T.uint16)
                    span = T.ceildiv(visible, CHUNK_POOLS) * TOP_K
                    T.copy(Scores[bx, :span], scores[:span])
                    T.copy(LocalIds[bx, :span], ids[:span])
                    merge_chunks(Scores, bx, visible, scores, ords, by_chunk, hist, bounds, local_ids, top)
                else:
                    with T.SimdVF():
                        mask = S.pset(32)
                        lanes = S.vci(T.int32(0), "int32")
                        for j in range(TOP_K // 64):
                            S.vsts(top[j * 64], S.vadds(lanes, j * 64, mask), mask, "NORM_B32")
                expand(top, length, visible, Y, bx)

        return merge

    if pools >= 65536:
        select = make_long_selector(pools)

        @T.prim_func
        def long_pack(
            X: T.Tensor((batch, pools), T.bfloat16),
            Pos: T.Tensor((batch,), T.int64),
            Ends: T.Tensor((requests,), T.int64),
            Y: T.Tensor((batch, width), T.int32),
        ):
            with T.Kernel(batch) as bx:
                top = T.alloc_shared((512,), T.int32)
                length = T.if_then_else(bx < Ends[requests - 1], T.cast(Pos[bx] + 1, T.int32), 0)
                visible = length // 4
                select(X, bx, visible, top)
                expand(top, length, visible, Y, bx)

        return long_pack

    groups = pools // 128
    block = 16
    lanes = groups // block
    assert groups % block == 0 and lanes <= 32
    assert pools % 256 == 0 and pools < 65536

    @T.prim_func
    def main(
        X: T.Tensor((batch, pools), T.bfloat16),
        Pos: T.Tensor((batch,), T.int64),
        Ends: T.Tensor((requests,), T.int64),
        Y: T.Tensor((batch, width), T.int32),
    ):
        with T.Kernel(batch) as bx:
            raw = T.alloc_shared((pools,), T.bfloat16)
            ords = T.view(raw, (pools,), dtype=T.uint16)
            hist = T.alloc_shared((256,), T.uint16)
            bounds = T.alloc_shared((8,), T.int32)
            gt_base = T.alloc_shared((lanes,), T.uint32)
            eq_base = T.alloc_shared((lanes,), T.uint32)
            gt_total = T.alloc_shared((lanes,), T.uint32)
            eq_total = T.alloc_shared((lanes,), T.uint32)
            ge = T.alloc_shared((groups,), T.uint32)
            eq = T.alloc_shared((groups,), T.uint32)
            top = T.alloc_shared((512,), T.uint16)
            length = T.if_then_else(bx < Ends[requests - 1], T.cast(Pos[bx] + 1, T.int32), 0)
            visible = length // 4
            # Rank full four-token pools using the query position to exclude future tokens.
            if visible >= 512:
                span = T.ceildiv(visible, 256) * 256 if clip else pools
                T.copy(X[bx, 0:span], raw[0:span])
                with T.SimdVF():
                    m = S.pset(16)
                    sign = S.vdup(T.uint16(32768), "uint16", m)
                    bins0 = S.alloc_var("uint16")
                    bins1 = S.alloc_var("uint16")
                    bins0 = S.vdup(T.uint16(0), "uint16", m)
                    bins1 = S.vdup(T.uint16(0), "uint16", m)
                    both = S.pset(8)
                    zeros = S.vdup(T.uint16(0), "uint16", m)
                    offsets = T.reinterpret(S.vci(T.int16(0), "int16"), "uint16x128")
                    for j in range(span // 256):
                        a = S.vld(ords[j * 256], "NORM")
                        b = S.vld(ords[j * 256 + 128], "NORM")
                        ka = S.vsel(S.vnot(a, m), S.vxor(a, sign, m), S.vcmps(a, T.uint16(32768), m, op="ge"))
                        kb = S.vsel(S.vnot(b, m), S.vxor(b, sign, m), S.vcmps(b, T.uint16(32768), m, op="ge"))
                        ia = S.vadds(offsets, T.cast(j * 256, T.uint16), m)
                        ib = S.vadds(offsets, T.cast(j * 256 + 128, T.uint16), m)
                        oa = S.vsel(ka, zeros, S.vcmps(ia, T.cast(visible, T.uint16), m, op="lt"))
                        ob = S.vsel(kb, zeros, S.vcmps(ib, T.cast(visible, T.uint16), m, op="lt"))
                        S.vsts(ords[j * 256], oa, m, "NORM_B16")
                        S.vsts(ords[j * 256 + 128], ob, m, "NORM_B16")
                        lo, hi = S.vdintlv(T.reinterpret(oa, "uint8x256"), T.reinterpret(ob, "uint8x256"))
                        S.chistv2(bins0, hi, both, bin=0)
                        S.chistv2(bins1, hi, both, bin=1)
                    S.vsts(hist[0], bins0, m, "NORM_B16")
                    S.vsts(hist[128], bins1, m, "NORM_B16")
                with T.SimtVF(threads=256):
                    votes = T.alloc_fragment((1, 256), T.int32)
                    total = T.alloc_fragment((1,), T.int32)
                    for t in T.Parallel(256):
                        target = span - 511
                        votes[0, t] = T.cast(T.cast(hist[t], T.int32) < target, T.int32)
                    T.reduce_sum(votes, total, dim=1)
                    for t in T.Parallel(1):
                        cutoff = total[t]
                        bounds[0] = cutoff
                        prev = T.if_then_else(cutoff == 0, 0, T.cast(hist[T.max(cutoff - 1, 0)], T.int32))
                        bounds[1] = span - 511 - prev
                with T.SimdVF():
                    m = S.pset(16)
                    bytes_mask = S.pset(8)
                    c0 = S.alloc_var("uint16")
                    c1 = S.alloc_var("uint16")
                    c0 = S.vdup(T.uint16(0), "uint16", m)
                    c1 = S.vdup(T.uint16(0), "uint16", m)
                    for j in range(span // 256):
                        a = S.vld(ords[j * 256], "NORM")
                        b = S.vld(ords[j * 256 + 128], "NORM")
                        lo, hi = S.vdintlv(T.reinterpret(a, "uint8x256"), T.reinterpret(b, "uint8x256"))
                        yes = S.vcmps(hi, T.cast(bounds[0], T.uint8), bytes_mask, op="eq")
                        S.chistv2(c0, lo, yes, bin=0)
                        S.chistv2(c1, lo, yes, bin=1)
                    S.vsts(hist[0], c0, m, "NORM_B16")
                    S.vsts(hist[128], c1, m, "NORM_B16")
                with T.SimtVF(threads=256):
                    votes = T.alloc_fragment((1, 256), T.int32)
                    total = T.alloc_fragment((1,), T.int32)
                    for t in T.Parallel(256):
                        target = bounds[1]
                        votes[0, t] = T.cast(T.cast(hist[t], T.int32) < target, T.int32)
                    T.reduce_sum(votes, total, dim=1)
                    for t in T.Parallel(1):
                        cutoff = total[t]
                        bounds[2] = bounds[0] * 256 + cutoff
                with T.SimdVF():
                    m = S.pset(16)
                    scalar = S.pset(32, "PAT_VL1")
                    one = S.vdup(T.uint16(1), "uint16", m)
                    zero = S.vdup(T.uint16(0), "uint16", m)
                    for j in range(span // 128):
                        val = S.vld(ords[j * 128], "NORM")
                        high = S.vcmps(val, T.cast(bounds[2], T.uint16), m, op="gt")
                        same = S.vcmps(val, T.cast(bounds[2], T.uint16), m, op="eq")
                        S.vsts(ge[j], S.vcadd(S.vsel(one, zero, high), m), scalar, "ONEPT_B32")
                        S.vsts(eq[j], S.vcadd(S.vsel(one, zero, same), m), scalar, "ONEPT_B32")
                with T.SimtVF(threads=32):
                    for lane in T.Parallel(lanes):
                        gsum = T.alloc_var(T.uint32, init=0)
                        esum = T.alloc_var(T.uint32, init=0)
                        for slot in T.unroll(block):
                            j = lane * block + slot
                            if j < span // 128:
                                gc = ge[j]
                                ec = eq[j]
                                ge[j] = gsum
                                eq[j] = esum
                                gsum += gc
                                esum += ec
                        gt_total[lane] = gsum
                        eq_total[lane] = esum
                with T.SimtVF(threads=32):
                    for lane in T.Parallel(lanes):
                        gsum = T.alloc_var(T.uint32, init=0)
                        esum = T.alloc_var(T.uint32, init=0)
                        for prior in T.unroll(lanes):
                            if prior < lane:
                                gsum += gt_total[prior]
                                esum += eq_total[prior]
                        gt_base[lane] = gsum
                        eq_base[lane] = esum
                        if lane == lanes - 1:
                            bounds[3] = T.cast(gsum + gt_total[lane], T.int32)
                with T.SimdVF():
                    m = S.pset(16)
                    offsets = T.reinterpret(S.vci(T.int16(0), "int16"), "uint16x128")
                    for j in range(span // 128):
                        v = S.vld(ords[j * 128], "NORM")
                        high = S.vcmps(v, T.cast(bounds[2], T.uint16), m, op="gt")
                        same = S.vcmps(v, T.cast(bounds[2], T.uint16), m, op="eq")
                        ri = S.vadds(
                            T.reinterpret(S.vusqz(high, "int16"), "uint16x128"),
                            T.cast(ge[j] + gt_base[j // block], T.uint16),
                            m,
                        )
                        ti = S.vadds(
                            T.reinterpret(S.vusqz(same, "int16"), "uint16x128"),
                            T.cast(eq[j] + eq_base[j // block] + T.cast(bounds[3], T.uint32), T.uint16),
                            m,
                        )
                        ids = S.vadds(offsets, T.cast(j * 128, T.uint16), m)
                        S.vscatter(ids, top[0], ri, high)
                        S.vscatter(ids, top[0], ti, S.pand(same, S.vcmps(ti, T.uint16(512), m, op="lt"), m))
            else:
                with T.SimdVF():
                    m = S.pset(16)
                    ix = T.reinterpret(S.vci(T.int16(0), "int16"), "uint16x128")
                    for part in range(4):
                        S.vsts(top[part * 128], S.vadds(ix, T.cast(part * 128, T.uint16), m), m, "NORM_B16")
            expand(top, length, visible, Y, bx)

    return main
