# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def make_pool_ranker(pools, top_k, clip, *, preloaded=False):
    @T.macro
    def rank(X, bx, visible, raw, ords, hist, bounds, top):
        # At exactly top-k, the native indexer still places cutoff ties last.
        if visible >= top_k:
            span = T.ceildiv(visible, 256) * 256 if clip else pools
            if not preloaded:
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
                    target = span - (top_k - 1)
                    votes[0, t] = T.cast(T.cast(hist[t], T.int32) < target, T.int32)
                T.reduce_sum(votes, total, dim=1)
                for t in T.Parallel(1):
                    cutoff = total[t]
                    bounds[0] = cutoff
                    prev = T.if_then_else(cutoff == 0, 0, T.cast(hist[T.max(cutoff - 1, 0)], T.int32))
                    bounds[1] = span - (top_k - 1) - prev
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
                    # The low-byte CDF includes cutoff ties. The remaining scores are strictly greater.
                    bounds[3] = top_k - 1 + bounds[1] - T.cast(hist[cutoff], T.int32)
            with T.SimdVF():
                m = S.pset(16)
                offsets = T.reinterpret(S.vci(T.int16(0), "int16"), "uint16x128")
                one = S.vdup(T.uint16(1), "uint16", m)
                zero = S.vdup(T.uint16(0), "uint16", m)
                greater_base = S.alloc_var("uint16")
                equal_base = S.alloc_var("uint16")
                greater_base = zero
                equal_base = S.vdup(T.cast(bounds[3], T.uint16), "uint16", m)
                for j in range(span // 128):
                    val = S.vld(ords[j * 128], "NORM")
                    high = S.vcmps(val, T.cast(bounds[2], T.uint16), m, op="gt")
                    same = S.vcmps(val, T.cast(bounds[2], T.uint16), m, op="eq")
                    ids = S.vadds(offsets, T.cast(j * 128, T.uint16), m)
                    greater_rank = S.vadd(greater_base, T.reinterpret(S.vusqz(high, "int16"), "uint16x128"), m)
                    equal_rank = S.vadd(equal_base, T.reinterpret(S.vusqz(same, "int16"), "uint16x128"), m)
                    S.vscatter(ids, top[0], greater_rank, high)
                    S.vscatter(
                        ids, top[0], equal_rank, S.pand(same, S.vcmps(equal_rank, T.uint16(top_k), m, op="lt"), m)
                    )
                    greater_count = S.vcadd(S.vsel(one, zero, high), m)
                    equal_count = S.vcadd(S.vsel(one, zero, same), m)
                    greater_base = S.vadd(greater_base, S.vdupv(T.reinterpret(greater_count, "uint16x128"), m), m)
                    equal_base = S.vadd(equal_base, S.vdupv(T.reinterpret(equal_count, "uint16x128"), m), m)
        else:
            with T.SimdVF():
                m = S.pset(16)
                ix = T.reinterpret(S.vci(T.int16(0), "int16"), "uint16x128")
                for part in range(4):
                    S.vsts(top[part * 128], S.vadds(ix, T.cast(part * 128, T.uint16), m), m, "NORM_B16")

    return rank


def select_program(batch, pages, page_size, context, tail, clip, length_offset=0, *, physical_table_columns=None):
    top_k = 512
    pools = context // 4
    assert pools % 256 == 0 and pools <= 32768
    rank = make_pool_ranker(pools, top_k, clip)

    @T.macro
    def select(X, L, Y, Table, Physical):
        with T.Kernel(batch) as bx:
            raw = T.alloc_shared((pools,), T.bfloat16)
            ords = T.view(raw, (pools,), dtype=T.uint16)
            hist = T.alloc_shared((256,), T.uint16)
            bounds = T.alloc_shared((8,), T.int32)
            top = T.alloc_shared((top_k,), T.uint16)
            pool_ids = T.alloc_shared((top_k,), T.int32)
            visible = L[length_offset + bx] // 4
            rank(X, bx, visible, raw, ords, hist, bounds, top)
            if physical_table_columns is not None:
                pages_ub = T.alloc_shared((context // page_size,), T.int32)
                physical = T.alloc_shared((top_k,), T.int32)
                if visible >= top_k:
                    live_pages = T.ceildiv(visible, 32)
                    T.copy(Table[bx, :live_pages], pages_ub[:live_pages])
                    with T.SimtVF(threads=256):
                        for half in T.serial(2):
                            for lane in T.Parallel(256):
                                j = half * 256 + lane
                                pool = T.cast(top[j], T.int32)
                                physical[j] = pages_ub[T.shift_right(pool, 5)] * 32 + T.bitwise_and(pool, 31)
                    T.copy(physical, Physical[bx, :])
            with T.SimtVF(threads=256):
                for half in T.serial(2):
                    for lane in T.Parallel(256):
                        j = half * 256 + lane
                        pool_ids[j] = T.cast(top[j], T.int32)
            T.copy(pool_ids, Y[bx, :])

    if physical_table_columns is not None:

        @T.prim_func
        def physical_selection(
            X: T.Tensor((batch, pools), T.bfloat16),
            L: T.Tensor((length_offset + batch,), T.int32),
            Table: T.Tensor((batch, physical_table_columns), T.int32),
            Physical: T.Tensor((batch, top_k), T.int32),
            Y: T.Tensor((batch, top_k), T.int32),
        ):
            select(X, L, Y, Table, Physical)

        return physical_selection

    @T.prim_func
    def main(
        X: T.Tensor((batch, pools), T.bfloat16),
        L: T.Tensor((length_offset + batch,), T.int32),
        Y: T.Tensor((batch, top_k), T.int32),
    ):
        select(X, L, Y, Y, Y)

    return main
