# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def select_program(batch, pages, page_size, context, tail, length_offset=0):
    pools = context // 4
    groups = pools // 128
    assert pools % 256 == 0 and pools <= 32768

    @T.prim_func
    def main(
        X: T.Tensor((batch, pools), T.bfloat16),
        L: T.Tensor((length_offset + batch,), T.int32),
        Y: T.Tensor((batch, 512), T.int32),
    ):
        with T.Kernel(batch) as bx:
            raw = T.alloc_shared((pools,), T.bfloat16)
            ords = T.view(raw, (pools,), dtype=T.uint16)
            hist = T.alloc_shared((256,), T.uint16)
            bounds = T.alloc_shared((8,), T.int32)
            ge = T.alloc_shared((groups,), T.uint32)
            eq = T.alloc_shared((groups,), T.uint32)
            top = T.alloc_shared((512,), T.uint16)
            pool_ids = T.alloc_shared((512,), T.int32)
            visible = L[length_offset + bx] // 4
            if visible >= 512:
                T.copy(X[bx, :], raw)
                with T.SimdVF():
                    m = S.pset(16)
                    sign = S.vdup(T.uint16(32768), "uint16", m)
                    zeros = S.vdup(T.uint16(0), "uint16", m)
                    lanes = T.reinterpret(S.vci(T.int16(0), "int16"), "uint16x128")
                    for j in range(pools // 256):
                        a = S.vld(ords[j * 256], "NORM")
                        b = S.vld(ords[j * 256 + 128], "NORM")
                        ka = S.vsel(S.vnot(a, m), S.vxor(a, sign, m), S.vcmps(a, T.uint16(32768), m, op="ge"))
                        kb = S.vsel(S.vnot(b, m), S.vxor(b, sign, m), S.vcmps(b, T.uint16(32768), m, op="ge"))
                        ia = S.vadds(lanes, T.cast(j * 256, T.uint16), m)
                        ib = S.vadds(lanes, T.cast(j * 256 + 128, T.uint16), m)
                        oa = S.vsel(ka, zeros, S.vcmps(ia, T.cast(visible, T.uint16), m, op="lt"))
                        ob = S.vsel(kb, zeros, S.vcmps(ib, T.cast(visible, T.uint16), m, op="lt"))
                        S.vsts(ords[j * 256], oa, m, "NORM_B16")
                        S.vsts(ords[j * 256 + 128], ob, m, "NORM_B16")
                for digit in T.unroll(2):
                    with T.SimdVF():
                        m = S.pset(16)
                        bytes_mask = S.pset(8)
                        sign = S.vdup(T.uint16(32768), "uint16", m)
                        zeros = S.vdup(T.uint16(0), "uint16", m)
                        c0 = S.alloc_var("uint16")
                        c1 = S.alloc_var("uint16")
                        c0 = S.vdup(T.uint16(0), "uint16", m)
                        c1 = S.vdup(T.uint16(0), "uint16", m)
                        lanes = T.reinterpret(S.vci(T.int16(0), "int16"), "uint16x128")
                        for j in range(pools // 256):
                            a = S.vld(ords[j * 256], "NORM")
                            b = S.vld(ords[j * 256 + 128], "NORM")
                            lo, hi = S.vdintlv(T.reinterpret(a, "uint8x256"), T.reinterpret(b, "uint8x256"))
                            if digit == 0:
                                S.chistv2(c0, hi, bytes_mask, bin=0)
                                S.chistv2(c1, hi, bytes_mask, bin=1)
                            else:
                                yes = S.vcmps(hi, T.cast(bounds[0], T.uint8), bytes_mask, op="eq")
                                S.chistv2(c0, lo, yes, bin=0)
                                S.chistv2(c1, lo, yes, bin=1)
                        S.vsts(hist[0], c0, m, "NORM_B16")
                        S.vsts(hist[128], c1, m, "NORM_B16")
                    with T.SimtVF(threads=256):
                        votes = T.alloc_fragment((1, 256), T.int32)
                        total = T.alloc_fragment((1,), T.int32)
                        for t in T.Parallel(256):
                            target = T.if_then_else(digit == 0, pools - 511, bounds[1])
                            votes[0, t] = T.cast(T.cast(hist[t], T.int32) < target, T.int32)
                        T.reduce_sum(votes, total, dim=1)
                        for t in T.Parallel(1):
                            cutoff = total[t]
                            if digit == 0:
                                bounds[0] = cutoff
                                prev = T.if_then_else(cutoff == 0, 0, T.cast(hist[T.max(cutoff - 1, 0)], T.int32))
                                bounds[1] = pools - 511 - prev
                            else:
                                bounds[2] = bounds[0] * 256 + cutoff
                    # The second pass reads the ordered BF16 keys produced by the first.
                with T.SimdVF():
                    m = S.pset(16)
                    scalar = S.pset(32, "PAT_VL1")
                    one = S.vdup(T.uint16(1), "uint16", m)
                    zero = S.vdup(T.uint16(0), "uint16", m)
                    for j in range(groups):
                        val = S.vld(ords[j * 128], "NORM")
                        high = S.vcmps(val, T.cast(bounds[2], T.uint16), m, op="gt")
                        same = S.vcmps(val, T.cast(bounds[2], T.uint16), m, op="eq")
                        S.vsts(ge[j], S.vcadd(S.vsel(one, zero, high), m), scalar, "ONEPT_B32")
                        S.vsts(eq[j], S.vcadd(S.vsel(one, zero, same), m), scalar, "ONEPT_B32")
                with T.SimtVF(threads=32):
                    for t in T.Parallel(1):
                        gsum = T.alloc_var(T.uint32, init=0)
                        esum = T.alloc_var(T.uint32, init=0)
                        for j in T.serial(groups):
                            g = ge[j]
                            e = eq[j]
                            ge[j] = gsum
                            eq[j] = esum
                            gsum += g
                            esum += e
                        bounds[3] = T.cast(gsum, T.int32)
                with T.SimdVF():
                    m = S.pset(16)
                    lanes = T.reinterpret(S.vci(T.int16(0), "int16"), "uint16x128")
                    for j in range(groups):
                        v = S.vld(ords[j * 128], "NORM")
                        high = S.vcmps(v, T.cast(bounds[2], T.uint16), m, op="gt")
                        same = S.vcmps(v, T.cast(bounds[2], T.uint16), m, op="eq")
                        ri = S.vadds(T.reinterpret(S.vusqz(high, "int16"), "uint16x128"), T.cast(ge[j], T.uint16), m)
                        ti = S.vadds(
                            T.reinterpret(S.vusqz(same, "int16"), "uint16x128"),
                            T.cast(eq[j] + T.cast(bounds[3], T.uint32), T.uint16),
                            m,
                        )
                        ids = S.vadds(lanes, T.cast(j * 128, T.uint16), m)
                        S.vscatter(ids, top[0], ri, high)
                        S.vscatter(ids, top[0], ti, S.pand(same, S.vcmps(ti, T.uint16(512), m, op="lt"), m))
            else:
                with T.SimdVF():
                    m = S.pset(16)
                    ix = T.reinterpret(S.vci(T.int16(0), "int16"), "uint16x128")
                    for part in range(4):
                        S.vsts(top[part * 128], S.vadds(ix, T.cast(part * 128, T.uint16), m), m, "NORM_B16")
            with T.SimtVF(threads=256):
                for half in T.serial(2):
                    for lane in T.Parallel(256):
                        j = half * 256 + lane
                        pool_ids[j] = T.cast(top[j], T.int32)
            T.copy(pool_ids, Y[bx, :])

    return main
