# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


@T.macro
def encode_mx16(inp, codes, scales, bk):
    scale32 = T.alloc_shared((bk // 64, 16), T.int32)
    with T.SimdVF():
        mask = S.pset(32, "PAT_ALL")
        half = S.pset(32, "PAT_VL32")
        scalar = S.pset(32, "PAT_VL1")
        hi = S.vcmps(S.vci(T.int32(0), "int32"), T.int32(32), mask, op="ge")
        zero = S.vdup(T.int32(0), "int32", mask)
        bias = S.vdup(T.int32(8), "int32", mask)
        high_exp = S.vdup(T.int32(254), "int32", mask)
        sign = S.vdup(T.uint32(0x80000000), "uint32", mask)
        for r in range(16):
            for g in range(bk // 64):
                packed = S.vld(inp[r, g * 64], "UNPK_B16")
                x = S.vcvt(packed, "float32", mask, part=0)
                ax = S.vabs(x, mask)
                m0 = S.vdupv(S.vcmax(ax, half), mask)
                m1 = S.vdupv(S.vcmax(ax, hi), mask)
                e0 = S.vmax(S.vsub(S.vshrs(T.reinterpret(m0, "int32x64"), 23, mask), bias, mask), zero, mask)
                e1 = S.vmax(S.vsub(S.vshrs(T.reinterpret(m1, "int32x64"), 23, mask), bias, mask), zero, mask)
                ep = S.vor(e0, S.vshls(e1, 8, mask), mask)
                S.vsts(scale32[g, r], ep, scalar, "ONEPT_B32")
                exp = S.vsel(e0, e1, half)
                factor = T.reinterpret(S.vshls(S.vsub(high_exp, exp, mask), 23, mask), "float32x64")
                magnitude = S.vmul(x, factor, mask)
                signed_zero = T.reinterpret(S.vand(T.reinterpret(x, "uint32x64"), sign, mask), "float32x64")
                z = S.vsel(signed_zero, magnitude, S.vcmp(exp, zero, mask, op="eq"))
                z = S.vmins(S.vmaxs(z, T.float32(-448.0), mask), T.float32(448.0), mask)
                fp8 = S.vcvt(z, "float8_e4m3fn", mask)
                S.vsts(codes[r, g * 64], fp8, mask, "PK4_B32")
    with T.SimtVF(threads=128):
        for g, r in T.Parallel(bk // 64, 16):
            scales[g, r] = T.cast(scale32[g, r], T.int16)
