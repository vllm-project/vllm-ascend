# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T  # type: ignore[import-not-found]
from tilelang.ascend.language import simd as S  # type: ignore[import-not-found]


@T.macro
def normalize_qkv(QKV, gamma, row, out, eps, width, offset):
    partial = T.alloc_shared((64,), T.float32)
    with T.SimtVF(threads=64):
        for lane in T.Parallel(64):
            partial[lane] = 0.0
    with T.SimdVF():
        mask = S.pset(32, "PAT_ALL")
        scalar = S.pset(32, "PAT_VL1")
        if width == 1536:
            for j in range(8):
                x = S.vcvt(S.vld(QKV[row, j * 64], "UNPK_B16"), "float32", mask, part=0)
                t = S.vcvt(S.vld(QKV[row, (16 + j) * 64], "UNPK_B16"), "float32", mask, part=0)
                merged = S.vadd(S.vmul(x, x, mask), S.vmul(t, t, mask), mask)
                S.vsts(partial[j], S.vcadd(merged, mask), scalar, "ONEPT_B32")
            for j in range(8):
                x = S.vcvt(S.vld(QKV[row, (8 + j) * 64], "UNPK_B16"), "float32", mask, part=0)
                S.vsts(partial[8 + j], S.vcadd(S.vmul(x, x, mask), mask), scalar, "ONEPT_B32")
        else:
            for j in range(4):
                x = S.vcvt(S.vld(QKV[row, offset + j * 64], "UNPK_B16"), "float32", mask, part=0)
                y = S.vcvt(S.vld(QKV[row, offset + (4 + j) * 64], "UNPK_B16"), "float32", mask, part=0)
                pair = S.vadd(S.vmul(x, x, mask), S.vmul(y, y, mask), mask)
                S.vsts(partial[j], S.vcadd(pair, mask), scalar, "ONEPT_B32")
        S.mem_bar("VST_VLD")
        total = S.vdupv(S.vcadd(S.vld(partial[0], "NORM"), mask), mask)
        mean = S.vadds(S.vmuls(total, T.float32(1.0 / width), mask), T.float32(eps), mask)
        one = S.vdup(T.float32(1.0), "float32", mask)
        recip = S.alloc_var("float32")
        recip = S.vdiv(one, mean, mask)
        estimate = S.vsqrt(recip, mask)
        t = S.vmul(S.vmuls(mean, T.float32(-0.5), mask), estimate, mask)
        correction = S.alloc_var("float32")
        correction = S.vdup(T.float32(1.5), "float32", mask)
        S.vmula(correction, t, estimate, mask)
        refined = S.alloc_var("float32")
        refined = S.vmul(estimate, correction, mask)
        residual = S.alloc_var("float32")
        residual = S.vdup(T.float32(1.0), "float32", mask)
        S.vmula(residual, S.vmuls(mean, T.float32(-1.0), mask), recip, mask)
        S.vmula(recip, S.vmuls(refined, T.float32(-1.0), mask), refined, mask)
        S.vmula(residual, mean, recip, mask)
        step = S.vmul(residual, refined, mask)
        S.vmula(refined, step, S.vdup(T.float32(0.5), "float32", mask), mask)
        inf = T.reinterpret(T.uint32(0x7F800000), T.float32)
        inv = S.vsel(S.vdup(T.float32(0.0), "float32", mask), refined, S.vcmps(mean, inf, mask, op="eq"))
        inv = S.vsel(S.vdup(inf, "float32", mask), inv, S.vcmps(mean, T.float32(0.0), mask, op="eq"))
        for i in range(width // 64):
            x = S.vcvt(S.vld(QKV[row, offset + i * 64], "UNPK_B16"), "float32", mask, part=0)
            w = S.vcvt(S.vld(gamma[i * 64], "UNPK_B16"), "float32", mask, part=0)
            val = S.vmul(S.vmul(x, inv, mask), w, mask)
            S.vsts(out[row, i * 64], S.vcvt(val, "bfloat16", mask, part=0), mask, "PK_B32")
