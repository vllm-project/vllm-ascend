# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def pool_norm(tokens):
    @T.prim_func
    def main(
        PoolParts: T.Tensor((tokens, 2048), T.float32),
        Gamma: T.Tensor((128,), T.float32),
        Beta: T.Tensor((128,), T.float32),
        Current: T.Tensor((tokens, 256), T.float32),
    ):
        with T.Kernel(tokens) as row:
            tile = T.alloc_shared((2048,), T.float32)
            raw = T.alloc_shared((256,), T.float32)
            dst = T.alloc_shared((256,), T.float32)
            gamma = T.alloc_shared((128,), T.float32)
            beta = T.alloc_shared((128,), T.float32)
            T.copy(PoolParts[row, :], tile)
            T.copy(Gamma[:], gamma)
            T.copy(Beta[:], beta)
            with T.SimdVF():
                m = S.pset(32, "PAT_ALL")
                for block in range(4):
                    c = block * 64
                    a0 = S.vld(tile[c], "NORM")
                    a1 = S.vadd(a0, S.vld(tile[256 + c], "NORM"), m)
                    a2 = S.vadd(a1, S.vld(tile[512 + c], "NORM"), m)
                    a3 = S.vadd(a2, S.vld(tile[768 + c], "NORM"), m)
                    a4 = S.vadd(a3, S.vld(tile[1024 + c], "NORM"), m)
                    a5 = S.vadd(a4, S.vld(tile[1280 + c], "NORM"), m)
                    a6 = S.vadd(a5, S.vld(tile[1536 + c], "NORM"), m)
                    a7 = S.vadd(a6, S.vld(tile[1792 + c], "NORM"), m)
                    bf = S.vcvt(a7, "bfloat16", m, round="ROUND_A", part=0)
                    S.vsts(raw[c], S.vcvt(bf, "float32", m, part=0), m, "NORM_B32")
            with T.SimdVF():
                m = S.pset(32, "PAT_ALL")
                a = S.vld(raw[0], "NORM")
                b = S.vld(raw[64], "NORM")
                mean = S.vdupv(
                    S.vcadd(S.vadd(S.vmuls(a, T.float32(1.0 / 128), m), S.vmuls(b, T.float32(1.0 / 128), m), m), m), m
                )
                x = S.vsub(a, mean, m)
                y = S.vsub(b, mean, m)
                sx = S.vmuls(S.vmul(x, x, m), T.float32(1.0 / 128), m)
                sy = S.vmuls(S.vmul(y, y, m), T.float32(1.0 / 128), m)
                var = S.vadds(S.vdupv(S.vcadd(S.vadd(sx, sy, m), m), m), T.float32(1e-6), m)
                std = S.vsqrt(var, m)
                for half in range(2):
                    c = half * 64
                    centered = S.vsub(S.vld(raw[c], "NORM"), mean, m)
                    v = S.vdiv(centered, std, m)
                    S.vsts(
                        dst[c], S.vadd(S.vmul(v, S.vld(gamma[c], "NORM"), m), S.vld(beta[c], "NORM"), m), m, "NORM_B32"
                    )
                    S.vsts(dst[128 + c], S.vld(raw[128 + c], "NORM"), m, "NORM_B32")
            T.copy(dst, Current[row, :])

    return main
