# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def state_norm(tokens, rows, output_rows):
    @T.prim_func
    def main(
        OC: T.Tensor((output_rows, 16384), T.float8_e4m3fn),
        Gamma: T.Tensor((128,), T.float32),
        Beta: T.Tensor((128,), T.float32),
        Projected: T.Tensor((tokens, 256), T.float32),
    ):
        P = T.Tensor((rows, 7 * 256), T.float32, OC.data)
        with T.Kernel(tokens) as bx:
            src = T.alloc_shared((256,), T.bfloat16)
            gamma = T.alloc_shared((128,), T.float32)
            beta = T.alloc_shared((128,), T.float32)
            dst = T.alloc_shared((256,), T.float32)
            parts = T.alloc_shared((7 * 256,), T.float32)
            T.copy(P[bx, :], parts)
            with T.SimtVF(threads=256):
                for i in T.Parallel(256):
                    v = T.alloc_var(T.float32)
                    v = parts[i]
                    for j in T.unroll(6):
                        v += parts[(j + 1) * 256 + i]
                    src[i] = T.cast(v, T.bfloat16)
            T.copy(Gamma[:], gamma)
            T.copy(Beta[:], beta)
            with T.SimdVF():
                m = S.pset(32, "PAT_ALL")
                a = S.vcvt(S.vld(src[0], "UNPK_B16"), "float32", m, part=0)
                b = S.vcvt(S.vld(src[64], "UNPK_B16"), "float32", m, part=0)
                avg = S.vdupv(
                    S.vcadd(S.vadd(S.vmuls(a, T.float32(1.0 / 128), m), S.vmuls(b, T.float32(1.0 / 128), m), m), m), m
                )
                x = S.vsub(a, avg, m)
                y = S.vsub(b, avg, m)
                v = S.vdupv(
                    S.vcadd(
                        S.vadd(
                            S.vmuls(S.vmul(x, x, m), T.float32(1.0 / 128), m),
                            S.vmuls(S.vmul(y, y, m), T.float32(1.0 / 128), m),
                            m,
                        ),
                        m,
                    ),
                    m,
                )
                variance = S.vadds(v, T.float32(1e-6), m)
                root = S.vsqrt(variance, m, precision="ftz_true")
                one = S.vdup(T.float32(1.0), "float32", m)
                approx = S.vdiv(one, root, m, precision="exact")
                product = S.vmul(approx, variance, m)
                error_product = S.alloc_var("float32")
                error_product = S.vmuls(product, T.float32(-1.0), m)
                S.vmula(error_product, approx, variance, m)
                minus_approx = S.vmuls(approx, T.float32(-1.0), m)
                residual = S.alloc_var("float32")
                residual = one
                S.vmula(residual, minus_approx, product, m)
                S.vmula(residual, minus_approx, error_product, m)
                inv = S.alloc_var("float32")
                inv = approx
                S.vmula(inv, S.vmuls(approx, T.float32(0.5), m), residual, m)
                left = S.alloc_var("float32")
                left = S.vld(beta[0], "NORM")
                S.vmula(left, S.vmul(x, inv, m), S.vld(gamma[0], "NORM"), m)
                right = S.alloc_var("float32")
                right = S.vld(beta[64], "NORM")
                S.vmula(right, S.vmul(y, inv, m), S.vld(gamma[64], "NORM"), m)
                S.vsts(dst[0], left, m, "NORM_B32")
                S.vsts(dst[64], right, m, "NORM_B32")
                for k in range(2):
                    S.vsts(
                        dst[128 + k * 64],
                        S.vcvt(S.vld(src[128 + k * 64], "UNPK_B16"), "float32", m, part=0),
                        m,
                        "NORM_B32",
                    )
            T.copy(dst, Projected[bx, :])

    return main
