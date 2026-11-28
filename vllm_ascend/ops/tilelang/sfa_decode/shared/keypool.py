# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="valid-type"

import tilelang.ascend.language as T  # type: ignore[import-not-found]
from tilelang.ascend.language import simd as S  # type: ignore[import-not-found]


@T.macro
def pool_probability(value, denominator, mask):
    probability = S.vcvt(S.vdiv(value, denominator, mask), "bfloat16", mask, round="ROUND_A", part=0)
    return S.vcvt(probability, "float32", mask, part=0)


@T.macro
def pool_four_keys(history, first, current, current_row, ape, bf, packed, scale):
    with T.SimdVF():
        mask = S.pset(32, "PAT_ALL")
        for half in range(2):
            start = half * 64
            g0 = S.vadd(S.vld(history[first, 128 + start], "NORM"), S.vld(ape[0, start], "NORM"), mask)
            g1 = S.vadd(S.vld(history[first + 1, 128 + start], "NORM"), S.vld(ape[1, start], "NORM"), mask)
            g2 = S.vadd(S.vld(history[first + 2, 128 + start], "NORM"), S.vld(ape[2, start], "NORM"), mask)
            g3 = S.vadd(S.vld(current[current_row, 128 + start], "NORM"), S.vld(ape[3, start], "NORM"), mask)
            mx = S.vmax(S.vmax(g0, g1, mask), S.vmax(g2, g3, mask), mask)
            e0 = S.vexpdif(g0, mx, mask)
            e1 = S.vexpdif(g1, mx, mask)
            e2 = S.vexpdif(g2, mx, mask)
            e3 = S.vexpdif(g3, mx, mask)
            den = S.vadd(S.vadd(e0, e2, mask), S.vadd(e1, e3, mask), mask)
            p0 = pool_probability(e0, den, mask)
            p1 = pool_probability(e1, den, mask)
            p2 = pool_probability(e2, den, mask)
            p3 = pool_probability(e3, den, mask)
            value = S.vmul(S.vld(history[first, start], "NORM"), p0, mask)
            value = S.vadd(value, S.vmul(S.vld(history[first + 1, start], "NORM"), p1, mask), mask)
            value = S.vadd(value, S.vmul(S.vld(history[first + 2, start], "NORM"), p2, mask), mask)
            value = S.vadd(value, S.vmul(S.vld(current[current_row, start], "NORM"), p3, mask), mask)
            S.vsts(bf[start], S.vcvt(value, "bfloat16", mask, round="ROUND_A", part=0), mask, "PK_B32")
    with T.SimdVF():
        mask = S.pset(32, "PAT_ALL")
        scalar = S.pset(32, "PAT_VL1")
        a = S.vcvt(S.vld(bf[0], "UNPK_B16"), "float32", mask, part=0)
        b = S.vcvt(S.vld(bf[64], "UNPK_B16"), "float32", mask, part=0)
        peak = S.vdupv(S.vcmax(S.vmax(S.vabs(a, mask), S.vabs(b, mask), mask), mask), mask)
        unit = S.vdup(T.float32(1.0), "float32", mask)
        scaled = S.vdiv(peak, S.vdup(T.float32(448.0), "float32", mask), mask)
        sf = S.vsel(unit, scaled, S.vcmps(peak, T.float32(0.0), mask, op="eq"))
        S.vsts(scale[0], sf, scalar, "ONEPT_B32")
        S.vsts(
            packed[0],
            S.vcvt(S.vdiv(a, sf, mask, precision="ftz_true"), "float8_e4m3fn", mask),
            mask,
            "PK4_B32",
        )
        S.vsts(
            packed[64],
            S.vcvt(S.vdiv(b, sf, mask, precision="ftz_true"), "float8_e4m3fn", mask),
            mask,
            "PK4_B32",
        )


@T.macro
def normalize_current(raw, gamma, beta, dst, epsilon, rollback):
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
        var = S.vadds(S.vdupv(S.vcadd(S.vadd(sx, sy, m), m), m), T.float32(epsilon), m)
        if rollback:
            # The retained rollback state uses the native refined reciprocal and fused affine transform.
            root = S.vsqrt(var, m, precision="ftz_true")
            one = S.vdup(T.float32(1.0), "float32", m)
            approx = S.vdiv(one, root, m, precision="exact")
            product = S.vmul(approx, var, m)
            error_product = S.alloc_var("float32")
            error_product = S.vmuls(product, T.float32(-1.0), m)
            S.vmula(error_product, approx, var, m)
            minus_approx = S.vmuls(approx, T.float32(-1.0), m)
            residual = S.alloc_var("float32")
            residual = one
            S.vmula(residual, minus_approx, product, m)
            S.vmula(residual, minus_approx, error_product, m)
            inv = S.alloc_var("float32")
            inv = approx
            S.vmula(inv, S.vmuls(approx, T.float32(0.5), m), residual, m)
        else:
            std = S.vsqrt(var, m)
        for half in range(2):
            c = half * 64
            centered = S.vsub(S.vld(raw[c], "NORM"), mean, m)
            weight = S.vld(gamma[c], "NORM")
            bias = S.vld(beta[c], "NORM")
            if rollback:
                value = S.alloc_var("float32")
                value = bias
                S.vmula(value, S.vmul(centered, inv, m), weight, m)
                S.vsts(dst[c], value, m, "NORM_B32")
            else:
                value = S.vdiv(centered, std, m, precision="ftz_true")
                S.vsts(dst[c], S.vadd(S.vmul(value, weight, m), bias, m), m, "NORM_B32")
            S.vsts(dst[128 + c], S.vld(raw[128 + c], "NORM"), m, "NORM_B32")


@T.macro
def normalize_parts(Parts, Weight, Bias, row, dst, count, epsilon, rollback):
    tile = T.alloc_shared((count, 256), T.float32)
    raw = T.alloc_shared((256,), T.float32)
    gamma = T.alloc_shared((128,), T.float32)
    beta = T.alloc_shared((128,), T.float32)
    T.copy(Parts[row, :, :], tile)
    T.copy(Weight[:], gamma)
    T.copy(Bias[:], beta)
    with T.SimdVF():
        mask = S.pset(32)
        for block in range(4):
            col = block * 64
            total = S.alloc_var("float32")
            total = S.vld(tile[0, col], "NORM")
            for k in T.unroll(count - 1):
                total = S.vadd(total, S.vld(tile[k + 1, col], "NORM"), mask)
            rounded = S.vcvt(total, "bfloat16", mask, round="ROUND_A", part=0)
            S.vsts(raw[col], S.vcvt(rounded, "float32", mask, part=0), mask, "NORM_B32")
    normalize_current(raw, gamma, beta, dst, epsilon, rollback)


def pool_norm(tokens, parts, *, rollback=False):
    @T.prim_func
    def main(
        PoolParts: T.Tensor((tokens, parts, 256), T.float32),
        Gamma: T.Tensor((128,), T.float32),
        Beta: T.Tensor((128,), T.float32),
        Current: T.Tensor((tokens, 256), T.float32),
    ):
        with T.Kernel(tokens) as row:
            dst = T.alloc_shared((256,), T.float32)
            normalize_parts(PoolParts, Gamma, Beta, row, dst, parts, 1e-6, rollback)
            T.copy(dst, Current[row, :])

    return main
