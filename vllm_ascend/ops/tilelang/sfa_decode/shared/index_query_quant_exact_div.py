# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


@T.macro
def _quantize_heads(src, code, sf, chunk, weights=None):
    with T.SimdVF():
        mask = S.pset(32, "PAT_ALL")
        scalar = S.pset(32, "PAT_VL1")
        one = S.vdup(T.float32(1.0), "float32", mask)
        for h in range(chunk):
            a = S.vcvt(S.vld(src[h, 0], "UNPK_B16"), "float32", mask, part=0)
            b = S.vcvt(S.vld(src[h, 64], "UNPK_B16"), "float32", mask, part=0)
            mx = S.vdupv(S.vcmax(S.vmax(S.vabs(a, mask), S.vabs(b, mask), mask), mask), mask)
            scale = S.vmuls(mx, T.float32(1.0 / 448.0), mask)
            outscale = S.vsel(one, scale, S.vcmps(scale, T.float32(0.0), mask, op="eq"))
            if weights is not None:
                gain = S.vcvt(S.vld(weights[h], "BRC_B16"), "float32", mask, part=0)
                S.vsts(sf[h], S.vmul(outscale, gain, mask), scalar, "ONEPT_B32")
            else:
                S.vsts(sf[h], outscale, scalar, "ONEPT_B32")
            mult = S.vsel(
                S.vdup(T.float32(2.0**64), "float32", mask),
                one,
                S.vcmps(mx, T.float32(2.0**-116), mask, op="lt"),
            )
            div = S.vmuls(S.vmul(mx, mult, mask), T.float32(1.0 / 448.0), mask)
            div = S.vsel(one, div, S.vcmps(mx, T.float32(0.0), mask, op="eq"))
            qa = S.vcvt(S.vdiv(S.vmul(a, mult, mask), div, mask, precision="ftz_true"), "float8_e4m3fn", mask)
            qb = S.vcvt(S.vdiv(S.vmul(b, mult, mask), div, mask, precision="ftz_true"), "float8_e4m3fn", mask)
            S.vsts(code[h, 0], qa, mask, "PK4_B32")
            S.vsts(code[h, 64], qb, mask, "PK4_B32")


def program(batch):
    chunk = 8

    @T.prim_func
    def main(
        IQRaw: T.Tensor((batch, 4096), T.bfloat16),
        IQ: T.Tensor((batch, 32, 128), T.float8_e4m3fn),
        IQS: T.Tensor((batch, 32), T.float32),
    ):
        with T.Kernel(batch * (32 // chunk)) as block:
            row = block // 4
            start = (block % 4) * chunk
            src = T.alloc_shared((chunk, 128), T.bfloat16)
            code = T.alloc_shared((chunk, 128), T.float8_e4m3fn)
            sf = T.alloc_shared((chunk,), T.float32)
            T.copy(IQRaw[row, start * 128 : (start + chunk) * 128], T.view(src, (chunk * 128,)))
            _quantize_heads(src, code, sf, chunk)
            T.copy(code, IQ[row, start : start + chunk, :])
            T.copy(sf, IQS[row, start : start + chunk])

    return main
