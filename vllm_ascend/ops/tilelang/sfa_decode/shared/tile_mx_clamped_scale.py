# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from .simd_mxrow import encode_mx16


@T.macro
def encode_block(inp, codes, scales, rows, columns, mode):
    with T.SimdVF():
        mask = S.pset(32, "PAT_ALL")
        half = S.pset(32, "PAT_VL32")
        scalar = S.pset(16, "PAT_VL1")
        hi = S.vcmps(S.vci(T.int32(0), "int32"), T.int32(32), mask, op="ge")
        zero = S.vdup(T.int32(0), "int32", mask)
        bias = S.vdup(T.int32(8), "int32", mask)
        peak_exp = S.vdup(T.int32(254), "int32", mask)
        for r in range(rows):
            for g in range(columns // 64):
                x = S.vcvt(S.vld(inp[r, g * 64], "UNPK_B16"), "float32", mask, part=0)
                ax = S.vabs(x, mask)
                m0 = S.vdupv(S.vcmax(ax, half), mask)
                m1 = S.vdupv(S.vcmax(ax, hi), mask)
                e0 = S.vmax(S.vsub(S.vshrs(T.reinterpret(m0, "int32x64"), 23, mask), bias, mask), zero, mask)
                e1 = S.vmax(S.vsub(S.vshrs(T.reinterpret(m1, "int32x64"), 23, mask), bias, mask), zero, mask)
                scale = S.vor(e0, S.vshls(e1, 8, mask), mask)
                S.vsts(scales[g, r], T.reinterpret(scale, "int16x128"), scalar, "ONEPT_B16", extent=1)
                exp = S.vsel(e0, e1, half)
                if mode == 1 or mode == 5:
                    mul = T.reinterpret(S.vshls(S.vsub(peak_exp, exp, mask), 23, mask), "float32x64")
                    factor = T.reinterpret(
                        S.vsel(zero, T.reinterpret(mul, "int32x64"), S.vcmp(exp, zero, mask, op="eq")), "float32x64"
                    )
                    z = S.vmul(x, factor, mask)
                    if mode == 5:
                        z = S.vmins(S.vmaxs(z, T.float32(-448.0), mask), T.float32(448.0), mask)
                else:
                    factor = T.reinterpret(S.vshls(S.vsub(peak_exp, exp, mask), 23, mask), "float32x64")
                    magnitude = S.vmul(x, factor, mask)
                    sign = S.vdup(T.uint32(0x80000000), "uint32", mask)
                    signed_zero = T.reinterpret(S.vand(T.reinterpret(x, "uint32x64"), sign, mask), "float32x64")
                    z = S.vsel(signed_zero, magnitude, S.vcmp(exp, zero, mask, op="eq"))
                    z = S.vmins(S.vmaxs(z, T.float32(-448.0), mask), T.float32(448.0), mask)
                S.vsts(codes[r, g * 64], S.vcvt(z, "float8_e4m3fn", mask, sat=True), mask, "PK4_B32")


def quant_program(batch, width, rows, columns, mode):
    assert width % columns == batch % rows == 0
    assert mode or rows == 16

    @T.prim_func
    def main(
        Value: T.Tensor((batch, width), T.bfloat16),
        C3: T.Tensor((batch, width), T.float8_e4m3fn),
        S3: T.Tensor((width // 64, batch), T.int16),
    ):
        with T.Kernel((batch // rows) * (width // columns)) as task:
            row = (task // (width // columns)) * rows
            k = (task % (width // columns)) * columns
            inp = T.alloc_shared((rows, columns), T.bfloat16)
            code = T.alloc_shared((rows, columns), T.float8_e4m3fn)
            scales = T.alloc_shared((columns // 64, rows), T.int16)
            T.copy(Value[row : row + rows, k : k + columns], inp)
            if mode == 0:
                encode_mx16(inp, code, scales, columns)
            else:
                encode_block(inp, code, scales, rows, columns, mode)
            T.copy(code, C3[row : row + rows, k : k + columns])
            T.copy(scales, S3[k // 64 : (k + columns) // 64, row : row + rows])

    return main
