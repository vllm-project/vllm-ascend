# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


@T.macro
def encode_native_layout(inp, codes, scales, rows, columns, from_accumulator):
    with T.SimdVF():
        p16 = S.pset(16, "PAT_ALL")
        p32 = S.pset(32, "PAT_ALL")
        p8 = S.pset(8, "PAT_ALL")
        one = S.pset(16, "PAT_VL1")
        lane = S.vci(T.int16(0), "int16")
        data_index = T.reinterpret(S.vshrs(lane, 4, p16), "uint16x128")
        next_lane = T.reinterpret(S.vmins(S.vadds(lane, T.int16(1), p16), T.int16(127), p16), "uint16x128")
        expmask = S.vdup(T.uint16(0x7F80), "uint16", p16)
        zero = S.vdup(T.uint16(0), "uint16", p16)
        invbias = S.vdup(T.uint16(254), "uint16", p16)
        nanbits = S.vdup(T.uint16(0x7FC0), "uint16", p16)
        for r in range(rows):
            for g in range(columns // 256):
                if from_accumulator:
                    # Match FIX's FP32-to-BF16 rounding before the MX scale reduction.
                    base = g * 256
                    a0 = S.vcvt(S.vld(inp[r, base], "NORM"), "bfloat16", p32, round="ROUND_R", part=0)
                    a1 = S.vcvt(S.vld(inp[r, base + 64], "NORM"), "bfloat16", p32, round="ROUND_R", part=0)
                    a2 = S.vcvt(S.vld(inp[r, base + 128], "NORM"), "bfloat16", p32, round="ROUND_R", part=0)
                    a3 = S.vcvt(S.vld(inp[r, base + 192], "NORM"), "bfloat16", p32, round="ROUND_R", part=0)
                    front, _ = S.vdintlv(a0, a1)
                    back, _ = S.vdintlv(a2, a3)
                    xx, yy = S.vdintlv(front, back)
                else:
                    xx, yy = S.vld2(inp[r, g * 256], dist="DINTLV_B16")
                xe = S.vand(T.reinterpret(xx, "uint16x128"), expmask, p16)
                ye = S.vand(T.reinterpret(yy, "uint16x128"), expmask, p16)
                peak = S.vshrs(S.vcgmax(S.vmax(xe, ye, p16), p16), 7, p16)
                delta = T.reinterpret(
                    S.vmaxs(S.vadds(T.reinterpret(peak, "int16x128"), T.int16(-8), p16), T.int16(0), p16), "uint16x128"
                )
                invalid = S.vcmps(peak, T.uint16(255), p16, op="eq")
                exp = S.vsel(S.vdup(T.uint16(255), "uint16", p16), delta, invalid)
                shift = S.vshls(S.vselr(exp, next_lane), 8, p16)
                pair = S.vor(exp, shift, p16)
                S.vsts(scales[g * 4, r], T.reinterpret(pair, "int16x128"), one, "ONEPT_B16", extent=1)
                for j in range(3):
                    gathered = S.vselr(pair, S.vdup(T.uint16(2 * (j + 1)), "uint16", p16))
                    S.vsts(scales[g * 4 + j + 1, r], T.reinterpret(gathered, "int16x128"), one, "ONEPT_B16", extent=1)
                inv = S.vshls(S.vsub(invbias, delta, p16), 7, p16)
                f = S.vsel(zero, inv, S.vcmp(delta, zero, p16, op="eq"))
                f = S.vsel(nanbits, f, invalid)
                factor = T.reinterpret(S.vselr(f, data_index), "bfloat16x128")
                x = S.vmul(xx, factor, p16)
                y = S.vmul(yy, factor, p16)
                u0 = S.vcvt(x, "float32", p16, part=0)
                u1 = S.vcvt(y, "float32", p16, part=0)
                u2 = S.vcvt(x, "float32", p16, part=1)
                u3 = S.vcvt(y, "float32", p16, part=1)
                a = S.vcvt(
                    S.vmins(S.vmaxs(u0, T.float32(-448.0), p32), T.float32(448.0), p32), "float8_e4m3fn", p32, part=0
                )
                b = S.vcvt(
                    S.vmins(S.vmaxs(u1, T.float32(-448.0), p32), T.float32(448.0), p32), "float8_e4m3fn", p32, part=1
                )
                c = S.vcvt(
                    S.vmins(S.vmaxs(u2, T.float32(-448.0), p32), T.float32(448.0), p32), "float8_e4m3fn", p32, part=2
                )
                d = S.vcvt(
                    S.vmins(S.vmaxs(u3, T.float32(-448.0), p32), T.float32(448.0), p32), "float8_e4m3fn", p32, part=3
                )
                v = S.vor(
                    S.vor(T.reinterpret(a, "uint8x256"), T.reinterpret(b, "uint8x256"), p8),
                    S.vor(T.reinterpret(c, "uint8x256"), T.reinterpret(d, "uint8x256"), p8),
                    p8,
                )
                S.vsts(codes[r, g * 256], T.reinterpret(v, "float8_e4m3fnx256"), p8, "NORM_B8")


def quant_program(batch, width, rows, columns, mode):
    assert batch % rows == width % columns == columns % 256 == 0

    @T.prim_func
    def main(
        Value: T.Tensor((batch, width), T.bfloat16),
        C3: T.Tensor((batch, width), T.float8_e4m3fn),
        S3: T.Tensor((width // 64, batch), T.int16),
    ):
        with T.Kernel((batch // rows) * (width // columns)) as task:
            row = (task // (width // columns)) * rows
            col = (task % (width // columns)) * columns
            x = T.alloc_shared((rows, columns), T.bfloat16)
            c = T.alloc_shared((rows, columns), T.float8_e4m3fn)
            s = T.alloc_shared((columns // 64, rows), T.int16)
            T.copy(Value[row : row + rows, col : col + columns], x)
            encode_native_layout(x, c, s, rows, columns, False)
            T.copy(c, C3[row : row + rows, col : col + columns])
            T.copy(s, S3[col // 64 : (col + columns) // 64, row : row + rows])

    return main
