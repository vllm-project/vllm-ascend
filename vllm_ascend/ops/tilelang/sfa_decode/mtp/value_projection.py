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


def project_and_quantize(batch, row_tile, *, balanced=False, input_major=False):
    heads, width, values = 64, 512, 256
    panel, k_tile = 256, 64
    weight_shape = (heads, width, values) if input_major else (heads, values, width)
    half = row_tile // 2

    @T.macro
    def load_panel(task, offset, Latent3, WV, x, w):
        row = task // heads * row_tile
        head = task % heads
        start = head * width + offset
        T.copy(Latent3[row : row + row_tile, start : start + panel], x)
        if input_major:
            T.copy(WV[head, offset : offset + panel, :], w, transpose=True)
        else:
            T.copy(WV[head, :, offset : offset + panel], w)

    @T.macro
    def accumulate_panel(x, w, a0, b0, acc, first):
        for k in T.Pipelined(panel // k_tile, num_stages=2):
            T.copy(x[:, k * k_tile : (k + 1) * k_tile], a0)
            T.copy(w[:, k * k_tile : (k + 1) * k_tile], b0)
            T.gemm(a0, b0, acc, transpose_B=True, clear_accum=first and k == 0)

    @T.macro
    def project(task, Latent3, WV, x, w, a0, b0, acc):
        for outer in T.Pipelined(width // panel, num_stages=2):
            load_panel(task, outer * panel, Latent3, WV, x, w)
            accumulate_panel(x, w, a0, b0, acc, outer == 0)

    @T.macro
    def store(task, sid, partial, codes, scales, C3, S3):
        row = task // heads * row_tile + sid * half
        col = task % heads * values
        encode_native_layout(partial, codes, scales, half, values, True)
        T.copy(codes, C3[row : row + half, col : col + values])
        T.copy(scales, S3[col // 64 : (col + values) // 64, row : row + half])

    @T.prim_func
    def main(
        Latent3: T.Tensor((batch, heads * width), T.bfloat16),
        WV: T.Tensor(weight_shape, T.bfloat16),
        C3: T.Tensor((batch, heads * values), T.float8_e4m3fn),
        S3: T.Tensor((heads * values // 64, batch), T.int16),
    ):
        with T.MixedKernel(28 if balanced else (batch // row_tile) * heads // 2, sids=2) as (cta, sid):
            x = T.alloc_l1((row_tile, panel), T.bfloat16)
            w = T.alloc_l1((values, panel), T.bfloat16)
            a0 = T.alloc_l0a((row_tile, k_tile), T.bfloat16)
            b0 = T.alloc_l0b((values, k_tile), T.bfloat16)
            acc = T.alloc_l0c((row_tile, values), T.float32)
            partial = T.alloc_shared((half, values), T.float32)
            codes = T.alloc_shared((half, values), T.float8_e4m3fn)
            scales = T.alloc_shared((values // 64, half), T.int16)
            if balanced:
                x1 = T.alloc_l1((row_tile, panel), T.bfloat16)
                w1 = T.alloc_l1((values, panel), T.bfloat16)
                # Refill the consumed first panel from the next head while Cube uses the second.
                load_panel(cta, 0, Latent3, WV, x, w)
                load_panel(cta, panel, Latent3, WV, x1, w1)
                accumulate_panel(x, w, a0, b0, acc, True)
                load_panel(cta + 28, 0, Latent3, WV, x, w)
                accumulate_panel(x1, w1, a0, b0, acc, False)
                load_panel(cta + 28, panel, Latent3, WV, x1, w1)
                T.dual_copy(acc, partial)
                accumulate_panel(x, w, a0, b0, acc, True)
                if cta < 8:
                    load_panel(cta + 56, 0, Latent3, WV, x, w)
                store(cta, sid, partial, codes, scales, C3, S3)
                accumulate_panel(x1, w1, a0, b0, acc, False)
                if cta < 8:
                    load_panel(cta + 56, panel, Latent3, WV, x1, w1)
                T.dual_copy(acc, partial)
                if cta < 8:
                    accumulate_panel(x, w, a0, b0, acc, True)
                    accumulate_panel(x1, w1, a0, b0, acc, False)
                store(cta + 28, sid, partial, codes, scales, C3, S3)
                if cta < 8:
                    T.dual_copy(acc, partial)
                    store(cta + 56, sid, partial, codes, scales, C3, S3)
            else:
                project(2 * cta, Latent3, WV, x, w, a0, b0, acc)
                T.dual_copy(acc, partial)
                project(2 * cta + 1, Latent3, WV, x, w, a0, b0, acc)
                store(2 * cta, sid, partial, codes, scales, C3, S3)
                T.dual_copy(acc, partial)
                store(2 * cta + 1, sid, partial, codes, scales, C3, S3)

    return main
