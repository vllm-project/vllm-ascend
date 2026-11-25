# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def update_program(requests, pages, ring_pages, tokens, retained_width=6, *, ring_table_stride=1):
    capacity = 9
    kind = "quant_div"

    @T.prim_func
    def main(
        Current: T.Tensor((tokens, 256), T.float32),
        Projected: T.Tensor((tokens, 256), T.float32),
        Pos: T.Tensor((tokens,), T.int64),
        RingPage: T.StridedTensor((requests,), (ring_table_stride,), T.int32),
        Ring: T.StridedTensor((ring_pages, capacity, 256), (289 * 256, 256, 1), T.float32),
        Starts: T.Tensor((requests + 1,), T.int32),
        IndexSlots: T.Tensor((tokens,), T.int64),
        Writes: T.Tensor((requests, retained_width), T.int64),
        Ape: T.Tensor((4, 128), T.float32),
        K: T.Tensor((pages, 34, 128), T.float8_e4m3fn),
        KS: T.StridedTensor((pages, 32), (1088, 1), T.float32),
    ):
        with T.Kernel(requests) as bx:
            ring = RingPage[bx]
            begin = Starts[bx]
            live = Starts[bx + 1] - begin
            window = T.alloc_shared((9, 256), T.float32)
            # Keep the speculative rows separate until the write indices select the retained tail.
            rollback = T.alloc_shared((6, 256), T.float32)
            ape = T.alloc_shared((4, 128), T.float32)
            bf = T.alloc_shared((128,), T.bfloat16)
            packed = T.alloc_shared((128,), T.float8_e4m3fn)
            scale = T.alloc_shared((8,), T.float32)
            if live > 0:
                first_position = T.cast(Pos[begin], T.int32)
                history = first_position % 4
                T.copy(Ape[:, :], ape)
                for j in T.serial(3):
                    if 3 - j <= history:
                        T.copy(Ring[T.max(ring, 0), (first_position - 3 + j) % capacity, :], window[j, :])
                for j in T.serial(live):
                    T.copy(Current[begin + j, :], window[3 + j, :])
                    T.copy(Projected[begin + j, :], rollback[j, :])
                for j in T.serial(live):
                    if retained_width - live + j >= 0:
                        dest = Writes[bx, retained_width - live + j]
                        if dest >= 0 and dest < ring_pages * 289 and dest % 289 < capacity:
                            T.copy(rollback[j, :], Ring[dest // 289, dest % 289, :])
                    slot = IndexSlots[begin + j]
                    if slot >= 0 and slot < pages * 32:
                        with T.SimdVF():
                            mask = S.pset(32, "PAT_ALL")
                            for half in range(2):
                                start = half * 64
                                g0 = S.vadd(S.vld(window[j, 128 + start], "NORM"), S.vld(ape[0, start], "NORM"), mask)
                                g1 = S.vadd(
                                    S.vld(window[j + 1, 128 + start], "NORM"), S.vld(ape[1, start], "NORM"), mask
                                )
                                g2 = S.vadd(
                                    S.vld(window[j + 2, 128 + start], "NORM"), S.vld(ape[2, start], "NORM"), mask
                                )
                                g3 = S.vadd(
                                    S.vld(window[j + 3, 128 + start], "NORM"), S.vld(ape[3, start], "NORM"), mask
                                )
                                mx = S.vmax(S.vmax(g0, g1, mask), S.vmax(g2, g3, mask), mask)
                                e0 = S.vexpdif(g0, mx, mask)
                                e1 = S.vexpdif(g1, mx, mask)
                                e2 = S.vexpdif(g2, mx, mask)
                                e3 = S.vexpdif(g3, mx, mask)
                                den = S.vadd(S.vadd(e0, e2, mask), S.vadd(e1, e3, mask), mask)
                                p0 = S.vcvt(
                                    S.vcvt(S.vdiv(e0, den, mask), "bfloat16", mask, round="ROUND_A", part=0),
                                    "float32",
                                    mask,
                                    part=0,
                                )
                                p1 = S.vcvt(
                                    S.vcvt(S.vdiv(e1, den, mask), "bfloat16", mask, round="ROUND_A", part=0),
                                    "float32",
                                    mask,
                                    part=0,
                                )
                                p2 = S.vcvt(
                                    S.vcvt(S.vdiv(e2, den, mask), "bfloat16", mask, round="ROUND_A", part=0),
                                    "float32",
                                    mask,
                                    part=0,
                                )
                                p3 = S.vcvt(
                                    S.vcvt(S.vdiv(e3, den, mask), "bfloat16", mask, round="ROUND_A", part=0),
                                    "float32",
                                    mask,
                                    part=0,
                                )
                                value = S.vmul(S.vld(window[j, start], "NORM"), p0, mask)
                                value = S.vadd(value, S.vmul(S.vld(window[j + 1, start], "NORM"), p1, mask), mask)
                                value = S.vadd(value, S.vmul(S.vld(window[j + 2, start], "NORM"), p2, mask), mask)
                                value = S.vadd(value, S.vmul(S.vld(window[j + 3, start], "NORM"), p3, mask), mask)
                                S.vsts(
                                    bf[start], S.vcvt(value, "bfloat16", mask, round="ROUND_A", part=0), mask, "PK_B32"
                                )
                        with T.SimdVF():
                            mask = S.pset(32, "PAT_ALL")
                            scalar = S.pset(32, "PAT_VL1")
                            a = S.vcvt(S.vld(bf[0], "UNPK_B16"), "float32", mask, part=0)
                            b = S.vcvt(S.vld(bf[64], "UNPK_B16"), "float32", mask, part=0)
                            peak = S.vdupv(S.vcmax(S.vmax(S.vabs(a, mask), S.vabs(b, mask), mask), mask), mask)
                            unit = S.vdup(T.float32(1.0), "float32", mask)
                            if kind == "quant_div":
                                scaled = S.vdiv(peak, S.vdup(T.float32(448.0), "float32", mask), mask)
                            elif kind == "quant_ieee":
                                scaled = S.vdiv(
                                    peak, S.vdup(T.float32(448.0), "float32", mask), mask, precision="ftz_true"
                                )
                            else:
                                scaled = S.vmuls(peak, T.float32(1.0 / 448.0), mask)
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
                        page = slot // 32
                        T.copy(packed, K[page, slot % 32, :])
                        with T.SimtVF(threads=32):
                            for c in T.Parallel(1):
                                KS[page, slot % 32] = scale[c]

    return main
