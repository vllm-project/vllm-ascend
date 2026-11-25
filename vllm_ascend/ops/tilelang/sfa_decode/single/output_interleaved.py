# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


def banked_mx(batch, width, out_width, bm, bn, panel, bk, mode, output_rows):
    nr = output_rows // bm
    nc = out_width // bn if bn else 28
    assert width % (2 * panel) == 0 and panel % bk == 0 and batch % bm == 0
    fp8 = T.float8_e4m3fn

    @T.macro
    def tile(gr, col, n, C3, S3, WO, OS, Y):
        a = T.alloc_l1((bm, panel), fp8)
        w = T.alloc_l1((n, panel), fp8)
        xs = T.alloc_l1((bm, panel // 64), T.uint16)
        ws = T.alloc_l1((n, panel // 64), T.uint16)
        a0 = T.alloc_l0a((bm, bk), fp8)
        b0 = T.alloc_l0b((n, bk), fp8)
        s0 = T.alloc_l0a_sf(a0)
        t0 = T.alloc_l0b_sf(b0)
        acc = T.alloc_l0c((bm, n), T.float32)
        # The second set of panels holds the next K tile while Cube consumes the first.
        aa = T.alloc_l1((bm, panel), fp8)
        ww = T.alloc_l1((n, panel), fp8)
        xss = T.alloc_l1((bm, panel // 64), T.uint16)
        wss = T.alloc_l1((n, panel // 64), T.uint16)
        a1 = T.alloc_l0a((bm, bk), fp8)
        b1 = T.alloc_l0b((n, bk), fp8)
        s1 = T.alloc_l0a_sf(a1)
        t1 = T.alloc_l0b_sf(b1)
        T.copy(C3[gr * bm : (gr + 1) * bm, (0) * panel : ((0) + 1) * panel], a)
        T.copy(WO[col : col + n, (0) * panel : ((0) + 1) * panel], w)
        T.copy(S3[(0) * (panel // 64) : ((0) + 1) * (panel // 64), gr * bm : (gr + 1) * bm], xs, transpose=True)
        T.copy(OS[(0) * (panel // 64) : ((0) + 1) * (panel // 64), col : col + n], ws, transpose=True)
        for pair in T.serial(width // (2 * panel) - 1):
            T.copy(C3[gr * bm : (gr + 1) * bm, (2 * pair + 1) * panel : ((2 * pair + 1) + 1) * panel], aa)
            T.copy(WO[col : col + n, (2 * pair + 1) * panel : ((2 * pair + 1) + 1) * panel], ww)
            T.copy(
                S3[(2 * pair + 1) * (panel // 64) : ((2 * pair + 1) + 1) * (panel // 64), gr * bm : (gr + 1) * bm],
                xss,
                transpose=True,
            )
            T.copy(
                OS[(2 * pair + 1) * (panel // 64) : ((2 * pair + 1) + 1) * (panel // 64), col : col + n],
                wss,
                transpose=True,
            )
            for jj in T.serial(panel // (2 * bk)):
                T.copy(a[:, (2 * jj) * bk : ((2 * jj) + 1) * bk], a0)
                T.copy(w[:, (2 * jj) * bk : ((2 * jj) + 1) * bk], b0)
                T.copy(xs[:, (2 * jj) * (bk // 64) : ((2 * jj) + 1) * (bk // 64)], s0)
                T.copy(ws[:, (2 * jj) * (bk // 64) : ((2 * jj) + 1) * (bk // 64)], t0)
                T.copy(a[:, (2 * jj + 1) * bk : ((2 * jj + 1) + 1) * bk], a1)
                T.copy(w[:, (2 * jj + 1) * bk : ((2 * jj + 1) + 1) * bk], b1)
                T.copy(xs[:, (2 * jj + 1) * (bk // 64) : ((2 * jj + 1) + 1) * (bk // 64)], s1)
                T.copy(ws[:, (2 * jj + 1) * (bk // 64) : ((2 * jj + 1) + 1) * (bk // 64)], t1)
                T.gemm_blockscaled(
                    a0, b0, acc, s0, t0, transpose_B=True, clear_accum=((2 * pair) == 0 and (2 * jj) == 0)
                )
                T.gemm_blockscaled(
                    a1, b1, acc, s1, t1, transpose_B=True, clear_accum=((2 * pair) == 0 and (2 * jj + 1) == 0)
                )
            T.copy(C3[gr * bm : (gr + 1) * bm, (2 * pair + 2) * panel : ((2 * pair + 2) + 1) * panel], a)
            T.copy(WO[col : col + n, (2 * pair + 2) * panel : ((2 * pair + 2) + 1) * panel], w)
            T.copy(
                S3[(2 * pair + 2) * (panel // 64) : ((2 * pair + 2) + 1) * (panel // 64), gr * bm : (gr + 1) * bm],
                xs,
                transpose=True,
            )
            T.copy(
                OS[(2 * pair + 2) * (panel // 64) : ((2 * pair + 2) + 1) * (panel // 64), col : col + n],
                ws,
                transpose=True,
            )
            for jj in T.serial(panel // (2 * bk)):
                T.copy(aa[:, (2 * jj) * bk : ((2 * jj) + 1) * bk], a0)
                T.copy(ww[:, (2 * jj) * bk : ((2 * jj) + 1) * bk], b0)
                T.copy(xss[:, (2 * jj) * (bk // 64) : ((2 * jj) + 1) * (bk // 64)], s0)
                T.copy(wss[:, (2 * jj) * (bk // 64) : ((2 * jj) + 1) * (bk // 64)], t0)
                T.copy(aa[:, (2 * jj + 1) * bk : ((2 * jj + 1) + 1) * bk], a1)
                T.copy(ww[:, (2 * jj + 1) * bk : ((2 * jj + 1) + 1) * bk], b1)
                T.copy(xss[:, (2 * jj + 1) * (bk // 64) : ((2 * jj + 1) + 1) * (bk // 64)], s1)
                T.copy(wss[:, (2 * jj + 1) * (bk // 64) : ((2 * jj + 1) + 1) * (bk // 64)], t1)
                T.gemm_blockscaled(
                    a0, b0, acc, s0, t0, transpose_B=True, clear_accum=((2 * pair + 1) == 0 and (2 * jj) == 0)
                )
                T.gemm_blockscaled(
                    a1, b1, acc, s1, t1, transpose_B=True, clear_accum=((2 * pair + 1) == 0 and (2 * jj + 1) == 0)
                )
        T.copy(C3[gr * bm : (gr + 1) * bm, (width // panel - 1) * panel : ((width // panel - 1) + 1) * panel], aa)
        T.copy(WO[col : col + n, (width // panel - 1) * panel : ((width // panel - 1) + 1) * panel], ww)
        T.copy(
            S3[
                (width // panel - 1) * (panel // 64) : ((width // panel - 1) + 1) * (panel // 64),
                gr * bm : (gr + 1) * bm,
            ],
            xss,
            transpose=True,
        )
        T.copy(
            OS[(width // panel - 1) * (panel // 64) : ((width // panel - 1) + 1) * (panel // 64), col : col + n],
            wss,
            transpose=True,
        )
        for jj in T.serial(panel // (2 * bk)):
            T.copy(a[:, (2 * jj) * bk : ((2 * jj) + 1) * bk], a0)
            T.copy(w[:, (2 * jj) * bk : ((2 * jj) + 1) * bk], b0)
            T.copy(xs[:, (2 * jj) * (bk // 64) : ((2 * jj) + 1) * (bk // 64)], s0)
            T.copy(ws[:, (2 * jj) * (bk // 64) : ((2 * jj) + 1) * (bk // 64)], t0)
            T.copy(a[:, (2 * jj + 1) * bk : ((2 * jj + 1) + 1) * bk], a1)
            T.copy(w[:, (2 * jj + 1) * bk : ((2 * jj + 1) + 1) * bk], b1)
            T.copy(xs[:, (2 * jj + 1) * (bk // 64) : ((2 * jj + 1) + 1) * (bk // 64)], s1)
            T.copy(ws[:, (2 * jj + 1) * (bk // 64) : ((2 * jj + 1) + 1) * (bk // 64)], t1)
            T.gemm_blockscaled(
                a0, b0, acc, s0, t0, transpose_B=True, clear_accum=((width // panel - 2) == 0 and (2 * jj) == 0)
            )
            T.gemm_blockscaled(
                a1, b1, acc, s1, t1, transpose_B=True, clear_accum=((width // panel - 2) == 0 and (2 * jj + 1) == 0)
            )
        for jj in T.serial(panel // (2 * bk)):
            T.copy(aa[:, (2 * jj) * bk : ((2 * jj) + 1) * bk], a0)
            T.copy(ww[:, (2 * jj) * bk : ((2 * jj) + 1) * bk], b0)
            T.copy(xss[:, (2 * jj) * (bk // 64) : ((2 * jj) + 1) * (bk // 64)], s0)
            T.copy(wss[:, (2 * jj) * (bk // 64) : ((2 * jj) + 1) * (bk // 64)], t0)
            T.copy(aa[:, (2 * jj + 1) * bk : ((2 * jj + 1) + 1) * bk], a1)
            T.copy(ww[:, (2 * jj + 1) * bk : ((2 * jj + 1) + 1) * bk], b1)
            T.copy(xss[:, (2 * jj + 1) * (bk // 64) : ((2 * jj + 1) + 1) * (bk // 64)], s1)
            T.copy(wss[:, (2 * jj + 1) * (bk // 64) : ((2 * jj + 1) + 1) * (bk // 64)], t1)
            T.gemm_blockscaled(
                a0, b0, acc, s0, t0, transpose_B=True, clear_accum=((width // panel - 1) == 0 and (2 * jj) == 0)
            )
            T.gemm_blockscaled(
                a1, b1, acc, s1, t1, transpose_B=True, clear_accum=((width // panel - 1) == 0 and (2 * jj + 1) == 0)
            )
        T.copy(acc, Y[gr * bm : (gr + 1) * bm, col : col + n])

    @T.prim_func
    def main(
        C3: T.Tensor((batch, width), fp8),
        S3: T.Tensor((width // 64, batch), T.uint16),
        WO: T.Tensor((out_width, width), fp8),
        OS: T.Tensor((width // 64, out_width), T.uint16),
        Y: T.Tensor((output_rows, out_width), T.bfloat16),
    ):
        with T.Kernel(nr * nc) as task:
            if bn:
                tile(task % nr, (task // nr) * bn, bn, C3, S3, WO, OS, Y)
            else:
                if task % 28 < 4:
                    tile(task // 28, (task % 28) * 160, 160, C3, S3, WO, OS, Y)
                else:
                    tile(task // 28, 640 + (task % 28 - 4) * 144, 144, C3, S3, WO, OS, Y)

    return main
