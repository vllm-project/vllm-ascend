# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T

from ..mtp.native_output import MXFP8_NZ_C0, make_native_weight_loader


def banked_mx(batch, width, out_width, bm, bn, panel, bk, mode, output_rows, *, native_output=False):
    weight_shape = (out_width // MXFP8_NZ_C0, width, MXFP8_NZ_C0) if native_output else (out_width, width)
    nr = output_rows // bm
    nc = out_width // bn if bn else 28
    assert width % (2 * panel) == 0 and panel % bk == 0 and batch % bm == 0
    fp8 = T.float8_e4m3fn

    load_native = make_native_weight_loader(panel)

    @T.macro
    def tile(gr, col, n, C3, S3, WO, OS, Y):
        a = T.alloc_l1((bm, panel), fp8)
        w = T.alloc_l1((panel, n) if native_output else (n, panel), fp8)
        xs = T.alloc_l1((bm, panel // 64), T.uint16)
        ws = T.alloc_l1((n, panel // 64), T.uint16)
        a0 = T.alloc_l0a((bm, bk), fp8)
        b0 = T.alloc_l0b((n, bk), fp8)
        s0 = T.alloc_l0a_sf(a0)
        t0 = T.alloc_l0b_sf(b0)
        acc = T.alloc_l0c((bm, n), T.float32)
        for p in T.Pipelined(width // panel, num_stages=2):
            T.copy(C3[gr * bm : (gr + 1) * bm, (p) * panel : ((p) + 1) * panel], a)
            if native_output:
                load_native(WO, w, p, col, n)
            else:
                T.copy(WO[col : col + n, (p) * panel : ((p) + 1) * panel], w)
            T.copy(S3[(p) * (panel // 64) : ((p) + 1) * (panel // 64), gr * bm : (gr + 1) * bm], xs, transpose=True)
            T.copy(OS[(p) * (panel // 64) : ((p) + 1) * (panel // 64), col : col + n], ws, transpose=True)
            for j in T.Pipelined(panel // bk, num_stages=2):
                T.copy(a[:, (j) * bk : ((j) + 1) * bk], a0)
                if native_output:
                    T.copy(w[j * bk : (j + 1) * bk, :], b0, transpose=True)
                else:
                    T.copy(w[:, (j) * bk : ((j) + 1) * bk], b0)
                T.copy(xs[:, (j) * (bk // 64) : ((j) + 1) * (bk // 64)], s0)
                T.copy(ws[:, (j) * (bk // 64) : ((j) + 1) * (bk // 64)], t0)
                T.gemm_blockscaled(a0, b0, acc, s0, t0, transpose_B=True, clear_accum=((p) == 0 and (j) == 0))
        T.copy(acc, Y[gr * bm : (gr + 1) * bm, col : col + n])

    @T.prim_func
    def main(
        C3: T.Tensor((batch, width), fp8),
        S3: T.Tensor((width // 64, batch), T.uint16),
        WO: T.Tensor(weight_shape, fp8),
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
