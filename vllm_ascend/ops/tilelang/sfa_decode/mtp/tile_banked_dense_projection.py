# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


def projection(batch, width, out_width, bm, bn, panel, bk, column_first, *, column_tiles=None):
    nr = batch // bm
    nc = column_tiles or out_width // bn
    blocks_per_tile, wider_tiles = divmod(out_width // 16, nc)
    assert width % panel == 0 and panel % bk == 0 and batch % bm == 0
    fp8 = T.float8_e4m3fn
    # A single 32-row M tile consumes each projection weight once.
    weight_policy = "NOTALLOC_KEEP" if batch == 32 else "NORMAL_FV"

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
        for p in T.Pipelined(width // panel, num_stages=2):
            T.copy(C3[gr * bm : (gr + 1) * bm, (p) * panel : ((p) + 1) * panel], a)
            T.copy(WO[col : col + n, (p) * panel : ((p) + 1) * panel], w, l2_cache_ctrl=weight_policy)
            T.copy(S3[(p) * (panel // 64) : ((p) + 1) * (panel // 64), gr * bm : (gr + 1) * bm], xs, transpose=True)
            T.copy(OS[(p) * (panel // 64) : ((p) + 1) * (panel // 64), col : col + n], ws, transpose=True)
            for j in T.Pipelined(panel // bk, num_stages=2):
                T.copy(a[:, (j) * bk : ((j) + 1) * bk], a0)
                T.copy(w[:, (j) * bk : ((j) + 1) * bk], b0)
                T.copy(xs[:, (j) * (bk // 64) : ((j) + 1) * (bk // 64)], s0)
                T.copy(ws[:, (j) * (bk // 64) : ((j) + 1) * (bk // 64)], t0)
                T.gemm_blockscaled(a0, b0, acc, s0, t0, transpose_B=True, clear_accum=((p) == 0 and (j) == 0))
        T.copy(acc, Y[gr * bm : (gr + 1) * bm, col : col + n])

    @T.prim_func
    def main(
        C3: T.Tensor((batch, width), fp8),
        S3: T.Tensor((width // 64, batch), T.uint16),
        WO: T.Tensor((out_width, width), fp8),
        OS: T.Tensor((width // 64, out_width), T.uint16),
        Y: T.Tensor((batch, out_width), T.bfloat16),
    ):
        with T.Kernel(nr * nc) as task:
            if column_tiles:
                # The small decode projection uses one M tile. Spread complete 16-column blocks across the Cubes.
                gr = task % nr if column_first else task // nc
                lane = task // nr if column_first else task % nc
                if lane < wider_tiles:
                    width_n = (blocks_per_tile + 1) * 16
                    tile(gr, lane * width_n, width_n, C3, S3, WO, OS, Y)
                else:
                    start = (lane * blocks_per_tile + wider_tiles) * 16
                    tile(gr, start, blocks_per_tile * 16, C3, S3, WO, OS, Y)
            elif column_first:
                tile(task % nr, (task // nr) * bn, bn, C3, S3, WO, OS, Y)
            else:
                tile(task // nc, (task % nc) * bn, bn, C3, S3, WO, OS, Y)

    return main
