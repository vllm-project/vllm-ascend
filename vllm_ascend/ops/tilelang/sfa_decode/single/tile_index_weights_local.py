# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


def local_head(batch, rows, k_tile, *, prepacked_heads):
    half = rows // 2
    offset = 0 if prepacked_heads else batch

    @T.prim_func
    def main(
        Panel: T.Tensor((batch, 4096), T.float32),
        HeadPanel: T.Tensor((offset + 32, 4096), T.float32),
        IW: T.Tensor((batch, 32), T.bfloat16),
    ):
        with T.MixedKernel(batch // rows, sids=2) as (task, sid):
            x = T.alloc_l1((rows, k_tile), T.float32)
            w = T.alloc_l1((32, k_tile), T.float32)
            dot = T.alloc_l0c((rows, 32), T.float32)
            partial = T.alloc_shared((half, 32), T.float32)
            total = T.alloc_shared((half, 32), T.float32)
            out = T.alloc_shared((half, 32), T.bfloat16)
            with T.SimtVF(threads=128):
                for r, h in T.Parallel(half, 32):
                    total[r, h] = 0.0
            for k in T.Pipelined(4096 // k_tile, num_stages=2):
                T.copy(Panel[task * rows : (task + 1) * rows, k * k_tile : (k + 1) * k_tile], x)
                T.copy(HeadPanel[offset : offset + 32, k * k_tile : (k + 1) * k_tile], w)
                T.gemm(x, w, dot, transpose_B=True, clear_accum=True)
                T.dual_copy(dot, partial)
                with T.SimtVF(threads=128):
                    for r, h in T.Parallel(half, 32):
                        total[r, h] += partial[r, h]
            with T.SimtVF(threads=128):
                for r, h in T.Parallel(half, 32):
                    out[r, h] = T.cast(T.cast(total[r, h], T.bfloat16) * T.bfloat16(1.0 / 64), T.bfloat16)
            T.dual_copy(out, IW[task * rows : (task + 1) * rows, :])

    return main
