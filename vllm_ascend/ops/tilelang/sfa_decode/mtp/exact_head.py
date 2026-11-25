# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


def pack_x(rows):
    @T.prim_func
    def main(X: T.Tensor((rows, 4096), T.bfloat16), QB: T.Tensor((rows, 16384), T.bfloat16)):
        HPX = T.Tensor((rows, 4096), T.float32, QB.data)
        with T.Kernel(rows * 2) as task:
            i = task // 2
            j = task % 2 * 2048
            src = T.alloc_shared((2048,), T.bfloat16)
            dst = T.alloc_shared((2048,), T.float32)
            T.copy(X[i, j : j + 2048], src)
            with T.SimtVF(threads=256):
                for y in T.Parallel(2048):
                    dst[y] = T.cast(src[y], T.float32)
            T.copy(dst, HPX[i, j : j + 2048])

    return main


def dot(rows):
    @T.macro
    def panel(X, HP, Parts, group, part, start, n):
        a = T.alloc_l1((32, n), T.float32)
        w = T.alloc_l1((32, n), T.float32)
        c = T.alloc_l0c((32, 32), T.float32)
        T.copy(X[group * 32 : group * 32 + 32, start : start + n], a)
        T.copy(HP[part, :, :n], w)
        T.gemm(a, w, c, transpose_B=True, clear_accum=True)
        T.copy(c, Parts[group, part, :, :])

    @T.prim_func
    def main(
        QB: T.Tensor((rows, 16384), T.bfloat16),
        Panel: T.Tensor((rows + 32, 4096), T.float32),
        Parts: T.Tensor((rows // 32, 32, 32, 32), T.float32),
    ):
        HPX = T.Tensor((rows, 4096), T.float32, QB.data)
        HP = T.Tensor((28, 32, 176), T.float32, Panel.data)
        with T.Kernel(rows // 32 * 28) as task:
            i = task // 28
            j = task % 28
            if j < 27:
                panel(HPX, HP, Parts, i, j, j * 147 // 16 * 16, 176)
            else:
                panel(HPX, HP, Parts, i, j, 3968, 128)

    return main
