# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="valid-type"

import tilelang.ascend.language as T  # type: ignore[import-not-found]

from . import INDEX_HEAD_PANEL_SHAPE


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
        Panel: T.Tensor(INDEX_HEAD_PANEL_SHAPE, T.float32),
        Parts: T.Tensor((rows // 32, 32, 32, 32), T.float32),
    ):
        HPX = T.Tensor((rows, 4096), T.float32, QB.data)
        with T.Kernel(rows // 32 * 28) as task:
            i = task // 28
            j = task % 28
            if j < 27:
                panel(HPX, Panel, Parts, i, j, j * 147 // 16 * 16, 176)
            else:
                panel(HPX, Panel, Parts, i, j, 3968, 128)

    return main


def finish(rows):
    @T.prim_func
    def main(Parts: T.Tensor((rows // 32, 32, 32, 32), T.float32), IW: T.Tensor((rows, 32), T.bfloat16)):
        with T.Kernel(rows // 4) as tile:
            first = (tile % 8) * 4
            buf = T.alloc_shared((28, 4, 32), T.float32)
            out = T.alloc_shared((4, 32), T.bfloat16)
            T.copy(Parts[tile // 8, 0:28, first : first + 4, :], buf)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(4, 32):
                    v = T.alloc_var(T.float32, init=0.0)
                    for part in T.serial(28):
                        v += buf[part, i, j]
                    out[i, j] = T.cast(T.cast(v, T.bfloat16) * T.bfloat16(1.0 / 64), T.bfloat16)
            T.copy(out, IW[tile * 4 : tile * 4 + 4, :])

    return main
