# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


def pack(batch, prepacked_heads):
    rows = batch if prepacked_heads else batch + 32

    @T.prim_func
    def main(
        X: T.Tensor((batch, 4096), T.bfloat16),
        IWeight: T.Tensor((32, 4096), T.bfloat16),
        Panel: T.Tensor((rows, 4096), T.float32),
    ):
        with T.Kernel(rows * 2) as task:
            row = task // 2
            col = task % 2 * 2048
            src = T.alloc_shared((2048,), T.bfloat16)
            dst = T.alloc_shared((2048,), T.float32)
            if prepacked_heads or row < batch:
                T.copy(X[row, col : col + 2048], src)
            else:
                T.copy(IWeight[row - batch, col : col + 2048], src)
            with T.SimtVF(threads=256):
                for j in T.Parallel(2048):
                    dst[j] = T.cast(src[j], T.float32)
            T.copy(dst, Panel[row, col : col + 2048])

    return main
