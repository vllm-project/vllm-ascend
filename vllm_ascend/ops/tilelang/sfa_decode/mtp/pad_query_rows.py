# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


def prepare(tokens, rows):
    @T.prim_func
    def main(
        X: T.Tensor((tokens, 4096), T.bfloat16),
        Input: T.Tensor((rows, 4096), T.bfloat16),
        Norm: T.Tensor((rows, 1536), T.bfloat16),
        UnusedLatent: T.Tensor((rows, 32768), T.bfloat16),
    ):
        with T.Kernel(rows) as r:
            row = T.alloc_shared((4096,), T.bfloat16)
            if r < tokens:
                T.copy(X[r, :], row)
            else:
                T.clear(row)
                T.copy(row[:1536], Norm[r, :])
                for j in T.serial(8):
                    T.copy(row, UnusedLatent[r, j * 4096 : (j + 1) * 4096])
            T.copy(row, Input[r, :])

    return main
