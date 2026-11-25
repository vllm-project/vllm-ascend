# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


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
