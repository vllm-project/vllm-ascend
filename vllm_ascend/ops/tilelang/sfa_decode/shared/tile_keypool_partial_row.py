# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="valid-type"

import tilelang.ascend.language as T  # type: ignore[import-not-found]


def partial_projection(batch, parts, bm, bn, *, rollback=False):
    assert batch % bm == 256 % bn == 0 and parts <= 32
    width = ((4096 + parts - 1) // parts) // 16 * 16
    width = 640 if rollback else width
    last = 4096 - (parts - 1) * width

    @T.macro
    def panel(x, w, p, gr, col, seg, kk, start):
        a = T.alloc_l1((bm, kk), T.bfloat16)
        h = T.alloc_l1((bn, kk), T.bfloat16)
        acc = T.alloc_l0c((bm, bn), T.float32)
        T.copy(x[gr : gr + bm, start : start + kk], a)
        T.copy(w[col : col + bn, start : start + kk], h)
        T.gemm(a, h, acc, transpose_B=True, clear_accum=True)
        T.copy(acc, p[gr : gr + bm, seg * 256 + col : seg * 256 + col + bn])

    @T.prim_func
    def main(
        X: T.Tensor((batch, 4096), T.bfloat16),
        KGate: T.Tensor((256, 4096), T.bfloat16),
        Value: T.Tensor((batch, 16384), T.bfloat16),
    ):
        p = T.Tensor((batch, parts * 256), T.float32, Value.data)
        with T.Kernel((batch // bm) * (256 // bn) * parts) as task:
            seg = task % parts
            col = task // parts % (256 // bn) * bn
            gr = task // (parts * (256 // bn)) * bm
            if width == last:
                panel(X, KGate, p, gr, col, seg, width, seg * width)
            else:
                if seg < parts - 1:
                    panel(X, KGate, p, gr, col, seg, width, seg * width)
                else:
                    panel(X, KGate, p, gr, col, seg, last, (parts - 1) * width)

    return main
