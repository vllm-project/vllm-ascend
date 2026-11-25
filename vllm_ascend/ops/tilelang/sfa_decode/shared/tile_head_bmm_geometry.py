# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


def head_gemm(
    batch, heads, width, out_width, bm, bn, bk, stages, *, l1_k=None, tail_rows=None, row_tiles=1, input_major=False
):
    assert batch % bm == out_width % bn == width % bk == 0
    panel = bk if l1_k is None else l1_k
    assert panel > 0 and width % panel == panel % bk == 0
    tail_rows = bm if tail_rows is None else tail_rows
    assert 0 < tail_rows <= bm and tail_rows % 16 == 0

    weight_shape = (heads, width, out_width) if input_major else (heads, out_width, width)

    @T.macro
    def load_weight(WV, w, head, col, start, size):
        if input_major:
            T.copy(WV[head, start : start + size, col * bn : (col + 1) * bn], w, transpose=True)
        else:
            T.copy(WV[head, col * bn : (col + 1) * bn, start : start + size], w)

    @T.macro
    def multiply_panel(x, w, a0, b0, acc):
        for k in T.Pipelined(width // bk, num_stages=stages):
            T.copy(x[:, k * bk : (k + 1) * bk], a0)
            T.copy(w[:, k * bk : (k + 1) * bk], b0)
            T.gemm(a0, b0, acc, transpose_B=True, clear_accum=k == 0)

    if row_tiles > 1:
        assert batch % (bm * row_tiles) == 0 and tail_rows == bm and panel == width

        @T.prim_func
        def grouped(
            Latent3: T.Tensor((batch, heads * width), T.bfloat16),
            WV: T.Tensor(weight_shape, T.bfloat16),
            Value: T.Tensor((batch, heads * out_width), T.bfloat16),
        ):
            with T.Kernel(batch // (bm * row_tiles) * heads * (out_width // bn)) as task:
                head = task // (out_width // bn) % heads
                col = task % (out_width // bn)
                first = task // (heads * (out_width // bn)) * row_tiles
                x = T.alloc_l1((bm, width), T.bfloat16)
                w = T.alloc_l1((bn, width), T.bfloat16)
                a0 = T.alloc_l0a((bm, bk), T.bfloat16)
                b0 = T.alloc_l0b((bn, bk), T.bfloat16)
                acc = T.alloc_l0c((bm, bn), T.float32)
                load_weight(WV, w, head, col, 0, width)
                for offset in T.unroll(row_tiles):
                    row = (first + offset) * bm
                    out_col = head * out_width + col * bn
                    T.copy(Latent3[row : row + bm, head * width : (head + 1) * width], x)
                    multiply_panel(x, w, a0, b0, acc)
                    T.copy(acc, Value[row : row + bm, out_col : out_col + bn])

        return grouped

    @T.macro
    def project(gr, head, col, rows, Latent3, WV, Value):
        x = T.alloc_l1((rows, panel), T.bfloat16)
        w = T.alloc_l1((bn, panel), T.bfloat16)
        a0 = T.alloc_l0a((rows, bk), T.bfloat16)
        b0 = T.alloc_l0b((bn, bk), T.bfloat16)
        acc = T.alloc_l0c((rows, bn), T.float32)
        if panel == width:
            T.copy(Latent3[gr * bm : gr * bm + rows, head * width : (head + 1) * width], x)
            load_weight(WV, w, head, col, 0, width)
            multiply_panel(x, w, a0, b0, acc)
        elif panel > bk:
            for outer in T.Pipelined(width // panel, num_stages=stages):
                start = head * width + outer * panel
                T.copy(Latent3[gr * bm : gr * bm + rows, start : start + panel], x)
                load_weight(WV, w, head, col, outer * panel, panel)
                for k in T.Pipelined(panel // bk, num_stages=stages):
                    T.copy(x[:, k * bk : (k + 1) * bk], a0)
                    T.copy(w[:, k * bk : (k + 1) * bk], b0)
                    T.gemm(a0, b0, acc, transpose_B=True, clear_accum=outer == 0 and k == 0)
        else:
            for k in T.Pipelined(width // bk, num_stages=stages):
                T.copy(Latent3[gr * bm : gr * bm + rows, head * width + k * bk : head * width + (k + 1) * bk], x)
                load_weight(WV, w, head, col, k * bk, bk)
                T.copy(x, a0)
                T.copy(w, b0)
                T.gemm(a0, b0, acc, transpose_B=True, clear_accum=k == 0)
        T.copy(acc, Value[gr * bm : gr * bm + rows, head * out_width + col * bn : head * out_width + (col + 1) * bn])

    @T.prim_func
    def main(
        Latent3: T.Tensor((batch, heads * width), T.bfloat16),
        WV: T.Tensor(weight_shape, T.bfloat16),
        Value: T.Tensor((batch, heads * out_width), T.bfloat16),
    ):
        with T.Kernel((batch // bm) * heads * (out_width // bn)) as task:
            gr = task // (heads * (out_width // bn))
            head = (task // (out_width // bn)) % heads
            col = task % (out_width // bn)
            if tail_rows < bm and gr == batch // bm - 1:
                project(gr, head, col, tail_rows, Latent3, WV, Value)
            else:
                project(gr, head, col, bm, Latent3, WV, Value)

    return main
