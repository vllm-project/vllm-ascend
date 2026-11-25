# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T

from .tile_mx_clamped_scale import encode_block


def head_gemm(batch, heads, width, out_width, bm, bn, bk, stages, *, l1_k=None, row_tiles=1, input_major=False):
    assert batch % bm == out_width % bn == width % bk == 0
    panel = bk if l1_k is None else l1_k
    assert panel > 0 and width % panel == panel % bk == 0
    weight_shape = (heads, width, out_width) if input_major else (heads, out_width, width)

    @T.macro
    def load_weight(WV, w, head, col, start, size):
        if input_major:
            T.copy(WV[head, start : start + size, col * bn : (col + 1) * bn], w, transpose=True)
        else:
            T.copy(WV[head, col * bn : (col + 1) * bn, start : start + size], w)

    assert panel in (bk, width) and (row_tiles == 1 or panel == width)

    @T.prim_func
    def main(
        Latent3: T.Tensor((batch, heads * width), T.bfloat16),
        WV: T.Tensor(weight_shape, T.bfloat16),
        Value: T.Tensor((batch, heads * out_width), T.bfloat16),
    ):
        with T.Kernel(batch // (bm * row_tiles) * heads * (out_width // bn)) as task:
            head = task // (out_width // bn) % heads
            col = task % (out_width // bn)
            group = task // (heads * (out_width // bn)) * row_tiles
            x = T.alloc_l1((bm, panel), T.bfloat16)
            w = T.alloc_l1((bn, panel), T.bfloat16)
            a0 = T.alloc_l0a((bm, bk), T.bfloat16)
            b0 = T.alloc_l0b((bn, bk), T.bfloat16)
            acc = T.alloc_l0c((bm, bn), T.float32)
            if panel == width:
                load_weight(WV, w, head, col, 0, width)
            for offset in T.unroll(row_tiles):
                row = (group + offset) * bm
                if panel == width:
                    T.copy(Latent3[row : row + bm, head * width : (head + 1) * width], x)
                for k in T.Pipelined(width // bk, num_stages=stages):
                    if panel == width:
                        T.copy(x[:, k * bk : (k + 1) * bk], a0)
                        T.copy(w[:, k * bk : (k + 1) * bk], b0)
                    else:
                        start = head * width + k * bk
                        T.copy(Latent3[row : row + bm, start : start + bk], x)
                        load_weight(WV, w, head, col, k * bk, bk)
                        T.copy(x, a0)
                        T.copy(w, b0)
                    T.gemm(a0, b0, acc, transpose_B=True, clear_accum=k == 0)
                begin = head * out_width + col * bn
                T.copy(acc, Value[row : row + bm, begin : begin + bn])

    return main


def wuv_program(batch, row_tile, *, input_major=False):
    heads, width, values = 64, 512, 256
    panel, k_tile = 256, 64
    weight_shape = (heads, width, values) if input_major else (heads, values, width)
    half = row_tile // 2

    @T.macro
    def load_panel(task, offset, Latent3, WV, x, w):
        row = task // heads * row_tile
        head = task % heads
        start = head * width + offset
        T.copy(Latent3[row : row + row_tile, start : start + panel], x)
        if input_major:
            T.copy(WV[head, offset : offset + panel, :], w, transpose=True)
        else:
            T.copy(WV[head, :, offset : offset + panel], w)

    @T.macro
    def accumulate_panel(x, w, a0, b0, acc, first):
        for k in T.Pipelined(panel // k_tile, num_stages=2):
            T.copy(x[:, k * k_tile : (k + 1) * k_tile], a0)
            T.copy(w[:, k * k_tile : (k + 1) * k_tile], b0)
            T.gemm(a0, b0, acc, transpose_B=True, clear_accum=first and k == 0)

    @T.macro
    def project(task, Latent3, WV, x, w, a0, b0, acc):
        for outer in T.Pipelined(width // panel, num_stages=2):
            load_panel(task, outer * panel, Latent3, WV, x, w)
            accumulate_panel(x, w, a0, b0, acc, outer == 0)

    @T.macro
    def store(task, sid, partial, codes, scales, C3, S3):
        row = task // heads * row_tile + sid * half
        col = task % heads * values
        encode_block(partial, codes, scales, half, values, fp32_input=True)
        T.copy(codes, C3[row : row + half, col : col + values])
        T.copy(scales, S3[col // 64 : (col + values) // 64, row : row + half])

    @T.prim_func
    def main(
        Latent3: T.Tensor((batch, heads * width), T.bfloat16),
        WV: T.Tensor(weight_shape, T.bfloat16),
        C3: T.Tensor((batch, heads * values), T.float8_e4m3fn),
        S3: T.Tensor((heads * values // 64, batch), T.int16),
    ):
        with T.MixedKernel((batch // row_tile) * heads // 2, sids=2) as (cta, sid):
            x = T.alloc_l1((row_tile, panel), T.bfloat16)
            w = T.alloc_l1((values, panel), T.bfloat16)
            a0 = T.alloc_l0a((row_tile, k_tile), T.bfloat16)
            b0 = T.alloc_l0b((values, k_tile), T.bfloat16)
            acc = T.alloc_l0c((row_tile, values), T.float32)
            partial = T.alloc_shared((half, values), T.float32)
            codes = T.alloc_shared((half, values), T.float8_e4m3fn)
            scales = T.alloc_shared((values // 64, half), T.int16)
            project(2 * cta, Latent3, WV, x, w, a0, b0, acc)
            T.dual_copy(acc, partial)
            project(2 * cta + 1, Latent3, WV, x, w, a0, b0, acc)
            store(2 * cta, sid, partial, codes, scales, C3, S3)
            T.dual_copy(acc, partial)
            store(2 * cta + 1, sid, partial, codes, scales, C3, S3)

    return main
