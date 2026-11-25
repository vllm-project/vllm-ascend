# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T

from .tile_mx_interleaved import encode_native_layout


def project_and_quantize(batch, row_tile, *, balanced=False, input_major=False):
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
        encode_native_layout(partial, codes, scales, half, values, True)
        T.copy(codes, C3[row : row + half, col : col + values])
        T.copy(scales, S3[col // 64 : (col + values) // 64, row : row + half])

    @T.prim_func
    def main(
        Latent3: T.Tensor((batch, heads * width), T.bfloat16),
        WV: T.Tensor(weight_shape, T.bfloat16),
        C3: T.Tensor((batch, heads * values), T.float8_e4m3fn),
        S3: T.Tensor((heads * values // 64, batch), T.int16),
    ):
        with T.MixedKernel(28 if balanced else (batch // row_tile) * heads // 2, sids=2) as (cta, sid):
            x = T.alloc_l1((row_tile, panel), T.bfloat16)
            w = T.alloc_l1((values, panel), T.bfloat16)
            a0 = T.alloc_l0a((row_tile, k_tile), T.bfloat16)
            b0 = T.alloc_l0b((values, k_tile), T.bfloat16)
            acc = T.alloc_l0c((row_tile, values), T.float32)
            partial = T.alloc_shared((half, values), T.float32)
            codes = T.alloc_shared((half, values), T.float8_e4m3fn)
            scales = T.alloc_shared((values // 64, half), T.int16)
            if balanced:
                x1 = T.alloc_l1((row_tile, panel), T.bfloat16)
                w1 = T.alloc_l1((values, panel), T.bfloat16)
                # Refill the consumed first panel from the next head while Cube uses the second.
                load_panel(cta, 0, Latent3, WV, x, w)
                load_panel(cta, panel, Latent3, WV, x1, w1)
                accumulate_panel(x, w, a0, b0, acc, True)
                load_panel(cta + 28, 0, Latent3, WV, x, w)
                accumulate_panel(x1, w1, a0, b0, acc, False)
                load_panel(cta + 28, panel, Latent3, WV, x1, w1)
                T.dual_copy(acc, partial)
                accumulate_panel(x, w, a0, b0, acc, True)
                if cta < 8:
                    load_panel(cta + 56, 0, Latent3, WV, x, w)
                store(cta, sid, partial, codes, scales, C3, S3)
                accumulate_panel(x1, w1, a0, b0, acc, False)
                if cta < 8:
                    load_panel(cta + 56, panel, Latent3, WV, x1, w1)
                T.dual_copy(acc, partial)
                if cta < 8:
                    accumulate_panel(x, w, a0, b0, acc, True)
                    accumulate_panel(x1, w1, a0, b0, acc, False)
                store(cta + 28, sid, partial, codes, scales, C3, S3)
                if cta < 8:
                    T.dual_copy(acc, partial)
                    store(cta + 56, sid, partial, codes, scales, C3, S3)
            else:
                project(2 * cta, Latent3, WV, x, w, a0, b0, acc)
                T.dual_copy(acc, partial)
                project(2 * cta + 1, Latent3, WV, x, w, a0, b0, acc)
                store(2 * cta, sid, partial, codes, scales, C3, S3)
                T.dual_copy(acc, partial)
                store(2 * cta + 1, sid, partial, codes, scales, C3, S3)

    return main
