# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


def stages(batch, width, *, prepacked_heads):
    parts = (4096 + width - 1) // width
    aligned = (width + 31) // 32 * 32
    read = aligned + 16
    assert read <= 4096 and batch % 32 == 0
    rows = parts * (batch if prepacked_heads else batch + 32)
    base = rows * aligned // 32
    offset = 0 if prepacked_heads else parts * batch

    @T.prim_func
    def pack(
        X: T.Tensor((batch, 4096), T.bfloat16),
        IWeight: T.Tensor((32, 4096), T.bfloat16),
        Panel: T.Tensor((rows, aligned), T.float32),
    ):
        with T.Kernel(rows // 8) as tile:
            i = tile * 8
            src = T.alloc_shared((8, read), T.bfloat16)
            dst = T.alloc_shared((8, aligned), T.float32)
            part = (
                i // batch
                if prepacked_heads
                else T.if_then_else(i < parts * batch, i // batch, (i - parts * batch) // 32)
            )
            start = part * width
            lo = T.min((start // 16) * 16, 4096 - read)
            shift = start - lo
            valid = T.min(width, 4096 - start)
            if prepacked_heads or i < parts * batch:
                row = i % batch
                T.copy(X[row : row + 8, lo : lo + read], src)
            else:
                row = (i - parts * batch) % 32
                T.copy(IWeight[row : row + 8, lo : lo + read], src)
            with T.SimtVF(threads=256):
                for r, c in T.Parallel(8, aligned):
                    dst[r, c] = T.if_then_else(
                        c < valid, T.cast(src[r, T.min(c + shift, read - 1)], T.float32), T.float32(0.0)
                    )
            T.copy(dst, Panel[i : i + 8, :])

    @T.prim_func
    def cube(
        Panel: T.Tensor((rows, aligned), T.float32),
        HeadPanel: T.Tensor((offset + parts * 32, aligned), T.float32),
        Parts: T.Tensor((base + parts * batch, 32), T.float32),
    ):
        with T.Kernel(parts * (batch // 32)) as task:
            row = task % (batch // 32) * 32
            p = task // (batch // 32)
            a = T.alloc_l1((32, 32), T.float32)
            w = T.alloc_l1((32, 32), T.float32)
            acc = T.alloc_l0c((32, 32), T.float32)
            for k in T.Pipelined(aligned // 32, num_stages=1):
                T.copy(Panel[p * batch + row : p * batch + row + 32, k * 32 : k * 32 + 32], a)
                T.copy(HeadPanel[offset + p * 32 : offset + p * 32 + 32, k * 32 : k * 32 + 32], w)
                T.gemm(a, w, acc, transpose_B=True, clear_accum=k == 0)
            T.copy(acc, Parts[base + p * batch + row : base + p * batch + row + 32, :])

    @T.prim_func
    def reduce(Parts: T.Tensor((base + parts * batch, 32), T.float32), IW: T.Tensor((batch, 32), T.bfloat16)):
        with T.Kernel(batch // 4) as tile:
            row = tile * 4
            partial = T.alloc_shared((4, 32), T.float32)
            accum = T.alloc_shared((4, 32), T.float32)
            out = T.alloc_shared((4, 32), T.bfloat16)
            T.copy(Parts[base + row : base + row + 4, :], accum)
            for k in T.serial(parts - 1):
                T.copy(Parts[base + (k + 1) * batch + row : base + (k + 1) * batch + row + 4, :], partial)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(4, 32):
                        accum[i, j] += partial[i, j]
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(4, 32):
                    out[i, j] = T.cast(T.cast(accum[i, j], T.bfloat16) * T.bfloat16(1.0 / 64), T.bfloat16)
            T.copy(out, IW[row : row + 4, :])

    return pack, cube, reduce
