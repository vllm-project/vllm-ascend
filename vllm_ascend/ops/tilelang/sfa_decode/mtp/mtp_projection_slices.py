# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


def kg_slices(nr, bb, bm=32, bn=128):
    k, n, parts = 4096, 256, 7

    @T.macro
    def panel(X, WG, P, m, col, j, offset, span):
        a = T.alloc_l1((bm, 128), T.bfloat16)
        w = T.alloc_l1((bn, 128), T.bfloat16)
        a0 = T.alloc_l0a((bm, 128), T.bfloat16)
        b0 = T.alloc_l0b((bn, 128), T.bfloat16)
        acc = T.alloc_l0c((bm, bn), T.float32)
        for z in T.Pipelined(span // 128, num_stages=1):
            kk = offset + z * 128
            T.copy(X[m : m + bm, kk : kk + 128], a)
            T.copy(WG[0, col : col + bn, kk : kk + 128], w)
            T.copy(a, a0)
            T.copy(w, b0)
            T.gemm(a0, b0, acc, transpose_B=True, clear_accum=z == 0)
        T.copy(acc, P[m : m + bm, j * n + col : j * n + col + bn])

    @T.prim_func
    def main(
        X: T.Tensor((nr, k), T.bfloat16),
        GateWeight: T.Tensor((1, n, k), T.bfloat16),
        OC: T.Tensor((bb, 16384), T.float8_e4m3fn),
    ):
        P = T.Tensor((nr, n * parts), T.float32, OC.data)
        with T.Kernel(nr // bm * (n // bn) * parts) as t:
            j = t % parts
            col = t // parts % (n // bn) * bn
            m = t // (parts * (n // bn)) * bm
            if j < 6:
                panel(X, GateWeight, P, m, col, j, j * 640, 640)
            else:
                panel(X, GateWeight, P, m, col, j, 3840, 256)

    return main


def qa_slices(nr, bb, bm=32, bn=64, bk=256, *, stages=1):
    k, n, parts = 4096, 2048, 3

    @T.macro
    def panel(C3, S3, WO, OS, P, m, col, j, offset, span):
        a = T.alloc_l1((bm, bk), T.float8_e4m3fn)
        w = T.alloc_l1((bn, bk), T.float8_e4m3fn)
        xs = T.alloc_l1((bm, bk // 64), T.uint16)
        ws = T.alloc_l1((bn, bk // 64), T.uint16)
        a0 = T.alloc_l0a((bm, bk), T.float8_e4m3fn)
        b0 = T.alloc_l0b((bn, bk), T.float8_e4m3fn)
        s0 = T.alloc_l0a_sf(a0)
        t0 = T.alloc_l0b_sf(b0)
        acc = T.alloc_l0c((bm, bn), T.float32)
        for z in T.Pipelined(span // bk, num_stages=stages):
            kk = offset + z * bk
            T.copy(C3[m : m + bm, kk : kk + bk], a)
            T.copy(WO[col : col + bn, kk : kk + bk], w)
            T.copy(S3[kk // 64 : (kk + bk) // 64, m : m + bm], xs, transpose=True)
            T.copy(OS[kk // 64 : (kk + bk) // 64, col : col + bn], ws, transpose=True)
            T.copy(a, a0)
            T.copy(w, b0)
            T.copy(xs, s0)
            T.copy(ws, t0)
            T.gemm_blockscaled(a0, b0, acc, s0, t0, transpose_B=True, clear_accum=z == 0)
        T.copy(acc, P[m : m + bm, j * n + col : j * n + col + bn])

    @T.prim_func
    def main(
        AC: T.Tensor((nr, k), T.float8_e4m3fn),
        AX: T.Tensor((k // 64, nr), T.uint16),
        AW: T.Tensor((n, k), T.float8_e4m3fn),
        AWX: T.Tensor((k // 64, n), T.uint16),
        V: T.Tensor((bb, 16384), T.bfloat16),
    ):
        P = T.Tensor((nr, parts * n), T.float32, V.data)
        with T.Kernel(nr // bm * (n // bn) * parts) as t:
            j = t % parts
            col = t // parts % (n // bn) * bn
            m = t // (parts * (n // bn)) * bm
            if j < 2:
                panel(AC, AX, AW, AWX, P, m, col, j, 1536 * j, 1536)
            else:
                panel(AC, AX, AW, AWX, P, m, col, j, 3072, 1024)

    return main
