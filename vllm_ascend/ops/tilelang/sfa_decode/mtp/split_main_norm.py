# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T

from ..shared.main_kv_norm_pack import make_norm_pack


def program(batch, pages, page_size, context, include_tail, mode, nr, bb):
    dtype = T.bfloat16
    norm = make_norm_pack(1, pages * page_size, 1e-5, mode)

    @T.prim_func
    def main(
        V: T.Tensor((bb, 16384), dtype),
        GQ: T.Tensor((1536,), dtype),
        GK: T.Tensor((512,), dtype),
        Slot: T.Tensor((batch, 1), T.int32),
        CacheRows: T.Tensor((pages * page_size, 132), T.int32),
        Norm: T.Tensor((batch, 1536), dtype),
    ):
        P = T.Tensor((nr, 3 * 2048), T.float32, V.data)
        with T.Kernel(batch) as row:
            src = T.alloc_shared((1, 2048), dtype)
            dst = T.alloc_shared((1, 1536), dtype)
            part = T.alloc_shared((3 * 2048,), T.float32)
            T.copy(P[row, :], part)
            with T.SimtVF(threads=256):
                for i in T.Parallel(2048):
                    pair = part[i] + part[i + 2048]
                    src[0, i] = T.cast(pair + part[i + 4096], T.bfloat16)
            norm(src, GQ, GK, Slot, CacheRows, row, 0, dst)
            T.copy(dst, Norm[row : row + 1, :])

    return main
