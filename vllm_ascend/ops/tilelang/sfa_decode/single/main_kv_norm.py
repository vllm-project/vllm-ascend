# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T

from ..shared.main_kv_norm_pack import make_norm_pack


def program(batch, pages, page_size, context, include_tail, mode):
    dtype = T.bfloat16
    norm = make_norm_pack(1, pages * page_size, 1e-5, mode)

    @T.prim_func
    def main(
        QKV: T.Tensor((batch, 2048), dtype),
        GQ: T.Tensor((1536,), dtype),
        GK: T.Tensor((512,), dtype),
        Slot: T.Tensor((batch, 1), T.int32),
        CacheRows: T.Tensor((pages * page_size, 132), T.int32),
        Norm: T.Tensor((batch, 1536), dtype),
    ):
        with T.Kernel(batch) as row:
            src = T.alloc_shared((1, 2048), dtype)
            dst = T.alloc_shared((1, 1536), dtype)
            T.copy(QKV[row : row + 1, :], src)
            norm(src, GQ, GK, Slot, CacheRows, row, 0, dst)
            T.copy(dst, Norm[row : row + 1, :])

    return main
