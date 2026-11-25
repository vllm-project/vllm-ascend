# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T


def mask_graph_lengths(batch, offset):
    @T.prim_func
    def main(
        Length: T.Tensor((batch,), T.int32),
        Slot: T.Tensor((batch, 1), T.int32),
        Effective: T.Tensor((offset + batch,), T.int32),
    ):
        with T.Kernel(T.ceildiv(batch, 32)) as block:
            first = block * 32
            with T.SimtVF(threads=32):
                for lane in T.Parallel(32):
                    row = first + lane
                    if row < batch:
                        Effective[offset + row] = T.if_then_else(Slot[row, 0] >= 0, Length[row], 0)

    return main


def expand_selected(batch, width, length_offset=0):
    @T.prim_func
    def main(
        Pools: T.Tensor((batch, 512), T.int32),
        Length: T.Tensor((length_offset + batch,), T.int32),
        Packed: T.Tensor((batch, width), T.int32),
    ):
        with T.Kernel(batch) as r:
            pools = T.alloc_shared((512,), T.int32)
            expanded = T.alloc_shared((2176,), T.int32)
            T.copy(Pools[r, :], pools)
            tokens = T.max(Length[length_offset + r], 0)
            complete = T.min(tokens // 4, 512) * 4
            with T.SimtVF(threads=256):
                for i in T.Parallel(2176):
                    if i < complete:
                        expanded[i] = pools[T.min(i // 4, 511)] * 4 + i % 4
                    elif i < complete + tokens % 4:
                        expanded[i] = tokens // 4 * 4 + i - complete
                    else:
                        expanded[i] = -1
            T.copy(expanded[:width], Packed[r, :])

    return main
