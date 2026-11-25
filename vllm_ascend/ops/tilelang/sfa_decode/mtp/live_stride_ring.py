# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="valid-type"

import tilelang.ascend.language as T  # type: ignore[import-not-found]

from ..shared.keypool import pool_four_keys


def update_program(requests, pages, ring_pages, tokens, retained_width=6, *, ring_table_stride=1):
    capacity = 9

    @T.prim_func
    def main(
        Current: T.Tensor((tokens, 256), T.float32),
        Projected: T.Tensor((tokens, 256), T.float32),
        Pos: T.Tensor((tokens,), T.int64),
        RingPage: T.StridedTensor((requests,), (ring_table_stride,), T.int32),
        Ring: T.StridedTensor((ring_pages, capacity, 256), (289 * 256, 256, 1), T.float32),
        Starts: T.Tensor((requests + 1,), T.int32),
        IndexSlots: T.Tensor((tokens,), T.int64),
        Writes: T.Tensor((requests, retained_width), T.int64),
        Ape: T.Tensor((4, 128), T.float32),
        K: T.Tensor((pages, 34, 128), T.float8_e4m3fn),
        KS: T.StridedTensor((pages, 32), (1088, 1), T.float32),
    ):
        with T.Kernel(requests) as bx:
            ring = RingPage[bx]
            begin = Starts[bx]
            live = Starts[bx + 1] - begin
            window = T.alloc_shared((9, 256), T.float32)
            rollback = T.alloc_shared((6, 256), T.float32)
            ape = T.alloc_shared((4, 128), T.float32)
            bf = T.alloc_shared((128,), T.bfloat16)
            packed = T.alloc_shared((128,), T.float8_e4m3fn)
            scale = T.alloc_shared((8,), T.float32)
            if live > 0:
                first_position = T.cast(Pos[begin], T.int32)
                history = first_position % 4
                T.copy(Ape[:, :], ape)
                for j in T.serial(3):
                    if 3 - j <= history:
                        T.copy(Ring[T.max(ring, 0), (first_position - 3 + j) % capacity, :], window[j, :])
                for j in T.serial(live):
                    T.copy(Current[begin + j, :], window[3 + j, :])
                    T.copy(Projected[begin + j, :], rollback[j, :])
                for j in T.serial(live):
                    if retained_width - live + j >= 0:
                        dest = Writes[bx, retained_width - live + j]
                        if dest >= 0 and dest < ring_pages * 289 and dest % 289 < capacity:
                            T.copy(rollback[j, :], Ring[dest // 289, dest % 289, :])
                    slot = IndexSlots[begin + j]
                    if slot >= 0 and slot < pages * 32:
                        pool_four_keys(window, j, window, j + 3, ape, bf, packed, scale)
                        page = slot // 32
                        T.copy(packed, K[page, slot % 32, :])
                        with T.SimtVF(threads=32):
                            for c in T.Parallel(1):
                                KS[page, slot % 32] = scale[c]

    return main
