# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 compact kernel for one channel-contiguous BF16 convolution cache."""

import tilelang
from tilelang import language as T


def build(rows, pages, page_stride, tile, index_stride=1, channels=10240, history=9, bulk=False, index_dtype="int32"):
    capacity = history + 3
    if channels % tile or tile % 16 or page_stride < capacity * channels:
        raise ValueError("unsupported compact cache geometry")
    partitions = channels // tile
    jobs = partitions * rows

    @T.prim_func
    def compact(
        cache: T.Tensor(((pages - 1) * page_stride + capacity * channels,), "bfloat16"),
        slots: T.Tensor(((rows - 1) * index_stride + 1,), index_dtype),
        lengths: T.Tensor((rows,), "int32"),
        finals: T.Tensor((rows,), "int32"),
        bounds: T.Tensor((rows,), "int32"),
        recovers: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel((jobs + 1) // 2, is_npu=True) as (core, sub):  # noqa: SIM117 - TileLang parses nested scopes
            with T.Scope("V"):
                plane = T.decl_buffer(
                    (pages, capacity, channels), "bfloat16", data=cache.data, strides=(page_stride, channels, 1)
                )
                job = core * 2 + sub
                if job < jobs:
                    row = job // partitions
                    lo = (job % partitions) * tile
                    source = T.Cast("int64", slots[row * index_stride])
                    length = lengths[row]
                    final = T.Cast("int64", finals[row])
                    boundary = T.Cast("int64", bounds[row])
                    recovery = recovers[row]
                    data = T.alloc_ub((history, tile), "bfloat16")
                    snapshot = T.alloc_ub((history, tile), "bfloat16")
                    if source > 0 and length > 0 and final > 0:
                        # Snapshot all source history before writing potentially aliased pages.
                        if bulk:
                            T.copy(plane[source, length - 1 : length - 1 + history, lo : lo + tile], data)
                            if boundary > 0:
                                T.copy(plane[source, recovery - 1 : recovery - 1 + history, lo : lo + tile], snapshot)
                                T.copy(snapshot, plane[boundary, 0:history, lo : lo + tile])
                            T.copy(data, plane[final, 0:history, lo : lo + tile])
                        else:
                            for h in T.unroll(history):
                                start = source * page_stride + (length - 1 + h) * channels + lo
                                T.copy(cache[start : start + tile], data[h, :])
                            if boundary > 0:
                                for h in T.unroll(history):
                                    start = source * page_stride + (recovery - 1 + h) * channels + lo
                                    T.copy(cache[start : start + tile], snapshot[h, :])
                                for h in T.unroll(history):
                                    start = boundary * page_stride + h * channels + lo
                                    T.copy(snapshot[h, :], cache[start : start + tile])
                            for h in T.unroll(history):
                                start = final * page_stride + h * channels + lo
                                T.copy(data[h, :], cache[start : start + tile])

    return tilelang.compile(
        compact,
        target="ascendc",
        platform="A3",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
