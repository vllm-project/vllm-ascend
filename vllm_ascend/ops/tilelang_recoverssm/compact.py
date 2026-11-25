# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate: copy contiguous-channel histories with two DMA staging slots.

Checkpoint and convolution cache ABIs are unchanged. Both histories are loaded
before any destination is overwritten, including source/destination aliasing.
Non-contiguous channels retain a scalar SIMT path for correctness.
"""

import tilelang.ascend.language as T
from tilelang.ascend.language.kernel import get_thread_binding


def compact_geometry(channels, hist, conv_dtype, align, threads=256):
    """Budget at most 128 KiB UB; use a whole channel row when it fits.

    Two rotating slots each hold final and, for align, boundary history.
    Round to a SIMT thread block so the existing strided fallback has no
    uncovered channels. No runtime device query or cache layout change.
    """
    item_bytes = 2 if conv_dtype in ("bfloat16", "float16") else 4
    histories = 2 if align else 1
    bytes_per_channel = 2 * histories * hist * item_bytes
    max_channels = (128 * 1024 // bytes_per_channel // threads) * threads
    if max_channels < threads:
        raise ValueError("convolution history exceeds the candidate UB budget")
    tile = min(((channels + threads - 1) // threads) * threads, max_channels)
    return tile, tile * bytes_per_channel


def compact_kernel(
    batch,
    layers,
    pages,
    hv,
    window,
    channels,
    hist,
    capacity,
    index_layout,
    state_dtype,
    conv_dtype,
    ss,
    rs,
    align,
    conv,
    value_tile=128,
    threads=256,
    cap=64,
    channel_tile=None,
    shared_dim_stride=None,
):
    if channel_tile is None:
        channel_tile, _ = compact_geometry(channels, hist, conv_dtype, align, threads)
    size, index_stride, index_dtype = index_layout
    chunks = (channels + channel_tile - 1) // channel_tile
    tasks = batch * layers * chunks
    blocks = min(cap, tasks)

    @T.prim_func
    def compact(
        ST: T.Tensor((4 * layers,), "int64"),
        CT: T.Tensor((4 * layers,), "int64"),
        PageIds: T.Tensor((size,), index_dtype),
        P: T.Tensor((6 * capacity,), "int32"),
    ):
        with T.Kernel(blocks) as block:  # noqa: SIM117 - TileLang scopes
            final_ub = T.alloc_shared((2, hist, channel_tile), conv_dtype)
            boundary_ub = T.alloc_shared((2, hist, channel_tile), conv_dtype)
            for slot in T.unroll(2):
                T.ascend_set_flag("MTE3_MTE2", slot)
            for wave in T.serial((tasks - 1 - block) // blocks + 1):
                slot = wave % 2
                task = wave * blocks + block
                row = task // (layers * chunks)
                layer = (task // chunks) % layers
                cbase = (task % chunks) * channel_tile
                width = T.min(channel_tile, channels - cbase)
                count = P[row]
                if count > 0:
                    src = T.cast(PageIds[row * index_stride[0]], "int64")
                    dst = T.cast(P[3 * capacity + row], "int64")
                    mid = T.cast(P[4 * capacity + row], "int64") if align else T.int64(0)
                    cut = P[5 * capacity + row] if align else 0
                    boundary = align and mid > 0 and mid < pages and cut > 0 and cut <= count and mid != dst
                    if src > 0 and src < pages and dst > 0 and dst < pages:
                        if CT[2 * layers + layer] == 1 if shared_dim_stride is None else shared_dim_stride == 1:
                            # Reinterpret the metadata as [page, token, channel],
                            # with contiguous channel runs; no GM layout changes.
                            cache = T.make_tensor_from_addr(
                                CT[layer],
                                (pages, hist + window - 1, channels),
                                conv_dtype,
                                strides=(CT[layers + layer], CT[3 * layers + layer], 1),
                            )
                            T.ascend_wait_flag("MTE3_MTE2", slot)
                            T.copy(
                                cache[src, count - 1 : count - 1 + hist, cbase : cbase + width],
                                final_ub[slot, :, 0:width],
                            )
                            if boundary and cut < count:
                                T.copy(
                                    cache[src, cut - 1 : cut - 1 + hist, cbase : cbase + width],
                                    boundary_ub[slot, :, 0:width],
                                )
                            T.ascend_set_flag("MTE2_MTE3", slot)
                            T.ascend_wait_flag("MTE2_MTE3", slot)
                            if boundary:
                                if cut == count:
                                    T.copy(final_ub[slot, :, 0:width], cache[mid, 0:hist, cbase : cbase + width])
                                else:
                                    T.copy(boundary_ub[slot, :, 0:width], cache[mid, 0:hist, cbase : cbase + width])
                            T.copy(final_ub[slot, :, 0:width], cache[dst, 0:hist, cbase : cbase + width])
                            T.ascend_set_flag("MTE3_MTE2", slot)
                        else:
                            cache_fallback = T.make_tensor_from_addr(
                                CT[layer],
                                (pages, channels, hist + window - 1),
                                conv_dtype,
                                strides=(CT[layers + layer], CT[2 * layers + layer], CT[3 * layers + layer]),
                            )
                            with T.SimtVF(threads=threads):
                                tid = get_thread_binding(0)
                                for ci in T.unroll(channel_tile // threads):
                                    channel = cbase + tid + ci * threads
                                    if channel < channels:
                                        final = T.alloc_local((hist,), conv_dtype)
                                        saved = T.alloc_local((hist,), conv_dtype)
                                        for j in T.unroll(hist):
                                            final[j] = cache_fallback[src, channel, count - 1 + j]
                                            if boundary:
                                                saved[j] = cache_fallback[src, channel, cut - 1 + j]
                                        if boundary:
                                            for j in T.unroll(hist):
                                                cache_fallback[mid, channel, j] = saved[j]
                                        for j in T.unroll(hist):
                                            cache_fallback[dst, channel, j] = final[j]
                            T.ascend_pipe_barrier("PIPE_ALL")
            for slot in T.unroll(2):
                T.ascend_wait_flag("MTE3_MTE2", slot)

    return compact.with_attr(
        "global_symbol", "compact_dma_c" + str(channel_tile) + ("_contig" if shared_dim_stride == 1 else "_generic")
    )
