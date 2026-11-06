# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Batch group-local GDN graph input writes without changing captured storage."""

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

_BLOCK_SIZE = 256


@triton.jit(do_not_specialize=["actual_rows", "graph_rows"])
def _graph_state_kernel(
    pointers,
    actual_rows,
    graph_rows,
    WIDTH: tl.constexpr,
    SOURCE_ROW_STRIDE: tl.constexpr,
    SOURCE_COL_STRIDE: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    CLEAR_WIDTH: tl.constexpr,
    CLEAR_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    source = tl.load(pointers + group * 3).to(tl.pointer_type(tl.int32))
    destination = tl.load(pointers + group * 3 + 1).to(tl.pointer_type(tl.int32))
    for tile in range(tl.cdiv(graph_rows * WIDTH, BLOCK)):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        rows = offsets // WIDTH
        columns = offsets % WIDTH
        values = tl.load(
            source + rows.to(tl.int64) * SOURCE_ROW_STRIDE + columns * SOURCE_COL_STRIDE,
            (offsets < graph_rows * WIDTH) & (rows < actual_rows),
            other=PAD_VALUE,
        )
        # One program owns each allocation, including its partial 32-byte tail.
        tl.store(destination + offsets, values, offsets < graph_rows * WIDTH)
    if CLEAR_WIDTH:
        clear = tl.load(pointers + group * 3 + 2).to(tl.pointer_type(tl.int32))
        for tile in range(tl.cdiv(graph_rows * CLEAR_WIDTH, BLOCK)):
            offsets = tile * BLOCK + tl.arange(0, BLOCK)
            tl.store(clear + offsets, CLEAR_VALUE, offsets < graph_rows * CLEAR_WIDTH)


class GDNGraphStateUpdater:
    """Cache only pointer metadata; request counts and contents refresh each step."""

    def __init__(self):
        self._signature = None
        self._pointers = None
        self._buffers = None

    def apply(self, updates):
        if not updates:
            return
        source, destination, actual_rows, graph_rows, padding, clear = updates[0]
        width = destination.size(1) if destination.ndim == 2 else 1
        source_row_stride = source.stride(0)
        source_col_stride = source.stride(1) if source.ndim == 2 else 1
        clear_width = 0 if clear is None else clear.size(1)
        device = destination.device
        # Validate every group before any writes are submitted.
        assert len({dst.data_ptr() for _, dst, *_ in updates}) == len(updates)
        for src, dst, actual, graph, pad, reset in updates:
            assert src.ndim in (1, 2) and src.ndim == dst.ndim
            assert src.dtype == dst.dtype == torch.int32
            assert src.device == dst.device == device and dst.is_contiguous()
            assert actual == actual_rows and graph == graph_rows and pad == padding
            assert 0 <= actual <= graph <= dst.size(0) and actual <= src.size(0)
            assert (src.size(1) if src.ndim == 2 else 1) >= width
            assert (dst.size(1) if dst.ndim == 2 else 1) == width
            assert src.stride(0) == source_row_stride
            assert (src.stride(1) if src.ndim == 2 else 1) == source_col_stride
            assert (reset is None) == (clear is None)
            if reset is not None:
                assert reset.device == device and reset.dtype == torch.int32 and reset.is_contiguous()
                assert reset.size(1) == clear_width and graph <= reset.size(0)
        if device.type == "cpu":
            for src, dst, actual, graph, pad, reset in updates:
                values = src[:actual, :width] if src.ndim == 2 else src[:actual]
                dst[:actual].copy_(values)
                dst[actual:graph].fill_(pad)
                if reset is not None:
                    reset[:graph].fill_(PAD_SLOT_ID)
            return
        signature = tuple(
            (src.data_ptr(), dst.data_ptr(), 0 if reset is None else reset.data_ptr())
            for src, dst, _, _, _, reset in updates
        )
        if self._signature != signature:
            host = torch.tensor(signature, dtype=torch.int64, pin_memory=True)
            self._pointers = host.to(device, non_blocking=True)
            # Retain allocations referenced by the cached pointer table.
            self._buffers = tuple((src, dst, reset) for src, dst, _, _, _, reset in updates)
            self._signature = signature
        if graph_rows:
            _graph_state_kernel[(len(updates),)](
                self._pointers,
                actual_rows,
                graph_rows,
                WIDTH=width,
                SOURCE_ROW_STRIDE=source_row_stride,
                SOURCE_COL_STRIDE=source_col_stride,
                PAD_VALUE=padding,
                CLEAR_WIDTH=clear_width,
                CLEAR_VALUE=PAD_SLOT_ID,
                BLOCK=_BLOCK_SIZE,
                multibuffer=False,
                unit_flag=False,
            )
