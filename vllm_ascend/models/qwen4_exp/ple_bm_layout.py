# SPDX-License-Identifier: Apache-2.0
"""Map padded PLE row slabs into an owner-backed BM allocation."""

import ctypes

import torch

from .ple_layout import PLELayout


def publish_rows(buffer, layout: PLELayout, start: int, rows: torch.Tensor) -> None:
    """Write rows to BM host addresses, respecting each slab's padding."""
    if not buffer.owner or buffer.closed or not buffer._ready:
        raise ValueError("PLE publication requires a live owner mapping")
    if rows.device.type != "cpu" or rows.dtype != torch.bfloat16 or rows.ndim != 2:
        raise ValueError("PLE rows must be a 2D CPU BF16 tensor")
    if rows.shape[1] * rows.element_size() != layout.row_bytes:
        raise ValueError("PLE row width does not match the mapping layout")
    if buffer.dtype != torch.bfloat16 or buffer.width != rows.shape[1]:
        raise ValueError("PLE mapping dtype or width does not match the rows")
    if layout.allocation_bytes > buffer.size:
        raise ValueError("PLE layout exceeds the backing allocation")
    rows = rows.contiguous()
    for left, right, offset in layout.spans(start, start + rows.shape[0]):
        address = buffer.handle.gva_to_va(buffer.gva + offset, buffer.runtime.bm.BmMemType.LOCAL_HOST)
        if not address:
            raise RuntimeError("PLE owner mapping is not CPU-addressable")
        ctypes.memmove(address, rows.data_ptr() + (left - start) * layout.row_bytes, (right - left) * layout.row_bytes)


def mapping_pointers(buffer, layout: PLELayout) -> torch.Tensor:
    """Return stable NPU pointers after checking each used slab is contiguous."""
    if buffer.closed or not buffer._ready or layout.allocation_bytes > buffer.size:
        raise ValueError("PLE mapping is closed or too small")
    pointers = []
    for start, end, offset in layout.spans(0, layout.rows):
        pointers.append(buffer.device_pointer(offset, (end - start) * layout.row_bytes))
    return torch.tensor(pointers, dtype=torch.int64, device=buffer.device)
