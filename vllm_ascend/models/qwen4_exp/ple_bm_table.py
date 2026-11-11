# SPDX-License-Identifier: Apache-2.0
"""Load and attach a Qwen PLE table through a shared BM owner/client."""

import torch

from .ple_bm_layout import mapping_pointers, publish_rows
from .ple_checkpoint import PLECheckpoint
from .ple_layout import PLELayout


def backing_rows(layout: PLELayout) -> int:
    """Express padded capacity as rows while preserving BM allocation granularity."""
    return layout.allocation_bytes // layout.row_bytes


def load_table(owner, checkpoint: PLECheckpoint, key: str, *, chunk_rows: int = 262144) -> PLELayout:
    """Populate one unpublished pool table, using bounded CPU staging."""
    if chunk_rows <= 0:
        raise ValueError("PLE staging rows must be positive")
    layout = PLELayout(checkpoint.rows, checkpoint.width * 2)
    buffer = owner.add(key, backing_rows(layout), checkpoint.width, torch.bfloat16)
    if buffer.size < layout.allocation_bytes:
        raise ValueError("BM allocation is too small for the padded PLE layout")
    stage = torch.empty((min(chunk_rows, checkpoint.rows), checkpoint.width), dtype=torch.bfloat16, device="cpu")
    for start in range(0, checkpoint.rows, len(stage)):
        rows = stage[: min(len(stage), checkpoint.rows - start)]
        checkpoint.copy_rows(rows, start)
        publish_rows(buffer, layout, start, rows)
    return layout


def attach_table(client, key: str, layout: PLELayout, width: int) -> torch.Tensor:
    """Attach and return pointers; caller must stop graph access before client close."""
    if width <= 0 or width * 2 != layout.row_bytes:
        raise ValueError("PLE width does not match the table layout")
    # A one-row view establishes the existing client's lease and mapping without
    # asking its DeepSeek-specific chunk view to span padded slab boundaries.
    client.view(key, 0, 1, rows=backing_rows(layout), width=width, dtype=torch.bfloat16)
    return mapping_pointers(client.buffers[key], layout)
