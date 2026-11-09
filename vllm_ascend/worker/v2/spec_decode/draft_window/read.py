# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def window_capacity_tokens(window_size: int, block_span_tokens: int) -> int:
    """Allocation bound for every start aligned to the consumer's block span."""
    return window_size + block_span_tokens - 1


def compute_draft_window_start(
    *,
    full_exact_end: int,
    window_size: int,
    block_span_tokens: int,
    future_offset: int = 0,
) -> int:
    """Return S=floor(max(E+future-W,0)/B)*B in absolute tokens."""
    start = max(full_exact_end + future_offset - window_size, 0) // block_span_tokens * block_span_tokens
    if start > full_exact_end:
        raise ValueError("window start exceeds exact end; window is smaller than the proposal horizon requires")
    return start


def materialize_draft_read(
    common_metadata: Any,
    valid_columns: Sequence[int],
    block_span: int,
    *,
    exact_cpu: torch.Tensor,
    starts_cpu: torch.Tensor,
    retain: Callable[..., None],
) -> None:
    """Copy only initialized [S,E) columns from the current compact Full table.

    Runtime supplies typed CPU snapshots; the fixed builder and BlockTables
    supply same-device int32 inputs already gathered into batch order. Each
    copied row and the exact-length DMA source belong to this proposal, so a
    later batch or Eagle consumer cannot overwrite an outstanding reader.
    """
    n = len(valid_columns)
    block_table = common_metadata.block_table_tensor
    exact_device = common_metadata.seq_lens
    upper_cpu = common_metadata.seq_lens_cpu_upper_bound[:n]
    padded_rows = common_metadata.num_reqs
    if block_table.shape[0] < n or any(count < 0 or count > block_table.shape[1] for count in valid_columns):
        raise ValueError("Invalid initialized compact column extent")
    cpu_values = (exact_cpu, upper_cpu, starts_cpu)
    if any(t.numel() != n for t in cpu_values):
        raise ValueError("Draft read planning requires one value per active request")
    if padded_rows < n or exact_device.numel() < n:
        raise ValueError("Missing device exact lengths or padded row extent")
    ends, upper, starts = (t.to(torch.int64) for t in cpu_values)
    if bool(((ends < 0) | (upper < ends) | (starts < 0) | (starts > ends)).any()):
        raise ValueError("Invalid exact, upper, or start ordering")
    if bool((starts % block_span != 0).any()):
        raise ValueError("This consumer cannot represent an intra-block read start")
    end_columns = (ends + block_span - 1) // block_span
    if any(count < int(end) for count, end in zip(valid_columns, end_columns)):
        raise ValueError("Compact source does not cover the exact readable extent")
    start_columns = starts // block_span
    counts = end_columns - start_columns
    width = max(1, int(counts.max()) if n else 0)
    table = block_table.new_zeros((padded_rows, width))
    retain(block_table, exact_device, table)
    for row, (start, count) in enumerate(zip(start_columns.tolist(), counts.tolist())):
        if count:
            # Both complete storages are already owned; row views add no
            # independent lifetime to this current-stream copy.
            table[row, :count].copy_(block_table[row, start : start + count], non_blocking=True)
    lengths_cpu = torch.zeros(
        padded_rows, dtype=exact_cpu.dtype, device="cpu", pin_memory=block_table.device.type == "npu"
    )
    lengths_cpu[:n] = ends - starts
    lengths = exact_device.new_empty(padded_rows)
    # Retain both actual operands before a copy can partially submit and fail.
    retain(lengths_cpu, lengths)
    lengths.copy_(lengths_cpu, non_blocking=True)
    # Once E is completed, its visible lengths are also the tight allocation
    # bounds. Query extents and positions remain owned by the original builder.
    common_metadata.block_table_tensor = table
    common_metadata.seq_lens = lengths
    common_metadata.seq_lens_cpu = common_metadata._seq_lens_cpu = lengths_cpu
    common_metadata.seq_lens_cpu_upper_bound = lengths_cpu
    common_metadata.max_seq_len = int(lengths_cpu.max()) if padded_rows else 0
    common_metadata._draft_window_exact_cpu = lengths_cpu
