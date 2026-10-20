# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Canonical direct-prefix read projections, independent of query and KV writes.

The caller supplies exact CPU lengths from the same preparation as device
lengths. An upper bound is never used to select a start or to prove residency.
Each projection owns its read storage; callers retain it until all native
readers complete. No request mapping or projected table is cached here.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class DirectPrefixSource:
    """Borrowed canonical owner and a snapshot of the current request mapping.

    ``valid_columns`` counts kernel-addressed columns, not physical allocator
    blocks. This provider is only valid for resident, direct-prefix layouts;
    offloaded, circular, compressed and sharded layouts need their own provider.
    """

    table: torch.Tensor
    request_rows: tuple[int, ...]
    valid_columns: tuple[int, ...]
    block_span: int

    @classmethod
    def from_block_tables(cls, owner: Any, gid: int, request_rows: np.ndarray):
        # AscendBlockTables.append_block_ids expands physical blocks into
        # kernel columns before updating num_blocks (including on vLLM main).
        rows = tuple(int(row) for row in request_rows)
        table = owner.block_tables[gid].gpu
        if any(row < 0 or row >= table.shape[0] for row in rows):
            raise ValueError("Draft read source contains an inactive request row")
        counts = tuple(int(owner.num_blocks.np[gid, row]) for row in rows)
        return cls(table, rows, counts, int(owner.kernel_block_sizes[gid]))


@dataclass(frozen=True)
class DraftReadView:
    """Owned read operands; deliberately carries no positions or write slots."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_lens_cpu_upper_bound: torch.Tensor
    starts_cpu: torch.Tensor
    max_seq_len: int

    def apply(self, common_metadata: Any) -> None:
        """Apply only to a newly created draft common metadata object."""
        common_metadata.block_table_tensor = self.block_table
        common_metadata.seq_lens = self.seq_lens
        common_metadata.seq_lens_cpu = self.seq_lens_cpu
        common_metadata._seq_lens_cpu = self.seq_lens_cpu
        common_metadata.seq_lens_cpu_upper_bound = self.seq_lens_cpu_upper_bound
        common_metadata.max_seq_len = self.max_seq_len


def materialize_direct_prefix(
    source: DirectPrefixSource,
    *,
    exact_cpu: torch.Tensor,
    exact_device: torch.Tensor,
    upper_cpu: torch.Tensor,
    starts_cpu: torch.Tensor,
    padded_rows: int,
) -> DraftReadView:
    """Gather from canonical rows, validating every column that can be read.

    CPU/device exact coherence is established by the preparation caller. This
    function performs no device-to-host transfer. Zero padding is only outside
    the exact readable extent; missing source columns are an error.
    """
    n = len(source.request_rows)
    cpu_values = (exact_cpu, upper_cpu, starts_cpu)
    if any(t.device.type != "cpu" or t.ndim != 1 or t.numel() != n for t in cpu_values):
        raise ValueError("Draft read planning requires one CPU value per active request")
    if any(t.dtype not in (torch.int32, torch.int64) for t in cpu_values):
        raise ValueError("Draft read lengths and starts must be integers")
    if source.table.ndim != 2 or source.block_span <= 0 or padded_rows < n:
        raise ValueError("Invalid direct-prefix geometry or padded row extent")
    if exact_device.ndim != 1 or exact_device.numel() < n:
        raise ValueError("Missing device exact lengths")
    if exact_device.device != source.table.device:
        raise ValueError("Draft source and exact lengths must be on the same device")
    if len(source.valid_columns) != n:
        raise ValueError("Missing canonical valid-column counts")
    if any(row < 0 or row >= source.table.shape[0] for row in source.request_rows):
        raise ValueError("Draft source row is out of range")
    ends = exact_cpu.to(torch.int64)
    starts = starts_cpu.to(torch.int64)
    upper = upper_cpu.to(torch.int64)
    if bool(((ends < 0) | (upper < ends) | (starts < 0) | (starts > ends)).any()):
        raise ValueError("Invalid exact, upper, or start ordering")
    if bool((starts % source.block_span != 0).any()):
        raise ValueError("This consumer cannot represent an intra-block read start")
    end_columns = (ends + source.block_span - 1) // source.block_span
    valid = torch.tensor(source.valid_columns, dtype=torch.int64)
    if bool(((valid < end_columns) | (valid > source.table.shape[1])).any()):
        raise ValueError("Canonical source does not cover the exact readable extent")
    start_columns = starts // source.block_span
    counts = end_columns - start_columns
    width = max(1, int(counts.max()) if n else 0)
    # Own only the read projection, never clone query/write/Target metadata.
    table = source.table.new_zeros((padded_rows, width))
    if n and source.table.shape[1]:
        offsets = torch.arange(width, device=source.table.device)
        indices = start_columns.to(source.table.device)[:, None] + offsets
        mask = offsets[None, :] < counts.to(source.table.device)[:, None]
        # Masked columns are not readable; clamp only their gather address.
        indices = indices.clamp(max=source.table.shape[1] - 1)
        rows = torch.tensor(source.request_rows, device=source.table.device)[:, None]
        table[:n] = torch.where(mask, source.table[rows, indices], 0)
    lengths_cpu = torch.zeros(padded_rows, dtype=exact_cpu.dtype)
    lengths_cpu[:n] = ends - starts
    upper_out = torch.zeros(padded_rows, dtype=upper_cpu.dtype)
    upper_out[:n] = upper - starts
    lengths = exact_device.new_zeros(padded_rows)
    lengths[:n] = exact_device[:n] - starts.to(exact_device.device)
    return DraftReadView(
        table,
        lengths,
        lengths_cpu,
        upper_out,
        starts.clone(),
        int(upper_out.max()) if padded_rows else 0,
    )
