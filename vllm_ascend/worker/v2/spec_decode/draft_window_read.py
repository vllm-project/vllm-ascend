# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Canonical direct-prefix read projections, independent of query and KV writes.

The caller supplies exact CPU lengths from the same preparation as device
lengths. An upper bound is never used to select a start or to prove residency.
Each projection owns its read storage; callers retain it until all native
readers complete. No request mapping or projected table is cached here.
"""

from collections.abc import Callable
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
class NativeSWASource:
    """Canonical logical table for an unchanged causal native-SWA consumer.

    Old columns may be null/evicted. Only the consumer's native band is
    certified resident; an extra Window intersects that band, never extends it.
    valid_columns is the logical table extent, not a resident Full prefix.
    """

    table: torch.Tensor
    request_rows: tuple[int, ...]
    valid_columns: tuple[int, ...]
    block_span: int
    native_window: int

    @classmethod
    def from_block_tables(cls, owner: Any, gid: int, request_rows: np.ndarray, native_window: int):
        rows = tuple(int(row) for row in request_rows)
        table = owner.block_tables[gid].gpu
        if native_window <= 0 or any(row < 0 or row >= table.shape[0] for row in rows):
            raise ValueError("Native SWA requires a valid current canonical mapping and positive window")
        counts = tuple(int(owner.num_blocks.np[gid, row]) for row in rows)
        return cls(table, rows, counts, int(owner.kernel_block_sizes[gid]), native_window)


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
        common_metadata._draft_window_read = self

    def exact_cpu_for(self, common_metadata: Any) -> torch.Tensor:
        """Certify the snapshot only while all of its read operands are intact."""
        if (
            common_metadata.seq_lens is not self.seq_lens
            or common_metadata.seq_lens_cpu is not self.seq_lens_cpu
            or common_metadata._seq_lens_cpu is not self.seq_lens_cpu
            or common_metadata.block_table_tensor is not self.block_table
        ):
            raise ValueError("Draft Window read operands were replaced after projection")
        return self.seq_lens_cpu


def _materialize_canonical_read(
    source: DirectPrefixSource | NativeSWASource,
    *,
    exact_cpu: torch.Tensor,
    exact_device: torch.Tensor,
    upper_cpu: torch.Tensor,
    starts_cpu: torch.Tensor,
    padded_rows: int,
    row_indices: torch.Tensor | None = None,
    retain: Callable[..., None] | None = None,
) -> DraftReadView:
    """Gather from canonical rows, validating every column that can be read.

    CPU/device exact and row-mapping coherence is established by the current
    preparation caller. Reuse its device row mapping when available. Transfer
    S once and derive column bounds on device instead of copying those same
    values separately. This function performs no device-to-host transfer.
    Zero padding is only outside the exact readable extent; missing source
    columns are an error.

    An owned caller can retain the actual converted CPU start buffer and the
    destination before submitting its asynchronous H2D. Unowned leaf callers
    retain the original blocking behavior. No completion or later CPU read is
    introduced here; the existing proposal owner supplies reader completion.
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
    if row_indices is not None and (
        row_indices.shape != (n,) or row_indices.dtype != torch.int64 or row_indices.device != source.table.device
    ):
        raise ValueError("Prepared Draft row mapping must match the source device and request extent")
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
    counts = end_columns - starts // source.block_span
    width = max(1, int(counts.max()) if n else 0)
    if retain is None:
        starts_device = starts.to(exact_device.device)
    else:
        # Retain this exact DMA source, not a clone used only for metadata.
        # Allocate/retain the destination before copy_ can enqueue then fail.
        retain(source.table, exact_device, starts, row_indices)
        starts_device = torch.empty_like(starts, device=exact_device.device)
        retain(starts_device)
        starts_device.copy_(starts, non_blocking=True)
    # Own only the read projection, never clone query/write/Target metadata.
    table = source.table.new_zeros((padded_rows, width))
    if retain is not None:
        retain(table)
    if n and source.table.shape[1]:
        offsets = torch.arange(width, device=source.table.device)
        start_columns = starts_device // source.block_span
        end_columns_device = torch.div(
            exact_device[:n].to(torch.int64) + source.block_span - 1, source.block_span, rounding_mode="floor"
        )
        indices = start_columns[:, None] + offsets
        mask = offsets[None, :] < (end_columns_device - start_columns)[:, None]
        # Masked columns are not readable; clamp only their gather address.
        indices = indices.clamp(max=source.table.shape[1] - 1)
        rows = row_indices
        if rows is None:
            rows = torch.tensor(source.request_rows, dtype=torch.int64, device=source.table.device)
        if source.table.is_contiguous():
            # Current CANN lowers advanced indexing to an AICPU path. A flat
            # index_select reads just the projected columns of this layout.
            flat_indices = rows[:, None] * source.table.shape[1] + indices
            selected = source.table.view(-1).index_select(0, flat_indices.reshape(-1)).view(n, width)
        else:
            # Preserve the original strided-table contract without flattening
            # or copying the canonical table into contiguous storage.
            selected = source.table[rows[:, None], indices]
        table[:n] = torch.where(mask, selected, 0)
    lengths_cpu = torch.zeros(padded_rows, dtype=exact_cpu.dtype)
    lengths_cpu[:n] = ends - starts
    upper_out = torch.zeros(padded_rows, dtype=upper_cpu.dtype)
    upper_out[:n] = upper - starts
    lengths = exact_device.new_zeros(padded_rows)
    if retain is not None:
        retain(lengths)
    lengths[:n] = exact_device[:n] - starts_device
    return DraftReadView(
        table,
        lengths,
        lengths_cpu,
        upper_out,
        starts.clone(),
        int(upper_out.max()) if padded_rows else 0,
    )


def materialize_direct_prefix(
    source: DirectPrefixSource,
    *,
    exact_cpu: torch.Tensor,
    exact_device: torch.Tensor,
    upper_cpu: torch.Tensor,
    starts_cpu: torch.Tensor,
    padded_rows: int,
    row_indices: torch.Tensor | None = None,
    retain: Callable[..., None] | None = None,
) -> DraftReadView:
    """Project a resident Full prefix using the existing canonical copy path."""
    if not isinstance(source, DirectPrefixSource):
        raise ValueError("Full-prefix projection requires its resident-prefix source")
    return _materialize_canonical_read(
        source,
        exact_cpu=exact_cpu,
        exact_device=exact_device,
        upper_cpu=upper_cpu,
        starts_cpu=starts_cpu,
        padded_rows=padded_rows,
        row_indices=row_indices,
        retain=retain,
    )


def materialize_native_swa_intersection(
    source: NativeSWASource,
    *,
    exact_cpu: torch.Tensor,
    exact_device: torch.Tensor,
    upper_cpu: torch.Tensor,
    starts_cpu: torch.Tensor,
    padded_rows: int,
    query_width: int,
    row_indices: torch.Tensor | None = None,
    retain: Callable[..., None] | None = None,
) -> DraftReadView:
    """Rebase a certified native read interval without claiming its evicted prefix.

    After cropping at S, native right-aligned q coordinates move by -S and
    native pre_tokens=P is unchanged. Mapping back yields original band
    intersected with [S,E). Thus no newly required resident key is introduced.
    The caller must retain its actual mask. An unmasked native consumer must
    separately certify that every key in [S,E) remains resident; it cannot
    inherit the causal-band proof above merely from the same KV spec.
    """
    if not isinstance(source, NativeSWASource) or source.native_window <= 0 or query_width <= 0:
        raise ValueError("Native SWA projection requires its typed positive band")
    if exact_cpu.device.type != "cpu" or starts_cpu.device.type != "cpu":
        raise ValueError("Native SWA planning reuses the completed native builder CPU snapshot")
    if bool((exact_cpu - starts_cpu < query_width).any()):
        raise ValueError("Extra Window must cover every native query")
    return _materialize_canonical_read(
        source,
        exact_cpu=exact_cpu,
        exact_device=exact_device,
        upper_cpu=upper_cpu,
        starts_cpu=starts_cpu,
        padded_rows=padded_rows,
        row_indices=row_indices,
        retain=retain,
    )
