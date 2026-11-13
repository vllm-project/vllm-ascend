# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query-row partitioning for the replicated V4.1 index K cache."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IndexerRowPartition:
    sizes: tuple[int, ...]
    rank: int

    @property
    def start(self) -> int:
        return sum(self.sizes[: self.rank])

    @property
    def stop(self) -> int:
        return self.start + self.sizes[self.rank]

    @property
    def rows(self) -> slice:
        return slice(self.start, self.stop)


def plan_indexer_rows(
    num_rows: int,
    tp_size: int,
    tp_rank: int,
    *,
    topk: int,
    ratio: int,
    full_graph: bool,
    query_offsets: list[int] | None = None,
    seq_lens: list[int] | None = None,
) -> IndexerRowPartition | None:
    if tp_size < 2 or num_rows < tp_size:
        return None
    width, extra = divmod(num_rows, tp_size)
    equal_sizes = tuple(width + (rank < extra) for rank in range(tp_size))
    if full_graph or query_offsets is None or seq_lens is None:
        # The FULL graph must choose exactly the same layout at capture and replay.
        return IndexerRowPartition(equal_sizes, tp_rank)
    if query_offsets[-1] < num_rows or len(seq_lens) + 1 != len(query_offsets):
        return None

    # The CPU sequence lengths may be optimistic after draft rejection. Use
    # them only to estimate work; the operator receives exact device lengths.
    clamped_offsets = np.minimum(query_offsets, num_rows)
    query_lens = np.diff(clamped_offsets)
    if clamped_offsets[0] != 0 or (query_lens < 0).any():
        return None
    prefix_lens = np.asarray(seq_lens) - query_lens
    row_positions = np.repeat(prefix_lens - clamped_offsets[:-1], query_lens) + np.arange(1, num_rows + 1)
    work = np.cumsum(np.maximum(row_positions, 0) // ratio, dtype=np.int64)
    # Favor replicated scoring when each row reads fewer keys on average than
    # it would exchange indices. This is a cost heuristic, independent of TP.
    total_work = int(work[-1])
    if total_work < num_rows * topk:
        return None
    # Limit the padded AllGather to about 1.25x the equal-row capacity.
    equal_width = max(equal_sizes)
    max_rows = equal_width + (equal_width + 3) // 4
    bounds = [0]
    for rank in range(1, tp_size):
        remaining_ranks = tp_size - rank
        target = int(work.searchsorted(total_work * rank // tp_size)) + 1
        earliest = max(bounds[-1] + 1, num_rows - remaining_ranks * max_rows)
        latest = min(bounds[-1] + max_rows, num_rows - remaining_ranks)
        bounds.append(min(max(target, earliest), latest))
    bounds.append(num_rows)
    return IndexerRowPartition(tuple(end - start for start, end in zip(bounds, bounds[1:])), tp_rank)


def rebase_indexer_lengths(
    query_offsets: torch.Tensor,
    seq_lens: torch.Tensor,
    partition: IndexerRowPartition,
    num_layout_rows: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Rebase the request segments without changing their physical K pages."""
    offsets = query_offsets.clamp(0, num_layout_rows)
    local_offsets = (offsets - partition.start).clamp(0, partition.stop - partition.start)
    dropped_suffix = (offsets[1:] - partition.stop).clamp_min(0)
    local_lengths = (seq_lens - dropped_suffix).clamp_min(0)
    local_lengths = local_lengths.masked_fill(local_offsets[1:] == local_offsets[:-1], 0)
    if ratio == 1:
        return local_offsets, local_lengths, None
    return local_offsets, local_lengths.div(ratio, rounding_mode="floor"), local_lengths.remainder(ratio)


def gather_indexer_rows(local: torch.Tensor, partition: IndexerRowPartition, group) -> torch.Tensor:
    max_rows = max(partition.sizes)
    if local.shape[0] < max_rows:
        local = F.pad(local, (0, 0, 0, max_rows - local.shape[0]))
    gathered = group.all_gather(local.contiguous(), dim=0)
    if all(size == max_rows for size in partition.sizes):
        return gathered
    return torch.cat([gathered[rank * max_rows : rank * max_rows + size] for rank, size in enumerate(partition.sizes)])
