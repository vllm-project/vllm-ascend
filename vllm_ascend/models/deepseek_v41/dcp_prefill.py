# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import dataclass
from typing import Protocol

import torch


class DCPGroup(Protocol):
    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor: ...


@dataclass(frozen=True)
class DCPReadLayout:
    """A request's sharded physical pages and its dense global attention page table.

    With one full physical block per DCP stripe, ranks store alternating
    logical pages. The materialized read view is transient; connector and
    scheduler block IDs always address the original rank-local cache.
    """

    local_page_ids: torch.Tensor
    block_table: torch.Tensor

    @classmethod
    def build(cls, local_block_table, seq_lens, max_seq_len: int, block_size: int, world_size: int):
        """Pack active rows on each rank. All ranks use the same global sequence lengths."""
        stripe = block_size * world_size
        width = (max_seq_len + stripe - 1) // stripe
        if width > local_block_table.size(1):
            raise ValueError("DCP block table does not cover the global sequence length")
        counts = (seq_lens.to(torch.long) + stripe - 1) // stripe
        columns = torch.arange(width, device=local_block_table.device)
        active = columns < counts[:, None]
        physical = local_block_table[:, :width][active].long()
        # all_gather concatenates rank buffers. Express stripe order in the
        # page table so kernels can consume the gathered allocation directly.
        offsets = counts.cumsum(dim=0) - counts
        logical = torch.arange(width * world_size, device=columns.device)
        table = offsets[:, None] + logical // world_size + (logical % world_size) * physical.numel()
        valid = logical * block_size < seq_lens[:, None]
        return cls(physical, table.masked_fill(~valid, 0).int())

    def gather(self, cache: torch.Tensor, group: DCPGroup) -> torch.Tensor:
        local = cache.index_select(0, self.local_page_ids)
        if not local.size(0):
            return local
        return group.all_gather(local, dim=0)
