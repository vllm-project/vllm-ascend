# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private query buffers for request-sharded, replicated-context DFlash2.

This module deliberately has no vLLM/NPU imports. Packing and restoring request
order can be exercised with real CPU tensors without an inference worker.
"""

from contextlib import contextmanager
from copy import copy
from typing import Any

import torch


class PCPQueryBuffers:
    """Pack complete query blocks; never change the shared context/cache state."""

    def __init__(self, speculator: Any, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.query_width = speculator.num_query_per_req
        self.draft_width = speculator.num_speculative_steps
        self.capacity = (speculator.max_num_reqs + world_size - 1) // world_size
        token_capacity = self.capacity * self.query_width
        sample_capacity = self.capacity * self.draft_width
        source = speculator.input_buffers
        self.input_buffers = copy(source)
        self.input_buffers.max_num_reqs = self.capacity
        self.input_buffers.max_num_tokens = token_capacity
        for name in ("input_ids", "positions", "is_padding"):
            setattr(self.input_buffers, name, torch.empty_like(getattr(source, name)[:token_capacity]))
        for name in ("seq_lens", "dcp_local_seq_lens"):
            if hasattr(source, name):
                setattr(self.input_buffers, name, torch.empty_like(getattr(source, name)[: self.capacity]))
        self.input_buffers.query_start_loc = (
            torch.arange(self.capacity + 1, dtype=source.query_start_loc.dtype, device=source.query_start_loc.device)
            * self.query_width
        )
        self.sample_indices = (
            (
                torch.arange(self.capacity, device=speculator.sample_indices.device)[:, None] * self.query_width
                + torch.arange(1, self.draft_width + 1, device=speculator.sample_indices.device)
            )
            .flatten()
            .to(speculator.sample_indices.dtype)
        )
        self.sample_pos = torch.empty_like(speculator.sample_pos[:sample_capacity])
        self.sample_idx_mapping = torch.empty_like(speculator.sample_idx_mapping[:sample_capacity])
        self.draft_tokens = torch.empty_like(speculator.draft_tokens[: self.capacity])
        self.block_tables = copy(speculator.block_tables)
        # Keep non-draft table references read-only. Only draft groups consume
        # the selected rows; their storage is private to this query pass.
        self.block_tables.input_block_tables = list(speculator.block_tables.input_block_tables)
        self.group_ids = tuple(speculator.draft_kv_cache_group_ids)
        for gid in self.group_ids:
            self.block_tables.input_block_tables[gid] = torch.empty_like(
                speculator.block_tables.input_block_tables[gid][: self.capacity]
            )
        self.block_tables.slot_mappings = torch.full_like(speculator.block_tables.slot_mappings[:, :token_capacity], -1)

    def owned_count(self, num_reqs: int) -> int:
        return len(range(self.rank, num_reqs, self.world_size))

    def pack(self, speculator: Any, num_reqs: int) -> int:
        """Called only after the global context KV update has completed."""
        count = self.owned_count(num_reqs)
        rows = slice(self.rank, num_reqs, self.world_size)
        num_tokens = count * self.query_width
        for name in ("input_ids", "positions", "is_padding"):
            self._pack_blocks(
                getattr(speculator.input_buffers, name),
                getattr(self.input_buffers, name),
                num_reqs,
                count,
                self.query_width,
            )
        self.input_buffers.seq_lens[:count].copy_(speculator.input_buffers.seq_lens[rows])
        for name in ("sample_pos", "sample_idx_mapping"):
            self._pack_blocks(getattr(speculator, name), getattr(self, name), num_reqs, count, self.draft_width)
        for gid in self.group_ids:
            self.block_tables.input_block_tables[gid][:count].copy_(
                speculator.block_tables.input_block_tables[gid][rows]
            )
            self._pack_blocks(
                speculator.block_tables.slot_mappings[gid],
                self.block_tables.slot_mappings[gid],
                num_reqs,
                count,
                self.query_width,
            )
        # Fill only the transmitted rows; padding must not leak old proposals.
        padded_count = (num_reqs + self.world_size - 1) // self.world_size
        self.draft_tokens[:padded_count].fill_(-1)
        assert num_tokens <= self.input_buffers.max_num_tokens
        return count

    def _pack_blocks(self, source, destination, num_reqs, count, width):
        # copy_ accepts strided source rows and avoids a temporary index tensor
        # or flatten/reshape allocation for each query field.
        destination[: count * width].view(count, width).copy_(
            source[: num_reqs * width].view(num_reqs, width)[self.rank :: self.world_size]
        )

    @contextmanager
    def activate(self, speculator: Any):
        names = ("input_buffers", "block_tables", "sample_indices", "sample_pos", "sample_idx_mapping", "draft_tokens")
        originals = {name: getattr(speculator, name) for name in names}
        try:
            for name in names:
                setattr(speculator, name, getattr(self, name))
            yield
        finally:
            for name, value in originals.items():
                setattr(speculator, name, value)

    def gather(self, group: Any, output: torch.Tensor, num_reqs: int) -> None:
        if num_reqs == 0:
            return
        padded_count = (num_reqs + self.world_size - 1) // self.world_size
        gathered = group.all_gather(self.draft_tokens[:padded_count], dim=0)
        # Gather is rank-major; round-robin ownership is request-major.
        ordered = gathered.view(self.world_size, padded_count, self.draft_width).transpose(0, 1)
        output[:num_reqs].copy_(ordered.reshape(-1, self.draft_width)[:num_reqs])
