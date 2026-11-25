# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass
class MLADraftWindow:
    """Prepare the same right-aligned KV window for every query in a draft block."""

    window_size: int
    block_size: int
    query_width: int
    replay_block_table: torch.Tensor | None = None
    replay_mask: torch.Tensor | None = None

    def prepare(
        self,
        block_table: torch.Tensor,
        seq_lens: list[int],
        num_reqs: int,
        *,
        capture: bool = False,
        replay: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        # Keep the first (potentially partial) page and mask its excluded keys.
        # One extra page may be needed when the window begins within a page.
        num_pages = (self.window_size + 2 * self.block_size - 2) // self.block_size
        if len(seq_lens) > num_reqs:
            raise ValueError("Draft MLA has more KV lengths than query rows")
        seq_lens = seq_lens + [0] * (num_reqs - len(seq_lens))
        if capture:
            # Capture uses dummy requests. Set up the FIA input shapes without
            # recording host copies of request metadata into the ACL graph.
            first_pages = torch.zeros(num_reqs, dtype=torch.long, device=block_table.device)
            prefix = torch.zeros_like(first_pages)
            visible_lens = [min(length, self.window_size) for length in seq_lens]
        else:
            starts = [max(0, length - self.window_size) for length in seq_lens]
            start_pages = [start // self.block_size for start in starts]
            prefix_lengths = [start % self.block_size for start in starts]
            visible_lens = [length - page * self.block_size for length, page in zip(seq_lens, start_pages)]
            first_pages = torch.tensor(start_pages, dtype=torch.long, device=block_table.device)
            prefix = torch.tensor(prefix_lengths, dtype=torch.long, device=block_table.device)

        page_offsets = torch.arange(num_pages, device=block_table.device)
        columns = (first_pages[:, None] + page_offsets).clamp_max(block_table.shape[1] - 1)
        if block_table.shape[0] < num_reqs:
            block_table = torch.nn.functional.pad(block_table, (0, 0, 0, num_reqs - block_table.shape[0]))
        window_table = block_table[:num_reqs].gather(1, columns)

        key_offsets = torch.arange(num_pages * self.block_size, device=block_table.device)
        mask = (key_offsets < prefix[:, None])[:, None, :].expand(-1, self.query_width, -1)
        if not replay:
            return window_table, mask.contiguous(), visible_lens

        # The graph task keeps these addresses. Allocate on the first update
        # so no writes recorded during capture can overwrite replay inputs.
        if self.replay_block_table is None or self.replay_mask is None:
            self.replay_block_table = torch.empty_like(window_table)
            self.replay_mask = torch.empty_like(mask, memory_format=torch.contiguous_format)
        self.replay_block_table.copy_(window_table)
        self.replay_mask.copy_(mask)
        return self.replay_block_table, self.replay_mask, visible_lens
