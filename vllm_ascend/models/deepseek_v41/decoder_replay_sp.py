# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from math import prod

import numpy as np
import torch
import torch.distributed as dist


def _pack_states(hidden_states: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    # Reinterpret storage instead of promoting BF16 hidden states to FP32 or
    # rounding pre-mix down to BF16. Only retained rows reach this function.
    return torch.cat(
        (
            hidden_states.flatten(1).contiguous().view(torch.int8),
            pre_mix.contiguous().view(torch.int8),
        ),
        dim=1,
    )


def _unpack_states(
    packed: torch.Tensor, hidden_states: torch.Tensor, pre_mix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_bytes = prod(hidden_states.shape[1:]) * hidden_states.element_size()
    # Clone also resets the storage offset for single-row receives: the mix
    # byte slice need not start at an FP32-aligned offset in the packed record.
    hidden = packed[:, :hidden_bytes].clone(memory_format=torch.contiguous_format).view(hidden_states.dtype)
    mix = packed[:, hidden_bytes:].clone(memory_format=torch.contiguous_format).view(pre_mix.dtype)
    return hidden.view(packed.shape[0], *hidden_states.shape[1:]), mix.view(packed.shape[0], *pre_mix.shape[1:])


@dataclass(frozen=True)
class SPReplayPlan:
    """Move retained rows between the original and compact contiguous SP shards.

    Query boundaries and kept lengths come from the runner's CPU batch. Both
    layouts preserve request/token order, so each source's retained rows are
    already grouped by destination. All-to-all receives them in source order;
    neither direction needs a device-side sort or a tensor-to-host count copy.
    """

    local_rows: torch.Tensor
    send_counts: list[int]
    recv_counts: list[int]
    original_local_tokens: int
    replay_local_tokens: int
    received_tokens: int

    @classmethod
    def build(
        cls,
        query_start_loc: np.ndarray,
        kept_lens: np.ndarray,
        original_tokens: int,
        replay_tokens: int,
        tp_size: int,
        tp_rank: int,
        rows: torch.Tensor,
    ) -> "SPReplayPlan":
        original_width = (original_tokens + tp_size - 1) // tp_size
        replay_width = replay_tokens // tp_size
        # Retained intervals and original SP shards are ordered. Sweep their
        # boundaries once: owner_offsets[r] counts retained rows before shard r.
        # This is O(num_requests + TP), independent of the retained token count.
        owner_offsets = []
        req = 0
        completed_rows = 0
        for boundary in range(tp_size + 1):
            original_boundary = boundary * original_width
            while req < len(kept_lens) and query_start_loc[req + 1] <= original_boundary:
                completed_rows += int(kept_lens[req])
                req += 1
            partial_rows = 0
            if req < len(kept_lens):
                kept_start = int(query_start_loc[req + 1] - kept_lens[req])
                partial_rows = max(0, original_boundary - kept_start)
            owner_offsets.append(completed_rows + partial_rows)

        begin, end = owner_offsets[tp_rank : tp_rank + 2]
        replay_begin = tp_rank * replay_width
        replay_end = replay_begin + replay_width
        send_counts = [
            max(0, min(end, (rank + 1) * replay_width) - max(begin, rank * replay_width)) for rank in range(tp_size)
        ]
        recv_counts = [
            max(0, min(replay_end, owner_end) - max(replay_begin, owner_begin))
            for owner_begin, owner_end in zip(owner_offsets, owner_offsets[1:])
        ]
        # _gather already wrote these ordered indices on device. Each original
        # rank owns one contiguous slice of the valid prefix; exclude padding.
        local_rows = rows[begin:end] - tp_rank * original_width
        return cls(
            local_rows,
            send_counts,
            recv_counts,
            original_width,
            replay_width,
            sum(recv_counts),
        )

    def compact(
        self, hidden_states: torch.Tensor, pre_mix: torch.Tensor, group: dist.ProcessGroup
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select locally and exchange both states in one byte-packed collective."""
        selected = _pack_states(
            hidden_states.index_select(0, self.local_rows), pre_mix.index_select(0, self.local_rows)
        )
        output = selected.new_zeros((self.replay_local_tokens, selected.shape[1]))
        dist.all_to_all_single(
            output[: self.received_tokens],
            selected,
            output_split_sizes=self.recv_counts,
            input_split_sizes=self.send_counts,
            group=group,
        )
        return _unpack_states(output, hidden_states, pre_mix)

    def restore(
        self, hidden_states: torch.Tensor, pre_mix: torch.Tensor, group: dist.ProcessGroup
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return only valid decoder rows to their original SP owners."""
        packed = _pack_states(hidden_states[: self.received_tokens], pre_mix[: self.received_tokens])
        selected = packed.new_empty((self.local_rows.numel(), packed.shape[1]))
        dist.all_to_all_single(
            selected,
            packed,
            output_split_sizes=self.send_counts,
            input_split_sizes=self.recv_counts,
            group=group,
        )
        hidden, mix = _unpack_states(selected, hidden_states, pre_mix)
        output_hidden = hidden.new_zeros((self.original_local_tokens, *hidden.shape[1:]))
        output_mix = mix.new_zeros((self.original_local_tokens, *mix.shape[1:]))
        output_hidden.index_copy_(0, self.local_rows, hidden)
        output_mix.index_copy_(0, self.local_rows, mix)
        return output_hidden, output_mix
