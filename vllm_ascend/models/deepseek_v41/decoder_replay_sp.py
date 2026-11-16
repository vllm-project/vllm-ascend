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
        device: torch.device,
    ) -> "SPReplayPlan":
        original_width = (original_tokens + tp_size - 1) // tp_size
        replay_width = replay_tokens // tp_size
        rows = np.concatenate(
            [np.arange(end - kept, end, dtype=np.int64) for end, kept in zip(query_start_loc[1:], kept_lens)]
        )
        owners = rows // original_width
        compact_rows = np.arange(rows.size, dtype=np.int64)
        owned = owners == tp_rank
        local_rows = rows[owned] - tp_rank * original_width
        send_counts = np.bincount(compact_rows[owned] // replay_width, minlength=tp_size).tolist()
        received_owners = owners[tp_rank * replay_width : (tp_rank + 1) * replay_width]
        recv_counts = np.bincount(received_owners, minlength=tp_size).tolist()
        return cls(
            torch.as_tensor(local_rows, device=device),
            send_counts,
            recv_counts,
            original_width,
            replay_width,
            len(received_owners),
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
