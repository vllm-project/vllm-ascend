# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


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

    def compact(self, state: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """Select locally before communicating; synthetic padding is never sent."""
        selected = state.index_select(0, self.local_rows)
        output = state.new_zeros((self.replay_local_tokens, *state.shape[1:]))
        dist.all_to_all_single(
            output[: self.received_tokens],
            selected,
            output_split_sizes=self.recv_counts,
            input_split_sizes=self.send_counts,
            group=group,
        )
        return output

    def restore(self, state: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """Return only valid decoder rows to their original SP owners."""
        selected = state.new_empty((self.local_rows.numel(), *state.shape[1:]))
        dist.all_to_all_single(
            selected,
            state[: self.received_tokens].contiguous(),
            output_split_sizes=self.send_counts,
            input_split_sizes=self.recv_counts,
            group=group,
        )
        output = state.new_zeros((self.original_local_tokens, *state.shape[1:]))
        output.index_copy_(0, self.local_rows, selected)
        return output
