# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


def compact_mtp_topk_indices(
    draft_model: nn.Module,
    token_indices_to_sample: torch.Tensor,
    num_input_tokens: int,
    tp_group: GroupCoordinator | None = None,
) -> None:
    """Move step-0 top-k rows to the query layout of subsequent MTP steps."""
    if tp_group is None:
        draft_model.compact_topk_indices(token_indices_to_sample)
        return
    if token_indices_to_sample.numel() == 0:
        return

    # DSA-CP stores each rank's query rows at the front of its buffer. MTP
    # keeps the padded input size, but steps 1+ contain one query per request.
    local_num_tokens = (num_input_tokens + tp_group.world_size - 1) // tp_group.world_size
    local_start = tp_group.rank_in_group * local_num_tokens
    local_indices = token_indices_to_sample - local_start
    owns_row = (local_indices >= 0) & (local_indices < local_num_tokens)
    gather_indices = local_indices.clamp(0, local_num_tokens - 1).long()
    local_num_reqs = max(0, min(local_num_tokens, token_indices_to_sample.numel() - local_start))

    seen_buffers: set[int] = set()
    for module in draft_model.modules():
        buffer = getattr(module, "topk_indices_buffer", None)
        if buffer is None or id(buffer) in seen_buffers:
            continue
        seen_buffers.add(id(buffer))

        # Exactly one rank owns each sampled row. Summing masked rows moves
        # only request_count * topk entries, rather than all step-0 tokens.
        # Keep indices and masks on device; no scalar D2H extraction is needed.
        rows = buffer.index_select(0, gather_indices)
        rows.masked_fill_(~owns_row.view(-1, *([1] * (rows.ndim - 1))), 0)
        rows = tp_group.all_reduce(rows)
        if local_num_reqs:
            buffer[:local_num_reqs].copy_(rows[local_start : local_start + local_num_reqs])
