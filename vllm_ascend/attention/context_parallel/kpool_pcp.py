# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Align KPool queries across PCP before exchanging DCP candidates/results."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KPoolPCPQueryLayout:
    restore_indices: torch.Tensor
    local_indices: torch.Tensor
    local_num_tokens: int
    num_actual_local_tokens: int


def build_kpool_pcp_query_layout(common, context, group) -> KPoolPCPQueryLayout:
    indices = context.padded_gather_idx
    if indices is None or indices.numel() % group.world_size:
        raise ValueError("KPool PCP+DCP requires equally padded PCP query rows.")
    local_num_tokens = indices.numel() // group.world_size
    if not 0 <= common.num_actual_tokens <= common.num_input_tokens <= local_num_tokens:
        raise ValueError("KPool PCP+DCP query counts exceed the padded PCP layout.")
    return KPoolPCPQueryLayout(
        restore_indices=context.hidden_restore_idx[: context.global_batch.num_tokens],
        local_indices=indices.view(group.world_size, local_num_tokens)[group.rank_in_group],
        local_num_tokens=local_num_tokens,
        num_actual_local_tokens=common.num_actual_tokens,
    )


def gather_kpool_pcp_queries(tensor, layout: KPoolPCPQueryLayout, group):
    """Remove PCP padding and duplicate decodes in canonical request order."""
    if tensor.shape[0] > layout.local_num_tokens:
        raise ValueError("KPool query rows exceed the PCP collective width.")
    if tensor.shape[0] < layout.local_num_tokens:
        padded = tensor.new_zeros((layout.local_num_tokens, *tensor.shape[1:]))
        padded[: tensor.shape[0]].copy_(tensor)
        tensor = padded
    gathered = group.all_gather(tensor.contiguous(), dim=0)
    return gathered.index_select(0, layout.restore_indices)


def restore_kpool_pcp_queries(tensor, layout: KPoolPCPQueryLayout, num_tokens: int, *, fill_value=0):
    """Return the original local query rows, masking padding and empty ranks."""
    if not 0 <= num_tokens <= layout.local_num_tokens:
        raise ValueError("KPool output rows exceed the PCP collective width.")
    output = tensor.index_select(0, layout.local_indices[:num_tokens])
    valid = torch.arange(num_tokens, device=tensor.device) < layout.num_actual_local_tokens
    return output.masked_fill(~valid.reshape(num_tokens, *([1] * (tensor.ndim - 1))), fill_value)
