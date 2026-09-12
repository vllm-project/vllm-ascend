# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared DSA token layouts for replicated-cache context parallelism."""

import torch
import torch.distributed as dist


def restore_tp_heads(output, tp_group):
    """Exchange [local tokens, all heads] for [all tokens, local heads]."""
    if tp_group.world_size == 1:
        return output
    tokens, heads, width = output.shape
    local_heads = heads // tp_group.world_size
    send = (
        output.view(tokens, tp_group.world_size, local_heads, width)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(-1, local_heads, width)
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=tp_group.device_group)
    return recv
