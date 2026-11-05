# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group
from vllm.forward_context import ForwardContext

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType


def collect_local_routing_rows(
    topk_ids: torch.Tensor, ctx: ForwardContext, *, dp_rank: int, tp_size: int
) -> torch.Tensor:
    """Undo the active Ascend prepare's token layout for capture."""
    counts = ctx.dp_metadata.num_tokens_across_dp_cpu if ctx.dp_metadata is not None else None
    if counts is not None:
        local_rows = int(counts[dp_rank].item())
    else:
        local_rows = _EXTRA_CTX.num_tokens
        if local_rows is None:
            local_rows = ctx.batch_descriptor.num_tokens if ctx.batch_descriptor is not None else topk_ids.shape[0]

    comm_type = _EXTRA_CTX.moe_comm_type
    if comm_type in (MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2):
        prepare = _EXTRA_CTX.moe_comm_method.prepare_finalize
        if prepare.replace_allreduce:
            # SP supplies equal shards; MC2 may add padding to EACH shard.
            total_rows = topk_ids.shape[0] * tp_size
        elif comm_type == MoECommType.ALLTOALL:
            total_rows = max(prepare.num_tokens, tp_size)
        else:
            total_rows = max(prepare.num_tokens, _EXTRA_CTX.padded_num_tokens)

        if tp_size > 1 and total_rows > 0:
            gathered = topk_ids.new_empty((total_rows, topk_ids.shape[1]))
            dist.all_gather(
                list(torch.tensor_split(gathered, tp_size, dim=0)),
                topk_ids.contiguous(),
                get_tp_group().device_group,
            )
            topk_ids = gathered
        if comm_type != MoECommType.ALLTOALL and prepare.replace_allreduce and total_rows > 0:
            # Use the pre-MC2 shard length, which can include graph padding.
            topk_ids = topk_ids.view(tp_size, -1, topk_ids.shape[1])[:, : prepare.num_tokens].reshape(
                -1, topk_ids.shape[1]
            )
        return topk_ids[:local_rows]

    if counts is None:
        return topk_ids[:local_rows]

    n = topk_ids.shape[0]
    if n == int(counts.sum().item()):
        start = int(counts[:dp_rank].sum().item())
    elif n == local_rows:
        start = 0
    elif ctx.dp_metadata.local_sizes is not None and n == sum(ctx.dp_metadata.local_sizes):
        # ALLGATHER + SP removes EP transport padding but retains SP padding.
        sizes = ctx.dp_metadata.local_sizes
        shards_per_dp = len(sizes) // len(counts)
        start = sum(sizes[: dp_rank * shards_per_dp])
    elif n == int(counts.max().item()) * len(counts):
        start = dp_rank * int(counts.max().item())
    else:
        raise AssertionError(f"Unexpected Ascend routing rows: {n}, local_rows={local_rows}, dp_rank={dp_rank}")
    return topk_ids[start : start + local_rows]
