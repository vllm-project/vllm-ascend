# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""STAIR hooks for the existing vLLM asynchronous EPLB worker."""

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed import get_eplb_group

from vllm_ascend.distributed.eplb.stair_policy import plan_rebalance


def run_stair_planner(
    model_state: Any,
    state: Any,
    old_mapping: torch.Tensor,
    cuda_stream: torch.cuda.Stream | None,
) -> torch.Tensor:
    """Plan on rank zero and broadcast fixed-shape placement tensors."""
    stats = model_state.eplb_stats
    if stats is None or stats.global_expert_load_window.ndim != 3:
        raise RuntimeError("STAIR requires a temporal logical-load window")
    stream = torch.cuda.stream(cuda_stream) if cuda_stream is not None else nullcontext()
    with stream:
        logical_load = stats.global_expert_load_window.cpu()

    coordinator = get_eplb_group()
    rank = coordinator.device_group.rank()
    num_ranks = coordinator.device_group.size()
    if num_ranks % stats.num_nodes:
        raise RuntimeError("STAIR requires ranks to divide evenly across nodes")
    ranks_per_node = num_ranks // stats.num_nodes
    node_by_rank = tuple(rank // ranks_per_node for rank in range(num_ranks))
    slots_per_rank = old_mapping.shape[1] // num_ranks
    shape = (old_mapping.shape[0], num_ranks, slots_per_rank)

    if rank == 0:
        plan = plan_rebalance(
            logical_load.numpy(),
            old_mapping.numpy().reshape(shape),
            model_state._stair_accepted_scores,
            node_by_rank,
            state._stair_config,
        )
        packed = torch.from_numpy(
            np.stack((plan.placement, plan.source_rank, plan.source_slot))
        )
        scores = torch.from_numpy(plan.accepted_scores)
    else:
        packed = torch.empty((3, *shape), dtype=torch.int64)
        scores = torch.empty(shape[0], dtype=torch.float64)

    if coordinator.cpu_group.size() > 1:
        source = dist.get_global_rank(coordinator.cpu_group, 0)
        dist.broadcast(packed, src=source, group=coordinator.cpu_group)
        dist.broadcast(scores, src=source, group=coordinator.cpu_group)
    model_state.communicator._stair_source_rank = packed[1].numpy()
    model_state.communicator._stair_source_slot = packed[2].numpy()
    model_state._stair_candidate_scores = scores.numpy()
    return packed[0].reshape_as(old_mapping).to(dtype=old_mapping.dtype)
