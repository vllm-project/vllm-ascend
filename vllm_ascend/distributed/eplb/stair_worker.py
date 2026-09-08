# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""STAIR hooks for the existing vLLM asynchronous EPLB worker."""

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed import get_eplb_group
from vllm.distributed.eplb.rebalance_execute import TransferMetadata

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
    node_by_rank = state._stair_node_by_rank
    if len(node_by_rank) != num_ranks:
        raise RuntimeError("STAIR topology does not match the EPLB group")
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
        packed = torch.from_numpy(np.stack((plan.placement, plan.source_rank, plan.source_slot)))
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


def transfer_stair_layer(
    old_layer_indices: torch.Tensor,
    new_layer_indices: torch.Tensor,
    expert_weights: list[torch.Tensor],
    expert_weights_buffer: list[torch.Tensor],
    ep_group: Any,
    communicator: Any,
    is_profile: bool = False,
    cuda_stream: torch.cuda.Stream | None = None,
    rank_mapping: dict[int, int] | None = None,
    layer_idx: int = 0,
) -> TransferMetadata:
    """Execute the exact source rank and slot selected by STAIR."""
    if is_profile or rank_mapping is not None:
        raise ValueError("STAIR explicit transfer does not support profile or elastic mode")
    old = old_layer_indices.numpy()
    new = new_layer_indices.numpy()
    num_ranks = ep_group.size()
    ep_rank = ep_group.rank()
    local_experts = old.size // num_ranks
    sources = communicator._stair_source_rank[layer_idx]
    source_slots = communicator._stair_source_slot[layer_idx]
    if sources.shape != (num_ranks, local_experts) or source_slots.shape != sources.shape:
        raise RuntimeError("STAIR source plan shape changed")

    is_unchanged = np.zeros(local_experts, dtype=np.bool_)
    is_received_locally = np.zeros(local_experts, dtype=np.bool_)
    recv_primary_mask = np.zeros(local_experts, dtype=np.bool_)
    recv_expert_ids = np.full(local_experts, -1, dtype=np.int64)
    recv_dst_rows = np.full(local_experts, -1, dtype=np.int32)
    recv_count = 0
    if hasattr(communicator, "set_transfer_context"):
        communicator.set_transfer_context(old, layer_idx)

    stream = torch.cuda.stream(cuda_stream) if cuda_stream is not None else nullcontext()
    with stream:
        for dst in range(num_ranks):
            for dst_slot in range(local_experts):
                expert = int(new[dst * local_experts + dst_slot])
                src = int(sources[dst, dst_slot])
                src_slot = int(source_slots[dst, dst_slot])
                if old[src * local_experts + src_slot] != expert:
                    raise RuntimeError("STAIR source does not own the requested expert")
                if src == dst:
                    if ep_rank == dst:
                        is_received_locally[dst_slot] = True
                        is_unchanged[dst_slot] = src_slot == dst_slot
                        if src_slot != dst_slot:
                            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                                buffer[dst_slot].copy_(weight[src_slot], non_blocking=True)
                    continue
                if ep_rank == src:
                    communicator.add_send([weight[src_slot] for weight in expert_weights], dst, expert)
                if ep_rank == dst:
                    communicator.add_recv([buffer[dst_slot] for buffer in expert_weights_buffer], src, expert)
                    recv_primary_mask[dst_slot] = True
                    recv_expert_ids[recv_count] = expert
                    recv_dst_rows[recv_count] = dst_slot
                    recv_count += 1
    communicator.execute()
    return TransferMetadata(
        is_unchanged=is_unchanged,
        is_received_locally=is_received_locally,
        recv_primary_mask=recv_primary_mask,
        recv_count=recv_count,
        recv_expert_ids=recv_expert_ids,
        recv_dst_rows=recv_dst_rows,
    )
