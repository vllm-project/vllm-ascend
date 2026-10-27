# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def build_allgather_routing(global_expert_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fixed-shape slot tables from the planner's logical-expert map.

    The planner stores [rank, logical expert] -> local slot. Copies have no
    immutable identity: a redundant slot can hold a different logical expert
    after rebalancing. Only the current slot identifies execution ownership.
    Called on CPU at initialization and by the background planner.
    """
    ep_size, num_experts = global_expert_map.shape
    local_counts = (global_expert_map >= 0).sum(dim=1)
    if not torch.all(local_counts == local_counts[0]):
        raise ValueError("ALLGATHER EPLB requires equal expert capacity on every rank")
    num_local = int(local_counts[0])
    counts = (global_expert_map >= 0).sum(dim=0).to(torch.int64)
    if torch.any(counts == 0):
        raise ValueError("Every logical expert must have at least one physical slot")
    slots = torch.zeros((ep_size, num_experts), dtype=torch.int32)
    for expert in range(num_experts):
        ranks = torch.where(global_expert_map[:, expert] >= 0)[0]
        slots[: len(ranks), expert] = (ranks * num_local + global_expert_map[ranks, expert]).to(torch.int32)
    return slots, counts


def route_allgather(logical_ids: torch.Tensor, slots: torch.Tensor, replica_counts: torch.Tensor) -> torch.Tensor:
    """Identical gathered rows choose identical copies on all EP ranks.

    Consecutive tokens rotate over the copies. No random state, host readback,
    or shape-changing table is needed during graph replay.
    """
    rows = torch.arange(logical_ids.shape[0], device=logical_ids.device).unsqueeze(1)
    copies = (rows + logical_ids) % replica_counts[logical_ids]
    return slots[copies, logical_ids].to(logical_ids.dtype)


def count_local_routes(
    physical_ids: torch.Tensor, valid_rows: torch.Tensor, ep_rank: int, num_local_experts: int
) -> torch.Tensor:
    """Count only valid, owner-local routes, in the actual weight-slot order."""
    local_ids = physical_ids.to(torch.int64) - ep_rank * num_local_experts
    owned = (local_ids >= 0) & (local_ids < num_local_experts) & valid_rows.unsqueeze(1)
    counts = torch.zeros(num_local_experts, dtype=torch.int64, device=physical_ids.device)
    counts.scatter_add_(0, local_ids.clamp(0, num_local_experts - 1).flatten(), owned.to(torch.int64).flatten())
    return counts
