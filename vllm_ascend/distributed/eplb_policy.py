# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""STAIR placement policy for Model Runner V2 EPLB."""

import time

import torch
from vllm.logger import logger

from vllm_ascend.eplb.core.policy.policy_stair import StairEplbPolicy
from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb


def _expand_logical_load_to_slots(
    logical_load: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
) -> torch.Tensor:
    """Convert logical-expert load to the per-replica input used by Swift."""
    if logical_load.ndim != 2:
        raise ValueError(f"logical_load must be two-dimensional, got shape {tuple(logical_load.shape)}.")
    if physical_to_logical_map.ndim != 2:
        raise ValueError(
            f"physical_to_logical_map must be two-dimensional, got shape {tuple(physical_to_logical_map.shape)}."
        )
    if logical_load.device.type != "cpu" or physical_to_logical_map.device.type != "cpu":
        raise ValueError("STAIR policy inputs must be CPU tensors.")
    if logical_load.shape[0] != physical_to_logical_map.shape[0]:
        raise ValueError(
            "logical_load and physical_to_logical_map must have the same number of layers, "
            f"got {logical_load.shape[0]} and {physical_to_logical_map.shape[0]}."
        )

    placement = physical_to_logical_map.detach().to(dtype=torch.long).clone()
    num_logical_experts = logical_load.shape[1]
    if bool((placement < 0).any()) or bool((placement >= num_logical_experts).any()):
        raise ValueError("physical_to_logical_map contains an invalid logical expert index.")

    replica_counts = torch.zeros(logical_load.shape, dtype=torch.long, device="cpu")
    replica_counts.scatter_add_(1, placement, torch.ones_like(placement, dtype=torch.long))
    if bool((replica_counts == 0).any()):
        raise ValueError("Every logical expert must have at least one physical replica.")

    slot_replica_counts = replica_counts.gather(1, placement)
    return logical_load.detach().to(dtype=torch.float64).gather(1, placement) / slot_replica_counts


def _reject_invalid_placement_layers(
    old_placement: torch.Tensor,
    proposed_placement: torch.Tensor,
    num_ranks: int,
    num_logical_experts: int,
) -> list[int]:
    """Reject plans that cannot be executed by the existing transfer path."""
    slots_per_rank = old_placement.shape[1] // num_ranks
    rejected_layers = []
    for layer_idx in range(old_placement.shape[0]):
        old_layer = old_placement[layer_idx]
        new_layer = proposed_placement[layer_idx]
        if (
            bool((new_layer < 0).any())
            or bool((new_layer >= num_logical_experts).any())
            or torch.unique(new_layer).numel() != num_logical_experts
        ):
            rejected_layers.append(layer_idx)
            continue

        old_ranks = old_layer.reshape(num_ranks, slots_per_rank)
        new_ranks = new_layer.reshape(num_ranks, slots_per_rank)
        for rank_idx in range(num_ranks):
            old_rank = old_ranks[rank_idx]
            new_rank = new_ranks[rank_idx]
            if torch.unique(new_rank).numel() != new_rank.numel():
                rejected_layers.append(layer_idx)
                break
            experts_kept_on_rank = torch.isin(new_rank, old_rank)
            if not torch.equal(new_rank[experts_kept_on_rank], old_rank[experts_kept_on_rank]):
                rejected_layers.append(layer_idx)
                break

    if rejected_layers:
        proposed_placement[rejected_layers] = old_placement[rejected_layers]
    return rejected_layers


def _calculate_swift_placement(
    logical_load: torch.Tensor,
    old_placement: torch.Tensor,
    num_nodes: int,
    num_ranks: int,
) -> torch.Tensor:
    """Generate a complete placement using the proven legacy Swift planner."""
    slots_per_rank = old_placement.shape[1] // num_ranks
    slot_load = _expand_logical_load_to_slots(logical_load, old_placement)
    legacy_placement = old_placement.reshape(logical_load.shape[0], num_ranks, slots_per_rank)
    legacy_slot_load = slot_load.reshape(logical_load.shape[0], num_ranks, slots_per_rank)

    policy = SwiftBalanceEplb()
    policy.num_die_per_host = num_ranks // num_nodes
    _, _, new_placement = policy.rebalance_experts(
        legacy_placement,
        legacy_slot_load,
        is_node_redundant=False,
    )
    return (
        torch.as_tensor(new_placement, dtype=torch.long, device="cpu")
        .reshape(logical_load.shape[0], old_placement.shape[1])
        .contiguous()
    )


class StairEplbPolicyAdapter:
    """State-owned STAIR adapter for the upstream policy call contract."""

    uses_expert_load_time_series = True

    def __init__(self) -> None:
        self.policy = StairEplbPolicy()

    def rebalance_experts(
        self,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if old_global_expert_indices is None:
            raise ValueError("STAIR EPLB requires the current physical-to-logical map.")
        if num_replicas <= 0 or num_groups <= 0 or num_nodes <= 0 or num_ranks <= 0:
            raise ValueError("num_replicas, num_groups, num_nodes, and num_ranks must be positive.")
        if num_replicas % num_ranks != 0:
            raise ValueError(f"num_replicas ({num_replicas}) must be divisible by num_ranks ({num_ranks}).")
        if num_ranks % num_nodes != 0:
            raise ValueError(f"num_ranks ({num_ranks}) must be divisible by num_nodes ({num_nodes}).")
        if weight.ndim != 3:
            raise ValueError(f"STAIR EPLB requires [window, layers, experts] load, got {tuple(weight.shape)}.")
        if weight.device.type != "cpu" or old_global_expert_indices.device.type != "cpu":
            raise ValueError("STAIR EPLB policy inputs must be CPU tensors.")
        if old_global_expert_indices.shape != (weight.shape[1], num_replicas):
            raise ValueError(
                "Current placement shape must be [layers, num_replicas], got "
                f"{tuple(old_global_expert_indices.shape)} for weight shape {tuple(weight.shape)} "
                f"and num_replicas={num_replicas}."
            )

        old_placement = old_global_expert_indices.detach().to(dtype=torch.long).clone()
        if not bool(weight.any()):
            return old_placement.contiguous()

        start_time = time.perf_counter()
        swift_candidate = _calculate_swift_placement(weight.sum(dim=0), old_placement, num_nodes, num_ranks)
        candidate_layer_count = int(torch.any(swift_candidate != old_placement, dim=1).sum().item())
        candidate_rejected_layers = _reject_invalid_placement_layers(
            old_placement,
            swift_candidate,
            num_ranks,
            weight.shape[2],
        )
        weight_array = weight.detach().contiguous().numpy()
        new_placement = self.policy.rebalance_experts(
            weight_array,
            old_placement.numpy(),
            swift_candidate.numpy(),
            num_ranks,
        )
        result = torch.from_numpy(new_placement).to(dtype=torch.long).contiguous()
        rejected_layers = _reject_invalid_placement_layers(
            old_placement,
            result,
            num_ranks,
            weight.shape[2],
        )
        changed_layer_count = int(torch.any(result != old_placement, dim=1).sum().item())
        logger.info(
            "STAIR EPLB policy completed in %.3f ms; Swift proposed %d layers; "
            "rejected %d invalid Swift layers and %d invalid final layers; accepted %d layers.",
            (time.perf_counter() - start_time) * 1000,
            candidate_layer_count,
            len(candidate_rejected_layers),
            len(rejected_layers),
            changed_layer_count,
        )
        return result

    def commit_layer(
        self,
        expert_load: torch.Tensor,
        layer_idx: int,
        committed_placement: torch.Tensor,
        num_ranks: int,
    ) -> None:
        """Update hysteresis only after the layer map and weights are committed."""
        self.policy.commit_layer(
            expert_load.detach().contiguous().numpy(),
            layer_idx,
            committed_placement.detach().contiguous().numpy(),
            num_ranks,
        )
