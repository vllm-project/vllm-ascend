# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Adapters for using Ascend EPLB policies with the vLLM V2 state."""

from typing import Any

import torch

ASCEND_V2_POLICY_TYPES = {
    "policy_swift_balancer": 2,
    "policy_flashlb": 3,
}


class AscendV2EplbPolicy:
    """Adapt an Ascend EPLB policy to the upstream vLLM policy contract."""

    def __init__(self, policy_name: str) -> None:
        from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory

        try:
            policy_type = ASCEND_V2_POLICY_TYPES[policy_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported Ascend V2 EPLB policy: {policy_name}") from exc
        self.policy_name = policy_name
        self._policy: Any = PolicyFactory.generate_policy(policy_type)

    @staticmethod
    def _build_physical_load(
        logical_load: torch.Tensor,
        physical_to_logical_map: torch.Tensor,
    ) -> torch.Tensor:
        """Split logical load equally over its current physical replicas."""
        if logical_load.ndim != 2 or physical_to_logical_map.ndim != 2:
            raise ValueError("EPLB logical load and mapping must both be 2-D")
        if logical_load.shape[0] != physical_to_logical_map.shape[0]:
            raise ValueError("EPLB logical load and mapping layer counts must match")
        if bool((physical_to_logical_map < 0).any()):
            raise ValueError("Ascend V2 EPLB policies do not support empty expert slots")
        if bool((physical_to_logical_map >= logical_load.shape[1]).any()):
            raise ValueError("EPLB mapping contains an invalid logical expert ID")

        mapping = physical_to_logical_map.long()
        load = logical_load.float()
        replica_count = torch.zeros_like(load)
        replica_count.scatter_add_(
            dim=1,
            index=mapping,
            src=torch.ones_like(mapping, dtype=load.dtype),
        )
        slot_replica_count = replica_count.gather(1, mapping)
        return load.gather(1, mapping) / slot_replica_count

    def rebalance_experts(
        self,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a physical-to-logical mapping in the upstream V2 format."""
        del num_groups, num_nodes
        if old_global_expert_indices is None:
            raise ValueError("Ascend V2 EPLB policies require the current expert mapping")
        if num_ranks <= 0 or num_replicas % num_ranks != 0:
            raise ValueError("Physical experts must be divisible by EP ranks")

        old_mapping = old_global_expert_indices.detach().cpu()
        if old_mapping.shape[1] != num_replicas:
            raise ValueError("Ascend V2 EPLB policies do not support changing the number of physical expert slots")

        physical_load = self._build_physical_load(
            weight.detach().cpu(),
            old_mapping,
        )
        num_layers = old_mapping.shape[0]
        experts_per_rank = num_replicas // num_ranks
        current_table = old_mapping.reshape(
            num_layers,
            num_ranks,
            experts_per_rank,
        )
        workload_table = physical_load.reshape_as(current_table)

        changed, _, new_deployment = self._policy.rebalance_experts(
            current_table,
            workload_table,
        )
        if not changed:
            return old_mapping.clone()

        new_mapping = torch.as_tensor(
            new_deployment,
            dtype=old_mapping.dtype,
            device="cpu",
        ).reshape(num_layers, num_replicas)
        if bool((new_mapping < 0).any()) or bool((new_mapping >= weight.shape[1]).any()):
            raise ValueError("Ascend V2 EPLB policy returned an invalid expert mapping")
        return new_mapping.contiguous()
