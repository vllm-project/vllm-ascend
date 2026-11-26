# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.distributed.eplb.policy import AbstractEplbPolicy, DefaultEplbPolicy

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy


class AscendDefaultEplbPolicy(DefaultEplbPolicy):
    """Default EPLB policy that preserves the placement of zero-load layers."""

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if old_global_expert_indices is not None:
            old_cpu = old_global_expert_indices.to(device="cpu", dtype=torch.long)
            zero_layers = ~weight.cpu().any(dim=-1)
            if bool(zero_layers.all()):
                return old_cpu.clone()
        else:
            old_cpu = None
            zero_layers = None

        new_indices = super().rebalance_experts(
            weight,
            num_replicas,
            num_groups,
            num_nodes,
            num_ranks,
            old_global_expert_indices,
        )
        if old_cpu is not None and zero_layers is not None and bool(zero_layers.any()):
            new_indices[zero_layers] = old_cpu[zero_layers]
        return new_indices


def create_eplb_policy(policy: str, stair_config: StairConfig) -> AbstractEplbPolicy:
    """Create the configured Ascend EPLB policy."""
    if policy == "default":
        return AscendDefaultEplbPolicy()
    if policy == "stair":
        return StairEplbPolicy(stair_config)
    raise ValueError(f"Unsupported EPLB policy: {policy!r}")
