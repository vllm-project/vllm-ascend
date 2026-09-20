# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Adapter for using Ascend SwiftBalancer with the vLLM V2 state."""

from typing import Any

import torch
from vllm.logger import logger

SWIFT_BALANCER_POLICY_TYPE = 2
DEFAULT_MAX_REBALANCED_LAYERS_PER_CYCLE = 10


class AscendV2EplbPolicy:
    """Adapt Ascend SwiftBalancer to the upstream vLLM policy contract."""

    ep_rank: int = 0

    def __init__(
        self,
        max_rebalanced_layers_per_cycle: int = DEFAULT_MAX_REBALANCED_LAYERS_PER_CYCLE,
        ep_rank: int = 0,
    ) -> None:
        from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory

        if max_rebalanced_layers_per_cycle <= 0:
            raise ValueError("max_rebalanced_layers_per_cycle must be greater than 0")
        self.max_rebalanced_layers_per_cycle = max_rebalanced_layers_per_cycle
        self.ep_rank = ep_rank
        self._policy: Any = PolicyFactory.generate_policy(SWIFT_BALANCER_POLICY_TYPE)
        self._bootstrap_rebalance_active = True

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

    @classmethod
    def _compute_hotness_imbalance(
        cls,
        logical_load: torch.Tensor,
        physical_to_logical_map: torch.Tensor,
        num_ranks: int,
    ) -> tuple[float, float, list[float]]:
        """Compute the V1-compatible per-layer rank-load imbalance."""
        if num_ranks <= 0 or physical_to_logical_map.shape[1] % num_ranks != 0:
            raise ValueError("Physical experts must be divisible by EP ranks")

        physical_load = cls._build_physical_load(
            logical_load,
            physical_to_logical_map,
        )
        rank_load = physical_load.reshape(
            physical_load.shape[0],
            num_ranks,
            -1,
        ).sum(dim=-1)
        mean_load = rank_load.mean(dim=1)
        max_load = rank_load.max(dim=1).values
        per_layer_imbalance = torch.where(
            mean_load > 0,
            max_load / mean_load,
            torch.ones_like(mean_load),
        )
        imbalance_list = per_layer_imbalance.tolist()
        return (
            sum(imbalance_list) / len(imbalance_list),
            max(imbalance_list),
            imbalance_list,
        )

    @staticmethod
    def _validate_replica_rank_placement(
        mapping: torch.Tensor,
        num_ranks: int,
    ) -> None:
        """Ensure copies of one logical expert reside on distinct EP ranks."""
        if (
            mapping.ndim != 2
            or num_ranks <= 0
            or mapping.shape[1] % num_ranks != 0
        ):
            raise ValueError(
                "Physical experts must be a 2-D tensor divisible by EP ranks"
            )

        rank_mapping = mapping.reshape(mapping.shape[0], num_ranks, -1)
        for layer_idx, layer_mapping in enumerate(rank_mapping):
            for rank_idx, local_mapping in enumerate(layer_mapping):
                if torch.unique(local_mapping).numel() != local_mapping.numel():
                    raise ValueError(
                        "SwiftBalancer placed replicas of the same logical "
                        f"expert on layer {layer_idx}, EP rank {rank_idx}"
                    )

    def _log_replica_changes(
        self,
        logical_load: torch.Tensor,
        old_mapping: torch.Tensor,
        new_mapping: torch.Tensor,
        num_ranks: int,
    ) -> None:
        """Log which logical experts gained or lost replicas and their ranks."""
        if self.ep_rank != 0:
            return

        num_logical_experts = logical_load.shape[1]
        experts_per_rank = old_mapping.shape[1] // num_ranks
        changed_layers = torch.nonzero(
            torch.any(new_mapping != old_mapping, dim=1),
            as_tuple=False,
        ).flatten()
        for layer_idx in changed_layers.tolist():
            old_counts = torch.bincount(
                old_mapping[layer_idx].long(),
                minlength=num_logical_experts,
            )
            new_counts = torch.bincount(
                new_mapping[layer_idx].long(),
                minlength=num_logical_experts,
            )
            changed_experts = torch.nonzero(
                old_counts != new_counts,
                as_tuple=False,
            ).flatten().tolist()
            changed_experts.sort(
                key=lambda expert_id: (
                    -float(logical_load[layer_idx, expert_id]),
                    expert_id,
                )
            )
            changes = []
            for expert_id in changed_experts:
                physical_ids = torch.nonzero(
                    new_mapping[layer_idx] == expert_id,
                    as_tuple=False,
                ).flatten()
                replica_ranks = torch.div(
                    physical_ids,
                    experts_per_rank,
                    rounding_mode="floor",
                ).tolist()
                changes.append(
                    "%d(load=%.0f,copies=%d->%d,ranks=%s)"
                    % (
                        expert_id,
                        float(logical_load[layer_idx, expert_id]),
                        int(old_counts[expert_id]),
                        int(new_counts[expert_id]),
                        replica_ranks,
                    )
                )
            logger.info(
                "[eplb/policy2] layer %d replica changes: %s",
                layer_idx,
                changes,
            )

    def _run_swift_balancer(
        self,
        current_table: torch.Tensor,
        workload_table: torch.Tensor,
        logical_load: torch.Tensor,
    ) -> tuple[int, Any, Any, bool]:
        """Run initial heat-driven cycles without steady-state thresholds."""
        bootstrap = bool(
            getattr(self, "_bootstrap_rebalance_active", True)
            and torch.any(logical_load > 0)
        )
        if not bootstrap:
            change, priority, deployment = self._policy.rebalance_experts(
                current_table,
                workload_table,
            )
            return change, priority, deployment, False

        old_imbalance_threshold = self._policy.imbalance_threshold
        old_increment = self._policy.increment
        self._policy.imbalance_threshold = 1.0
        self._policy.increment = 0.0
        if self.ep_rank == 0:
            logger.info(
                "[eplb/policy2] bootstrap rebalance uses collected heat "
                "without steady-state improvement thresholds"
            )
        try:
            change, priority, deployment = self._policy.rebalance_experts(
                current_table,
                workload_table,
            )
        finally:
            self._policy.imbalance_threshold = old_imbalance_threshold
            self._policy.increment = old_increment
        return change, priority, deployment, True

    def _log_hotness_imbalance(
        self,
        logical_load: torch.Tensor,
        old_mapping: torch.Tensor,
        new_mapping: torch.Tensor,
        num_ranks: int,
    ) -> None:
        """Log imbalance before and after the mapping selected for this cycle."""
        if self.ep_rank != 0:
            return

        current_mean, current_max, current_list = self._compute_hotness_imbalance(
            logical_load,
            old_mapping,
            num_ranks,
        )
        update_mean, update_max, update_list = self._compute_hotness_imbalance(
            logical_load,
            new_mapping,
            num_ranks,
        )
        self.latest_expert_hotness = {
            "current_mean": current_mean,
            "current_max": current_max,
            "update_mean": update_mean,
            "update_max": update_max,
            "current_imbalance_list": current_list,
            "update_imbalance_list": update_list,
        }
        logger.info(
            "[eplb/worker] Expert hotness imbalance, current: "
            "mean=%.3f max=%.3f, updated: mean=%.3f max=%.3f",
            current_mean,
            current_max,
            update_mean,
            update_max,
        )

    def _limit_rebalanced_layers(
        self,
        old_mapping: torch.Tensor,
        new_mapping: torch.Tensor,
        per_layer_priority: Any,
    ) -> torch.Tensor:
        """Keep only the highest-priority changed layers for this cycle."""
        changed_mask = torch.any(new_mapping != old_mapping, dim=1)
        changed_layers = torch.nonzero(changed_mask, as_tuple=False).flatten()
        num_changed_layers = changed_layers.numel()
        if num_changed_layers <= self.max_rebalanced_layers_per_cycle:
            return new_mapping

        if per_layer_priority is None:
            ordered_layers = changed_layers
        else:
            priority = torch.as_tensor(per_layer_priority, dtype=torch.long).flatten()
            if (
                priority.numel() != old_mapping.shape[0]
                or bool((priority < 0).any())
                or bool((priority >= old_mapping.shape[0]).any())
            ):
                raise ValueError("Ascend V2 EPLB policy returned invalid per-layer priorities")
            ordered_layers = priority[changed_mask[priority]]

        selected_layers = ordered_layers[: self.max_rebalanced_layers_per_cycle]
        limited_mapping = old_mapping.clone()
        limited_mapping[selected_layers] = new_mapping[selected_layers]
        logger.info(
            "Ascend V2 EPLB limits this cycle from %d changed layers "
            "to %d high-priority layers: %s",
            num_changed_layers,
            selected_layers.numel(),
            selected_layers.tolist(),
        )
        return limited_mapping

    def _log_policy_diagnostics(self, change: int) -> None:
        """Log the SwiftBalancer decision process on one EP rank."""
        if self.ep_rank != 0:
            return

        diagnostics = getattr(self._policy, "last_rebalance_diagnostics", None)
        summary = getattr(self._policy, "last_rebalance_summary", None)
        if not isinstance(diagnostics, list) or not isinstance(summary, dict):
            return

        reason_layers: dict[str, list[int]] = {}
        layer_details = []
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            layer = int(diagnostic["layer"])
            reason = str(diagnostic["reason"])
            reason_layers.setdefault(reason, []).append(layer)
            after_swap = diagnostic["after_swap_imbalance"]
            after_swap_text = "n/a" if after_swap is None else f"{float(after_swap):.6f}"
            layer_details.append(
                "%d:%.6f->%s/candidate=%s/improved=%s/%s"
                % (
                    layer,
                    float(diagnostic["initial_imbalance"]),
                    after_swap_text,
                    bool(diagnostic["candidate_changed"]),
                    bool(diagnostic["strictly_improved"]),
                    reason,
                )
            )

        logger.info(
            "[eplb/policy2] rebalance decision: global_change=%d, "
            "max_heat=%.3f->%.3f, improvement=%.3f%%, "
            "imbalance_threshold=%.3f, swap_threshold=%.3f, "
            "num_max_com=%d, raw_changed_layers=%s, reason_layers=%s",
            change,
            float(summary.get("npu_heat_all_origin", 0.0)),
            float(summary.get("npu_heat_all_after", 0.0)),
            float(summary.get("improvement_ratio", 0.0)) * 100,
            float(summary.get("imbalance_threshold", 0.0)),
            float(summary.get("swap_threshold", 0.0)),
            int(summary.get("num_max_com", 0)),
            summary.get("raw_changed_layers", []),
            reason_layers,
        )
        logger.info(
            "[eplb/policy2] per-layer decisions "
            "(layer:initial->after/reason): %s",
            layer_details,
        )

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
            raise ValueError(
                "Ascend V2 EPLB policies do not support changing the number "
                "of physical expert slots"
            )

        logical_load = weight.detach().cpu()
        physical_load = self._build_physical_load(
            logical_load,
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

        change, per_layer_priority, new_deployment, bootstrap = (
            self._run_swift_balancer(
                current_table,
                workload_table,
                logical_load,
            )
        )
        self._log_policy_diagnostics(change)

        new_mapping = torch.as_tensor(
            new_deployment,
            dtype=old_mapping.dtype,
            device="cpu",
        ).reshape(num_layers, num_replicas)
        if bool((new_mapping < 0).any()) or bool((new_mapping >= weight.shape[1]).any()):
            raise ValueError("Ascend V2 EPLB policy returned an invalid expert mapping")
        self._validate_replica_rank_placement(new_mapping, num_ranks)
        raw_changed_layers = int(
            torch.any(new_mapping != old_mapping, dim=1).sum()
        )
        new_mapping = self._limit_rebalanced_layers(
            old_mapping,
            new_mapping,
            per_layer_priority,
        )
        if bootstrap:
            self._bootstrap_rebalance_active = (
                raw_changed_layers > self.max_rebalanced_layers_per_cycle
            )
            if self.ep_rank == 0:
                logger.info(
                    "[eplb/policy2] bootstrap rebalance %s; "
                    "raw changed layers=%d",
                    "continues next cycle"
                    if self._bootstrap_rebalance_active
                    else "completed",
                    raw_changed_layers,
                )
        self._log_replica_changes(
            logical_load,
            old_mapping,
            new_mapping,
            num_ranks,
        )
        self._log_hotness_imbalance(
            logical_load,
            old_mapping,
            new_mapping,
            num_ranks,
        )
        return new_mapping.contiguous()
