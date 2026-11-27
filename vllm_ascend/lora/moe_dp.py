# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Coordinate expert LoRA residency before a cross-DP MoE forward.

Device slots are worker-local cache entries, not adapter identities. Expert
communication carries global adapter IDs and translates them at the receiver.
The CPU control collective also makes each receiver load its own expert shard;
exchanging IDs alone cannot make an absent adapter's weights available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


def lora_slots_to_ids(indices: torch.Tensor, slot_ids: torch.Tensor | None) -> torch.Tensor:
    if slot_ids is None:
        return indices
    return torch.where(indices >= 0, slot_ids[indices.clamp(min=0)], -1)


def lora_ids_to_slots(adapter_ids: torch.Tensor, slot_ids: torch.Tensor | None) -> torch.Tensor:
    if slot_ids is None:
        return adapter_ids
    # A fixed max_loras dimension keeps this usable during graph replay.
    matches = (adapter_ids.unsqueeze(-1) == slot_ids) & (adapter_ids.unsqueeze(-1) >= 0)
    return torch.where(matches.any(dim=-1), matches.to(torch.long).argmax(dim=-1), -1)


class MoELoRACoordinator:
    """One coordinator per runner, used only by MoE LoRA with DP > 1.

    All ranks in the MoE group, including idle DP ranks, must call synchronize
    once per forward. It runs before graph dispatch, never inside a graph or a
    per-layer hot path. The existing worker manager owns loading and eviction.
    """

    def __init__(
        self,
        worker_manager: LRUCacheWorkerLoRAManager,
        wrappers: list[PunicaWrapperNPU],
        cpu_group: dist.ProcessGroup,
    ) -> None:
        self.worker_manager = worker_manager
        # Upstream does not expose slot identities through the worker manager.
        # Keep access to its model manager here until it provides that API.
        self.model_manager = worker_manager._adapter_manager
        self.cpu_group = cpu_group
        self.wrappers = wrappers
        self.step_active = False
        self.num_active_loras = 0
        self._slot_ids = (-1,) * self.model_manager.lora_slots
        self.slot_ids = torch.full((self.model_manager.lora_slots,), -1, dtype=torch.long, device=worker_manager.device)
        for wrapper in wrappers:
            wrapper.moe_lora_slot_ids = self.slot_ids
            wrapper.moe_lora_active = False

    def _gather(self, value):
        values = [None] * dist.get_world_size(self.cpu_group)
        dist.all_gather_object(values, value, group=self.cpu_group)
        return values

    def synchronize(self, requests: set[LoRARequest], *, dummy_run: bool = False) -> None:
        self.step_active = False
        by_id = {}
        # Profiling uses already-resident synthetic adapters rather than real
        # checkpoint requests. Normal forwards must not reuse the previous
        # batch's activity, since local mappings are updated after this hook.
        # An idle DP rank also uses a dummy forward. Its explicit local
        # mapping is empty; the previous global count may describe a peer's
        # real adapter and must not keep later base-only steps active.
        dummy_count = max(len(wrapper.active_lora_slots) for wrapper in self.wrappers) if dummy_run else 0
        gathered = self._gather((list(requests), dummy_count))
        for rank_requests, _ in gathered:
            for request in rank_requests:
                previous = by_id.get(request.lora_int_id)
                if previous is not None and any(
                    getattr(previous, field) != getattr(request, field) for field in request.__struct_fields__
                ):
                    raise ValueError(f"Conflicting LoRA requests across MoE ranks for adapter {request.lora_int_id}.")
                by_id[request.lora_int_id] = request
        if len(by_id) > self.model_manager.lora_slots:
            raise ValueError(
                f"Cross-DP MoE needs {len(by_id)} simultaneous LoRA adapters, "
                f"but --max-loras={self.model_manager.lora_slots}. "
                "Set --max-loras to cover the union of adapters scheduled across DP ranks."
            )

        active_count = max(len(by_id), *(count for _, count in gathered))
        if not by_id:
            # No checkpoint can fail to load in a base-only or synthetic
            # batch, so neither activation nor its status collective is needed.
            self._set_activity(active_count)
            self.step_active = True
            return

        error = None
        try:
            self.worker_manager.set_active_adapters(set(by_id.values()), None)
            missing = by_id.keys() - set(self.model_manager.lora_index_to_id)
            if missing:
                raise RuntimeError(f"LoRA adapters are not resident after activation: {sorted(missing)}")
            self.refresh_slot_ids()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # A missing checkpoint or pinned-cache eviction must fail every peer
        # before any rank enters the MoE device collectives.
        errors = self._gather(error)
        if any(errors):
            raise RuntimeError(f"Cross-DP MoE LoRA activation failed on peer ranks: {errors}")
        self._set_activity(active_count)
        self.step_active = True

    def _set_activity(self, count: int) -> None:
        self.num_active_loras = count
        for wrapper in self.wrappers:
            wrapper.moe_lora_active = count > 0

    def refresh_slot_ids(self, *, update_activity: bool = False) -> None:
        ids = tuple(adapter_id if adapter_id is not None else -1 for adapter_id in self.model_manager.lora_index_to_id)
        if ids != self._slot_ids:
            self.slot_ids.copy_(torch.tensor(ids, dtype=torch.long, device="cpu"))
            self._slot_ids = ids
        if update_activity:
            # Capture hooks run outside execute_model. Their explicit dummy
            # adapter count selects the matching LoRA or base-only graph.
            self._set_activity(max(len(wrapper.active_lora_slots) for wrapper in self.wrappers))
