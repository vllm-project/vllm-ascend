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
from vllm.distributed.parallel_state import get_ep_group

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

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        worker_manager: LRUCacheWorkerLoRAManager,
        *,
        dp_size: int,
    ) -> MoELoRACoordinator | None:
        if dp_size <= 1:
            return None
        # Reuse the contexts installed by the existing MoE LoRA wrapper. The
        # runner need not depend on a particular expert wrapper or quantizer.
        wrappers = {
            context.punica_wrapper
            for module in model.modules()
            if (context := getattr(module, "_ascend_moe_lora_context", None)) is not None
        }
        if not wrappers:
            return None
        # EP's CPU group spans all ranks sharing experts, including flattened
        # MoE-TP execution without expert parallelism.
        return cls(worker_manager, list(wrappers), get_ep_group().cpu_group)

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
        self.step_active = False
        self.slot_ids = torch.full((self.model_manager.lora_slots,), -1, dtype=torch.long, device=worker_manager.device)
        for wrapper in wrappers:
            wrapper.moe_lora_slot_ids = self.slot_ids

    def _gather(self, value):
        values = [None] * dist.get_world_size(self.cpu_group)
        dist.all_gather_object(values, value, group=self.cpu_group)
        return values

    def synchronize(self, requests: set[LoRARequest]) -> None:
        self.step_active = False
        by_id = {}
        for rank_requests in self._gather(list(requests)):
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
        self.step_active = True

    def refresh_slot_ids(self) -> None:
        ids = [adapter_id if adapter_id is not None else -1 for adapter_id in self.model_manager.lora_index_to_id]
        self.slot_ids.copy_(torch.tensor(ids, dtype=torch.long, device="cpu"))
