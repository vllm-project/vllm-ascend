# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Storage and transfer descriptions for cross-layer shared expert slots."""

from dataclasses import dataclass

import numpy as np
import torch


class SharedExpertWeights:
    """Extend kernel-owned expert lists with the same buffers for every layer.

    Call before graph capture. Routing must leave new slots inactive until
    transfers finish. The caller owns metadata updates and stream ordering.
    """

    def __init__(self, layers, slots_per_rank: int):
        if isinstance(slots_per_rank, bool) or not isinstance(slots_per_rank, int) or slots_per_rank <= 0:
            raise ValueError("slots_per_rank must be a positive integer")
        layers = list(layers)
        if not layers:
            raise ValueError("a shared pool requires at least one MoE layer")
        self.base_slots = layers[0].local_num_experts
        if self.base_slots <= 0:
            raise ValueError("each rank must have at least one immutable base expert")
        # Use the quantization method contract, including fused scales. Do not
        # duplicate format-specific tensor-name lists from legacy EPLB.
        views = [list(layer.quant_method.get_eplb_weight_views(layer)) for layer in layers]
        signatures = None
        list_ids = set()
        for layer, tensors in zip(layers, views):
            if getattr(layer.moe_config, "has_bias", False):
                raise ValueError("shared expert pools do not support expert bias")
            if layer.local_num_experts != self.base_slots or not tensors:
                raise ValueError("shared pool layers must have matching nonempty base expert layouts")
            current = []
            for experts in tensors:
                if not isinstance(experts, list) or len(experts) != self.base_slots:
                    raise ValueError("shared pool weights must be kernel-owned per-expert lists")
                if id(experts) in list_ids:
                    raise ValueError("expert lists must be distinct before shared-pool allocation")
                list_ids.add(id(experts))
                first = experts[0]
                signature = (first.shape, first.dtype, first.device)
                if any((value.shape, value.dtype, value.device) != signature for value in experts):
                    raise ValueError("expert tensors must have matching shapes, dtypes and devices")
                current.append(signature)
            if signatures is not None and current != signatures:
                raise ValueError("all shared-pool layers must use identical expert tensor formats")
            signatures = current

        # Allocate everything before extending any model-owned list. Base
        # weights and their addresses remain unchanged.
        self.pool = [[torch.empty_like(experts[0]) for _ in range(slots_per_rank)] for experts in views[0]]
        self.views = views
        for tensors in views:
            for experts, shared in zip(tensors, self.pool):
                experts.extend(shared)

    def source(self, layer: int, base_slot: int) -> tuple[torch.Tensor, ...]:
        if not 0 <= layer < len(self.views) or not 0 <= base_slot < self.base_slots:
            raise IndexError("transfer sources must be immutable base experts")
        return tuple(experts[base_slot] for experts in self.views[layer])

    def destination(self, shared_slot: int) -> tuple[torch.Tensor, ...]:
        if not 0 <= shared_slot < len(self.pool[0]):
            raise IndexError("shared slot is outside the allocated pool")
        return tuple(experts[shared_slot] for experts in self.pool)


@dataclass(frozen=True)
class SharedSlotTransfer:
    destination_rank: int
    shared_slot: int
    old_layer: int | None
    new_layer: int | None
    source_rank: int | None
    source_slot: int | None


def plan_shared_slot_transfers(current: np.ndarray, candidate: np.ndarray, shared_slots: int):
    """Describe pool ownership changes without modifying weights or routing.

    Deactivate old owners before receiving and publish new owners only after
    all tensor transfers finish. Sources are always immutable base experts,
    so reusing a shared slot cannot overwrite another transfer's source.
    """
    if isinstance(shared_slots, bool) or not isinstance(shared_slots, int) or shared_slots <= 0:
        raise ValueError("shared_slots must be a positive integer")
    current = np.asarray(current)
    candidate = np.asarray(candidate)
    if current.ndim != 3 or current.shape != candidate.shape or 0 in current.shape:
        raise ValueError("placements must have matching nonempty [layers, ranks, slots] shapes")
    if not np.issubdtype(current.dtype, np.integer) or not np.issubdtype(candidate.dtype, np.integer):
        raise ValueError("expert IDs must be integers")
    base_slots = current.shape[-1] - shared_slots
    if base_slots <= 0:
        raise ValueError("placements must retain immutable base experts")
    num_experts = current.shape[1] * base_slots
    if not np.array_equal(current[:, :, :base_slots], candidate[:, :, :base_slots]):
        raise ValueError("shared-pool migration must not change base experts")
    for table in (current, candidate):
        if np.any(table < -1) or np.any(table >= num_experts):
            raise ValueError("expert ID is outside the logical expert range")
        base = table[:, :, :base_slots].reshape(table.shape[0], -1)
        if not np.all(np.sort(base, axis=-1) == np.arange(num_experts)):
            raise ValueError("every layer must have one immutable base copy of each expert")
        if np.any((table[:, :, base_slots:] >= 0).sum(axis=0) > 1):
            raise ValueError("a shared slot cannot be owned by multiple layers")
        for layer in table:
            for rank in layer:
                active = rank[rank >= 0]
                if len(active) != len(np.unique(active)):
                    raise ValueError("a rank cannot hold duplicate copies of an expert in one layer")

    transfers = []
    for slot in range(shared_slots):
        physical_slot = base_slots + slot
        for rank in range(current.shape[1]):
            if np.array_equal(current[:, rank, physical_slot], candidate[:, rank, physical_slot]):
                continue
            old = np.flatnonzero(current[:, rank, physical_slot] >= 0)
            new = np.flatnonzero(candidate[:, rank, physical_slot] >= 0)
            old_layer = int(old[0]) if old.size else None
            new_layer = int(new[0]) if new.size else None
            source_rank = source_slot = None
            if new_layer is not None:
                expert = candidate[new_layer, rank, physical_slot]
                source_rank, source_slot = map(int, np.argwhere(current[new_layer, :, :base_slots] == expert)[0])
            transfers.append(SharedSlotTransfer(rank, slot, old_layer, new_layer, source_rank, source_slot))
    return transfers
