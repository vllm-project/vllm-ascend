# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidencyPlan:
    generation: int
    slot_to_expert: tuple[int | None, ...]
    loads: tuple[tuple[int, int], ...]  # (logical expert, destination slot)
    use_full_pool: bool = False


class ExpertResidency:
    """Transactional deterministic placement; publishing follows completed copies."""

    def __init__(self, num_experts: int, num_slots: int):
        if not 0 < num_slots < num_experts:
            raise ValueError("residency requires 0 < num_slots < num_experts")
        self.num_experts = num_experts
        self.slot_to_expert: tuple[int | None, ...] = (None,) * num_slots
        self.generation = 0

    def plan(self, selected_ids) -> ResidencyPlan:
        selected_ids = tuple(selected_ids)
        if any(type(eid) is not int or not 0 <= eid < self.num_experts for eid in selected_ids):
            raise ValueError("selected expert IDs must be integers within the logical expert range")
        selected = set(selected_ids)
        if len(selected) > len(self.slot_to_expert):
            return ResidencyPlan(self.generation, self.slot_to_expert, (), use_full_pool=True)
        slots = list(self.slot_to_expert)
        missing = sorted(selected - set(slots))
        # Free slots first, then ascending slots whose experts are not needed.
        victims = [i for i, eid in enumerate(slots) if eid is None]
        victims += [i for i, eid in enumerate(slots) if eid is not None and eid not in selected]
        loads = tuple(zip(missing, victims))
        for eid, slot in loads:
            slots[slot] = eid
        return ResidencyPlan(self.generation, tuple(slots), loads)

    def commit(self, plan: ResidencyPlan) -> None:
        if plan.generation != self.generation:
            raise RuntimeError("stale residency plan")
        if plan.use_full_pool:
            return
        if len(plan.slot_to_expert) != len(self.slot_to_expert):
            raise ValueError("residency plan has the wrong capacity")
        occupied = [eid for eid in plan.slot_to_expert if eid is not None]
        if len(set(occupied)) != len(occupied) or any(not 0 <= eid < self.num_experts for eid in occupied):
            raise ValueError("residency plan must be a valid expert-to-slot bijection")
        self.slot_to_expert = plan.slot_to_expert
        self.generation += 1

    def logical_to_physical(self) -> tuple[int, ...]:
        mapping = [-1] * self.num_experts
        for slot, eid in enumerate(self.slot_to_expert):
            if eid is not None:
                mapping[eid] = slot
        return tuple(mapping)
