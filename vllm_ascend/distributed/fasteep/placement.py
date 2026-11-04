# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Plan expert-level migration without requiring whole-shard replicas."""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertTransfer:
    expert: int
    source_rank: int
    source_slot: int
    target_rank: int
    target_slot: int


def validate_placement(rows: list[list[int]]) -> tuple[int, int]:
    """Validate rank rows mapping each logical expert to a local slot."""
    if not rows or not rows[0]:
        raise ValueError("Expert placement must contain ranks and logical experts.")
    logical = len(rows[0])
    capacity = sum(slot >= 0 for slot in rows[0])
    for row in rows:
        if len(row) != logical or any(type(slot) is not int or slot < -1 for slot in row):
            raise ValueError("Invalid logical-to-local expert map.")
        if sorted(slot for slot in row if slot >= 0) != list(range(capacity)):
            raise ValueError("Ranks must have equal contiguous expert capacity.")
    if any(all(row[expert] < 0 for row in rows) for expert in range(logical)):
        raise ValueError("The active placement must cover every logical expert.")
    return logical, capacity


def plan_active_placement(current: list[list[int]], templates: list[list[int]], target: int) -> list[list[int]]:
    """Keep surviving experts; replace redundant slots with missing experts.

    Joining ranks initially request their startup placement. No device tensor
    is changed here; migration must complete before the new map is committed.
    """
    logical, capacity = validate_placement(current)
    if type(target) is not int or not 1 <= target <= len(templates):
        raise ValueError("Target rank count is outside the startup group.")
    template_shape = validate_placement(templates)
    if template_shape != (logical, capacity):
        raise ValueError("Startup and active placements must have equal slot layouts.")
    if target * capacity < logical:
        raise ValueError("Target group has insufficient slots for all logical experts.")
    rows = [list(current[rank] if rank < len(current) else templates[rank]) for rank in range(target)]
    counts = Counter(expert for row in rows for expert, slot in enumerate(row) if slot >= 0)
    for missing in range(logical):
        if counts[missing]:
            continue
        for row in reversed(rows):
            victim = next((expert for expert, slot in enumerate(row) if slot >= 0 and counts[expert] > 1), None)
            if victim is not None:
                row[missing], row[victim] = row[victim], -1
                counts[victim] -= 1
                counts[missing] += 1
                break
        else:
            raise ValueError("No redundant slot is available for a missing expert.")
    validate_placement(rows)
    return rows


def plan_transfers(current: list[list[int]], target: list[list[int]]) -> list[ExpertTransfer]:
    """Choose a live donor for every changed slot before any writes occur.

    An executor must snapshot outgoing expert values before overwriting any
    slots. This also handles cycles and different layouts on every rank.
    """
    if validate_placement(current) != validate_placement(target):
        raise ValueError("Migration cannot change the per-rank expert tensor shape.")
    transfers = []
    for rank, row in enumerate(target):
        for expert, slot in enumerate(row):
            if slot < 0 or (rank < len(current) and current[rank][expert] == slot):
                continue
            donor = next((source for source, source_row in enumerate(current) if source_row[expert] >= 0), None)
            if donor is None:
                raise ValueError(f"Expert {expert} has no live source.")
            transfers.append(ExpertTransfer(expert, donor, current[donor][expert], rank, slot))
    return transfers
