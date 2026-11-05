# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent contracts for model-owned layerwise cache layouts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol


@dataclass(frozen=True)
class LayerwisePolicy:
    num_shared_buffers: int
    num_prefetch_layers: int
    independent_layers: tuple[int, ...]
    excluded_owners: tuple[str, ...] = ()


@dataclass(frozen=True)
class CacheOwnerDescription:
    name: str
    physical_layer: int
    group_id: int
    resident: bool
    compatibility_class: str
    linked_set: str


@dataclass(frozen=True)
class AliasRule:
    """Explicit permission for two owners; block IDs remain group-owned."""

    first: str
    second: str
    disjoint_group_blocks: bool = False


@dataclass(frozen=True)
class BackingPlan:
    backing_id: str
    size_bytes: int
    alignment: int
    content_bytes: int


@dataclass(frozen=True)
class OwnerViewPlan:
    owner: str
    component: str
    backing_id: str
    offset_bytes: int
    dtype: str
    element_size: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]  # dtype elements, never bytes

    @property
    def span_bytes(self) -> int:
        return (1 + sum((size - 1) * stride for size, stride in zip(self.shape, self.strides))) * self.element_size


@dataclass(frozen=True)
class TransferRegionPlan:
    owner: str
    component: str
    backing_id: str
    offset_bytes: int
    content_bytes_per_block: int
    block_stride_bytes: int
    manager_block_size: int
    compression_ratio: int = 1


@dataclass(frozen=True)
class ReuseDependency:
    owner: str
    predecessors: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalCachePlan:
    adapter_id: str
    version: int
    num_blocks: int
    policy: LayerwisePolicy
    group_signature: str
    backings: tuple[BackingPlan, ...]
    owners: tuple[CacheOwnerDescription, ...]
    views: tuple[OwnerViewPlan, ...]
    transfer_regions: tuple[TransferRegionPlan, ...]
    aliases: tuple[AliasRule, ...]
    dependencies: tuple[ReuseDependency, ...]

    @property
    def required_bytes(self) -> int:
        return sum(backing.size_bytes for backing in unique_backings(self.backings).values())

    @property
    def signature(self) -> str:
        return stable_signature(asdict(self))

    @property
    def has_reuse(self) -> bool:
        return bool(self.dependencies)

    def view(self, owner: str, component: str | None = None) -> OwnerViewPlan:
        return next(
            view for view in self.views if view.owner == owner and (component is None or view.component == component)
        )

    def reserve(self, budget: int) -> PhysicalCachePlan:
        """Keep unused capacity as padding on the first backing."""
        remainder = budget - self.required_bytes
        if remainder < 0:
            raise ValueError(f"Layerwise layout needs {self.required_bytes} bytes, available {budget} bytes")
        first, *rest = unique_backings(self.backings).values()
        return replace(self, backings=(replace(first, size_bytes=first.size_bytes + remainder), *rest))


class CacheLayoutAdapter(Protocol):
    adapter_id: str
    version: int

    def build_layout(self, groups: list[Any], policy: LayerwisePolicy, num_blocks: int) -> PhysicalCachePlan: ...


def stable_signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def unique_backings(backings: tuple[BackingPlan, ...]) -> dict[str, BackingPlan]:
    result: dict[str, BackingPlan] = {}
    for backing in backings:
        if backing.backing_id in result and result[backing.backing_id] != backing:
            raise ValueError(f"Conflicting backing ID {backing.backing_id}")
        if backing.alignment <= 0 or not 0 <= backing.content_bytes <= backing.size_bytes:
            raise ValueError(f"Invalid backing {backing.backing_id}")
        result[backing.backing_id] = backing
    return result


def validate_physical_plan(plan: PhysicalCachePlan) -> None:
    backings = unique_backings(plan.backings)
    owners = {owner.name: owner for owner in plan.owners}
    if plan.num_blocks < 1 or len(owners) != len(plan.owners):
        raise ValueError("Invalid capacity or duplicate cache owner")
    views = {(view.owner, view.component): view for view in plan.views}
    regions = {(region.owner, region.component): region for region in plan.transfer_regions}
    if (
        len(views) != len(plan.views)
        or len(regions) != len(plan.transfer_regions)
        or set(views) != set(regions)
        or {v.owner for v in plan.views} != set(owners)
    ):
        raise ValueError("Views and transfer regions must cover every owner/component exactly once")
    for view in plan.views:
        backing = backings[view.backing_id]
        if (
            view.element_size <= 0
            or view.offset_bytes < 0
            or view.offset_bytes % view.element_size
            or not view.shape
            or view.shape[0] != plan.num_blocks
            or len(view.shape) != len(view.strides)
            or any(size < 1 for size in view.shape)
            or any(stride < 0 for stride in view.strides)
            or view.offset_bytes + view.span_bytes > backing.content_bytes
        ):
            raise ValueError(f"Invalid strided view for {view.owner}/{view.component}")
        region = regions[view.owner, view.component]
        if (
            region.backing_id != view.backing_id
            or region.offset_bytes != view.offset_bytes
            or not 0 < region.content_bytes_per_block <= region.block_stride_bytes
            or region.manager_block_size < 1
            or region.compression_ratio < 1
            or region.block_stride_bytes != view.strides[0] * view.element_size
            or region.content_bytes_per_block != math.prod(view.shape[1:]) * view.element_size
            or view.span_bytes != (plan.num_blocks - 1) * region.block_stride_bytes + region.content_bytes_per_block
            or region.offset_bytes + (plan.num_blocks - 1) * region.block_stride_bytes + region.content_bytes_per_block
            > backing.content_bytes
        ):
            raise ValueError(f"Invalid transfer region for {view.owner}/{view.component}")

    allowed = {frozenset((rule.first, rule.second)): rule for rule in plan.aliases}
    dependencies = {dep.owner: dep.predecessors for dep in plan.dependencies}
    if len(dependencies) != len(plan.dependencies):
        raise ValueError("Duplicate owner release dependencies")
    for dep in plan.dependencies:
        if dep.owner not in owners or owners[dep.owner].resident:
            raise ValueError("Reuse dependency targets a missing or resident owner")
        for predecessor in dep.predecessors:
            if predecessor not in owners or owners[predecessor].physical_layer >= owners[dep.owner].physical_layer:
                raise ValueError("Reuse dependencies must point to earlier physical layers")
    released_before: dict[str, set[str]] = {}
    for owner in sorted(plan.owners, key=lambda item: item.physical_layer):
        predecessors = dependencies.get(owner.name, ())
        released_before[owner.name] = set(predecessors)
        for predecessor in predecessors:
            released_before[owner.name].update(released_before[predecessor])

    for index, left in enumerate(plan.transfer_regions):
        for right in plan.transfer_regions[index + 1 :]:
            if left.backing_id != right.backing_id:
                continue
            left_end = (
                left.offset_bytes + (plan.num_blocks - 1) * left.block_stride_bytes + left.content_bytes_per_block
            )
            right_end = (
                right.offset_bytes + (plan.num_blocks - 1) * right.block_stride_bytes + right.content_bytes_per_block
            )
            if left_end <= right.offset_bytes or right_end <= left.offset_bytes:
                continue
            # Components in interleaved pages have overlapping bounding spans
            # but disjoint content. Both strides are bytes here.
            if left.block_stride_bytes == right.block_stride_bytes:
                stride = left.block_stride_bytes
                delta = (right.offset_bytes - left.offset_bytes) % stride
                if left.content_bytes_per_block <= delta and delta + right.content_bytes_per_block <= stride:
                    continue
            rule = allowed.get(frozenset((left.owner, right.owner)))
            if rule is None or left.owner == right.owner:
                raise ValueError(f"Unauthorized alias: {left.owner}/{left.component}, {right.owner}/{right.component}")
            a, b = owners[left.owner], owners[right.owner]
            if rule.disjoint_group_blocks:
                if a.group_id == b.group_id:
                    raise ValueError("Disjoint-block alias requires distinct groups")
            elif a.resident or b.resident or a.physical_layer == b.physical_layer:
                raise ValueError("Resident or simultaneous owners cannot share a rotating region")
            else:
                earlier, later = sorted((a, b), key=lambda owner: owner.physical_layer)
                if earlier.name not in released_before[later.name]:
                    raise ValueError("Rotating alias lacks a release dependency for its predecessor")
