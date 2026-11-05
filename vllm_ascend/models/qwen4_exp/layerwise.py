# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4 owner semantics and physical layout for GVA layerwise offload."""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec

from vllm_ascend.core.layerwise_cache_config import ModelKVCacheTensor, bind_model_plan
from vllm_ascend.core.layerwise_cache_plan import (
    AliasRule,
    BackingPlan,
    CacheOwnerDescription,
    LayerwisePolicy,
    OwnerViewPlan,
    PhysicalCachePlan,
    ReuseDependency,
    TransferRegionPlan,
    stable_signature,
)
from vllm_ascend.core.layerwise_cache_planner import assign_rotating_slots, normalize_layerwise_policy
from vllm_ascend.models.qwen4_exp.cache_config import (
    GDN,
    HIDDEN,
    PLE,
    QSA_COMPRESSED,
    QSA_MAIN,
    QSA_RAW,
    CacheOwner,
    _owner_sort_key,
    _private_page_bytes,
    build_qwen4_kv_cache_layout,
    make_qsa_kernel_page_cache,
    make_slab_view,
)


def _physical_layer(owner: CacheOwner) -> int:
    # PLE executes before the GDN of decoder layer 1, sharing that layer's
    # transfer completion. Hidden/draft owners have independent storage.
    if owner.role == HIDDEN:
        return 0
    return 1 if owner.role == PLE else _owner_sort_key(owner.layer_name)[0]


class Qwen4LayerwiseAdapter:
    adapter_id = "qwen4_exp"
    version = 1

    def policy(self, groups, extra_config, excluded_owners=()) -> LayerwisePolicy:
        layout = build_qwen4_kv_cache_layout(groups, 1)
        assert layout is not None
        layers = {_physical_layer(owner) for owner in layout.owners if owner.role in (QSA_MAIN, GDN)}
        return normalize_layerwise_policy(layers, extra_config, excluded_owners)

    def build_layout(
        self, groups: list[KVCacheGroupSpec], policy: LayerwisePolicy, num_blocks: int
    ) -> PhysicalCachePlan:
        original = build_qwen4_kv_cache_layout(groups, num_blocks)
        if original is None:
            raise ValueError("Qwen4 layerwise adapter requires a complete QSA hybrid layout")
        descriptions = []
        for owner in original.owners:
            layer = _physical_layer(owner)
            resident = (
                owner.role in (QSA_RAW, PLE, HIDDEN)
                or layer in policy.independent_layers
                or owner.layer_name in policy.excluded_owners
            )
            # Main and compressed are inseparable: excluding either excludes
            # the full linked unit, even though their bytes are in two slabs.
            if owner.role in (QSA_MAIN, QSA_COMPRESSED):
                resident |= any(
                    _physical_layer(other) == layer and other.layer_name in policy.excluded_owners
                    for other in original.owners
                    if other.role in (QSA_MAIN, QSA_COMPRESSED)
                )
            descriptions.append(
                CacheOwnerDescription(
                    owner.layer_name,
                    layer,
                    owner.group_id,
                    resident,
                    "qsa" if owner.role in (QSA_MAIN, QSA_COMPRESSED) else owner.role,
                    f"layer:{layer}",
                )
            )
        by_name = {owner.name: owner for owner in descriptions}
        for owner in descriptions:
            group = groups[owner.group_id]
            if getattr(group, "host_resident", False):
                raise ValueError("Qwen4 layerwise layout requires device-resident cache groups")
            if not owner.resident and not getattr(group, "enable_kv_transfer", True):
                raise ValueError(f"Rotating owner {owner.name} must participate in KV transfer")
        slots = assign_rotating_slots(descriptions, policy)
        state_owners = [owner for owner in original.owners if owner.role in (QSA_MAIN, GDN)]
        resident_layers = sorted(
            {_physical_layer(owner) for owner in state_owners if by_name[owner.layer_name].resident}
        )
        resident_slots = {layer: index for index, layer in enumerate(resident_layers)}
        shared_count = (
            max((slot for name, slot in slots.items() if by_name[name].compatibility_class in ("qsa", GDN)), default=-1)
            + 1
        )
        main_count = len(resident_slots) + shared_count
        stride = original.region("r0").block_stride
        cursor = main_count * stride * num_blocks
        offsets: dict[str, int] = {}
        for owner in state_owners:
            description = by_name[owner.layer_name]
            slot = (
                resident_slots[description.physical_layer]
                if description.resident
                else len(resident_slots) + slots[owner.layer_name]
            )
            offsets[owner.layer_name] = slot * stride * num_blocks

        compressed = [owner for owner in original.owners if owner.role == QSA_COMPRESSED]
        compressed_resident = [owner for owner in compressed if by_name[owner.layer_name].resident]
        compressed_slots = {owner.layer_name: index for index, owner in enumerate(compressed_resident)}
        compressed_count = (
            len(compressed_resident)
            + max(
                (slots[owner.layer_name] for owner in compressed if not by_name[owner.layer_name].resident), default=-1
            )
            + 1
        )
        compressed_stride = original.region("r5").block_stride
        for owner in compressed:
            slot = (
                compressed_slots[owner.layer_name]
                if by_name[owner.layer_name].resident
                else len(compressed_resident) + slots[owner.layer_name]
            )
            offsets[owner.layer_name] = cursor + slot * compressed_stride * num_blocks
        cursor += compressed_count * compressed_stride * num_blocks
        raw_stride = original.region("r4").block_stride
        for owner in original.owners:
            if owner.role == QSA_RAW:
                offsets[owner.layer_name] = cursor
                cursor += raw_stride * num_blocks

        backings = [BackingPlan("shared", cursor, original.alignment, cursor)]
        views, regions = [], []
        for owner in original.owners:
            spec = owner.spec
            backing_id = "shared"
            if owner.role in (PLE, HIDDEN):
                backing_id = f"private:{owner.layer_name}"
                size = _private_page_bytes(owner) * num_blocks
                backings.append(BackingPlan(backing_id, size, original.alignment, size))
                offsets[owner.layer_name] = 0
            offset = offsets[owner.layer_name]
            if owner.role == GDN:
                components = [
                    ("conv", spec.dtypes[0], tuple(spec.shapes[0]), offset, stride),
                    ("ssm", spec.dtypes[1], tuple(spec.shapes[1]), offset + original.region("r2").offset, stride),
                ]
            elif owner.role == PLE:
                components = [("state", spec.dtypes[0], tuple(spec.shapes[0]), 0, _private_page_bytes(owner))]
            else:
                page = {QSA_MAIN: stride, QSA_COMPRESSED: compressed_stride, QSA_RAW: raw_stride}.get(
                    owner.role, spec.page_size_bytes
                )
                components = [("cache", spec.dtype, (page // get_dtype_size(spec.dtype),), offset, page)]
            for component, dtype, shape, component_offset, block_stride in components:
                element_size = get_dtype_size(dtype)
                inner_strides = tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))
                views.append(
                    OwnerViewPlan(
                        owner.layer_name,
                        component,
                        backing_id,
                        component_offset,
                        str(dtype),
                        element_size,
                        (num_blocks, *shape),
                        (block_stride // element_size, *inner_strides),
                    )
                )
                regions.append(
                    TransferRegionPlan(
                        owner.layer_name,
                        component,
                        backing_id,
                        component_offset,
                        math.prod(shape) * element_size,
                        block_stride,
                        spec.block_size,
                        spec.tokens_per_state if owner.role == QSA_COMPRESSED else 1,
                    )
                )

        # Dependencies only relate owners in the same group/block lifetime.
        # Cross-group aliases are safe through BlockPool's unique ID ownership.
        chains: dict[tuple[int, int, str], list[CacheOwner]] = {}
        for owner in original.owners:
            if owner.role in (QSA_MAIN, QSA_COMPRESSED, GDN) and not by_name[owner.layer_name].resident:
                chains.setdefault((owner.group_id, offsets[owner.layer_name], owner.role), []).append(owner)
        dependencies = []
        for chain in chains.values():
            ordered = sorted(chain, key=_physical_layer)
            for previous, owner in zip(ordered, ordered[1:]):
                dependencies.append(ReuseDependency(owner.layer_name, (previous.layer_name,)))
        aliases = []
        for index, first in enumerate(original.owners):
            if first.role not in (QSA_MAIN, QSA_COMPRESSED, GDN):
                continue
            for second in original.owners[index + 1 :]:
                if (
                    second.role not in (QSA_MAIN, QSA_COMPRESSED, GDN)
                    or offsets[first.layer_name] != offsets[second.layer_name]
                ):
                    continue
                if first.group_id != second.group_id:
                    aliases.append(AliasRule(first.layer_name, second.layer_name, True))
                elif not by_name[first.layer_name].resident and not by_name[second.layer_name].resident:
                    aliases.append(AliasRule(first.layer_name, second.layer_name))
        return PhysicalCachePlan(
            self.adapter_id,
            self.version,
            num_blocks,
            policy,
            stable_signature([asdict(group) for group in groups]),
            tuple(backings),
            tuple(descriptions),
            tuple(views),
            tuple(regions),
            tuple(aliases),
            tuple(dependencies),
        )

    def export(self, config: KVCacheConfig, plan: PhysicalCachePlan) -> KVCacheConfig:
        original = build_qwen4_kv_cache_layout(config.kv_cache_groups, plan.num_blocks)
        assert original is not None
        backings = {backing.backing_id: backing for backing in plan.backings}
        tensors = []
        # Preserve the original role/group descriptor order and its owner order.
        for role in (QSA_MAIN, QSA_RAW, QSA_COMPRESSED, GDN, PLE, HIDDEN):
            buckets: dict[int | str, list[str]] = {}
            for owner in original.owners:
                if owner.role == role:
                    buckets.setdefault(owner.layer_name if role in (PLE, HIDDEN) else owner.group_id, []).append(
                        owner.layer_name
                    )
            for names in buckets.values():
                view = plan.view(names[0])
                tensors.append(
                    ModelKVCacheTensor(
                        size=backings[view.backing_id].size_bytes,
                        layers=names,
                        layer_stride=-1,
                        block_stride=view.strides[0] * view.element_size,
                        offset=view.offset_bytes,
                        backing_id=view.backing_id,
                        layout_signature=plan.signature,
                    )
                )
        return bind_model_plan(config, plan, tensors)

    def reshape(self, plan: PhysicalCachePlan, name: str, spec, backing: torch.Tensor, kernel_block_size: int):
        view = plan.view(name)
        description = next(owner for owner in plan.owners if owner.name == name)
        if description.compatibility_class == "qsa" and name.endswith(".attn"):
            return make_qsa_kernel_page_cache(
                backing,
                dtype=spec.dtype,
                num_blocks=plan.num_blocks,
                manager_block_size=spec.block_size,
                kernel_block_size=kernel_block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                head_size_v=spec.head_size_v,
                storage_offset=view.offset_bytes,
                block_stride=view.strides[0] * view.element_size,
            )
        if isinstance(spec, MambaSpec):
            return [
                make_slab_view(
                    backing,
                    dtype=dtype,
                    num_blocks=plan.num_blocks,
                    item_shape=tuple(shape),
                    storage_offset=component.offset_bytes,
                    block_stride=component.strides[0] * component.element_size,
                )
                for component, dtype, shape in zip(
                    (v for v in plan.views if v.owner == name), spec.dtypes, spec.shapes, strict=True
                )
            ]
        if description.compatibility_class in ("qsa", QSA_RAW):
            compressed = description.compatibility_class == "qsa"
            pages = spec.block_size // kernel_block_size if compressed else 1
            states = spec.get_num_kernel_states(kernel_block_size) if compressed else spec.block_size
            return make_slab_view(
                backing,
                dtype=spec.dtype,
                num_blocks=plan.num_blocks * pages,
                item_shape=(spec.num_kv_heads, states, spec.head_size),
                storage_offset=view.offset_bytes,
            ).transpose(1, 2)
        return None  # Hidden-state views retain the existing private reshape.
