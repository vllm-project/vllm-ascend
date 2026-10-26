"""QSA hybrid cache storage.

[Role] Plan block-packed K/V sharing a physical page with GDN SSM state.
[Invariants] Pool block IDs have one group owner; K+V, SSM and PLE alias r2.
[Timeouts & Locks] Pure layout arithmetic; no waits or locks.
[Side-Effect Policy] View construction aliases caller-owned storage without copies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import regex as re
import torch
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    get_kv_cache_compression_ratio,
)

QSA_MAIN: Final = "qsa_main"
QSA_RAW: Final = "qsa_raw"
QSA_COMPRESSED: Final = "qsa_compressed"
GDN: Final = "gdn"
PLE: Final = "ple"
HIDDEN: Final = "hidden"


@dataclass(frozen=True)
class CacheOwner:
    layer_name: str
    spec: KVCacheSpec
    role: str
    slot: int
    group_id: int


@dataclass(frozen=True)
class SlabRegion:
    """One region in a slot backing: all physical blocks are contiguous."""

    name: str
    offset: int
    page_size_bytes: int
    num_blocks: int

    @property
    def size(self) -> int:
        return self.page_size_bytes * self.num_blocks

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True)
class SixRegionKVCacheLayout:
    """Shared layer slots with packed K/V-or-state pages and two indexer slabs."""

    regions: tuple[SlabRegion, ...]
    owners: tuple[CacheOwner, ...]
    slot_count: int
    slot_backing_size: int
    alignment: int
    num_blocks: int

    def region(self, name: str) -> SlabRegion:
        return next(region for region in self.regions if region.name == name)

    def owner(self, layer_name: str) -> CacheOwner:
        return next(owner for owner in self.owners if owner.layer_name == layer_name)

    def slot_shared_by(self, slot: int) -> list[str]:
        return [owner.layer_name for owner in self.owners if owner.role != HIDDEN and owner.slot == slot]

    def region_name_for_owner(self, layer_name: str) -> str:
        return {
            QSA_MAIN: "r2",
            QSA_RAW: "r4",
            QSA_COMPRESSED: "r5",
            GDN: "r1",
            PLE: "r2",
        }[self.owner(layer_name).role]


def _group_member_specs(group: KVCacheGroupSpec) -> dict[str, KVCacheSpec]:
    if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
        return group.kv_cache_spec.kv_cache_specs
    return {name: group.kv_cache_spec for name in group.layer_names}


def _classify_spec(layer_name: str, spec: KVCacheSpec) -> str | None:
    if isinstance(spec, HiddenStateCacheSpec):
        return HIDDEN
    if layer_name.endswith(".raw_key_cache") and isinstance(spec, CircularBufferSpec):
        return QSA_RAW
    if (
        layer_name.endswith(".compressed_key_cache")
        and isinstance(spec, MLAAttentionSpec)
        and get_kv_cache_compression_ratio(spec) > 1
    ):
        return QSA_COMPRESSED
    if layer_name.endswith(".attn") and isinstance(spec, FullAttentionSpec):
        return QSA_MAIN
    if isinstance(spec, MambaSpec):
        if len(spec.shapes) == 2:
            return GDN
        if len(spec.shapes) == 1:
            return PLE
    return None


def _qsa_source_name(layer_name: str, role: str) -> str:
    suffix = {
        QSA_MAIN: ".attn",
        QSA_RAW: ".indexer.raw_key_cache",
        QSA_COMPRESSED: ".indexer.compressed_key_cache",
    }[role]
    if not layer_name.endswith(suffix):
        raise ValueError(f"Invalid {role} owner name: {layer_name}")
    return layer_name[: -len(suffix)]


def _layer_sort_key(layer_name: str) -> tuple[int, str]:
    matches = re.findall(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if not matches:
        raise ValueError(f"Cannot derive source layer index from {layer_name}")
    return int(matches[-1]), layer_name


def _owner_sort_key(layer_name: str) -> tuple[int, str]:
    matches = re.findall(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    return (int(matches[-1]) if matches else 1 << 30, layer_name)


def _state_bytes(spec: MambaSpec, index: int) -> int:
    return math.prod(spec.shapes[index]) * get_dtype_size(spec.dtypes[index])


def _qsa_main_page_bytes(spec: FullAttentionSpec) -> tuple[int, int]:
    dtype_size = get_dtype_size(spec.dtype)
    tokens_heads = spec.block_size * spec.num_kv_heads
    return (
        tokens_heads * spec.head_size * dtype_size,
        tokens_heads * spec.head_size_v * dtype_size,
    )


def _real_page_size_bytes(spec: KVCacheSpec) -> int:
    return int(getattr(spec, "real_page_size_bytes", spec.page_size_bytes))


def build_six_region_kv_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
    num_blocks: int,
) -> SixRegionKVCacheLayout | None:
    """Build the hybrid layout; r2 contains packed K/V or one GDN SSM/PLE state."""

    if num_blocks <= 0:
        raise ValueError("Six-region layout requires a positive block count")
    role_members: dict[str, list[tuple[str, KVCacheSpec]]] = {
        QSA_MAIN: [],
        QSA_RAW: [],
        QSA_COMPRESSED: [],
        GDN: [],
        PLE: [],
        HIDDEN: [],
    }
    owner_group_ids: dict[str, int] = {}
    unknown: list[tuple[str, KVCacheSpec]] = []
    for group_id, group in enumerate(kv_cache_groups):
        member_specs = _group_member_specs(group)
        for layer_name in group.layer_names:
            owner_group_ids[layer_name] = group_id
            spec = member_specs[layer_name]
            role = _classify_spec(layer_name, spec)
            if role is None:
                unknown.append((layer_name, spec))
            else:
                role_members[role].append((layer_name, spec))

    if not role_members[QSA_RAW]:
        return None
    required = (QSA_MAIN, QSA_RAW, QSA_COMPRESSED, GDN, PLE)
    missing = [role for role in required if not role_members[role]]
    if missing or unknown:
        details = [(name, type(spec).__name__) for name, spec in unknown]
        raise ValueError(f"Six-region QSA hybrid cache layout is incomplete: missing={missing}, unsupported={details}")

    qsa_by_role: dict[str, dict[str, tuple[str, KVCacheSpec]]] = {}
    for role in (QSA_MAIN, QSA_RAW, QSA_COMPRESSED):
        qsa_owners: dict[str, tuple[str, KVCacheSpec]] = {}
        for layer_name, spec in role_members[role]:
            source = _qsa_source_name(layer_name, role)
            if source in qsa_owners:
                raise ValueError(f"Duplicate {role} owner for source {source}")
            qsa_owners[source] = (layer_name, spec)
        qsa_by_role[role] = qsa_owners
    source_sets = {role: set(owners) for role, owners in qsa_by_role.items()}
    if len({frozenset(sources) for sources in source_sets.values()}) != 1:
        raise ValueError(f"QSA main/raw/compressed owners do not form a one-to-one source-layer mapping: {source_sets}")
    qsa_sources = sorted(source_sets[QSA_MAIN], key=_layer_sort_key)
    for role in (QSA_MAIN, QSA_RAW, QSA_COMPRESSED):
        role_members[role] = [qsa_by_role[role][source] for source in qsa_sources]
    for role in (GDN, PLE, HIDDEN):
        role_members[role].sort(key=lambda item: _owner_sort_key(item[0]))

    dtype_sizes = {
        get_dtype_size(dtype)
        for role in required
        for _, spec in role_members[role]
        for dtype in (spec.dtypes if isinstance(spec, MambaSpec) else (spec.dtype,))
    }
    alignment = math.lcm(16, *dtype_sizes)
    gdn_specs = [spec for _, spec in role_members[GDN] if isinstance(spec, MambaSpec)]
    max_ssm_bytes = max(_state_bytes(spec, 1) for spec in gdn_specs)
    for layer_name, spec in role_members[QSA_MAIN]:
        assert isinstance(spec, FullAttentionSpec)
        kv_token_bytes = spec.num_kv_heads * (spec.head_size + spec.head_size_v) * get_dtype_size(spec.dtype)
        minimum_block_size = round_up((max_ssm_bytes + kv_token_bytes - 1) // kv_token_bytes, 128)
        if spec.block_size < minimum_block_size or spec.block_size % 128:
            raise ValueError(
                f"{layer_name} block_size={spec.block_size} violates packed K/V-SSM alignment: "
                f"minimum={minimum_block_size}, kernel_multiple=128"
            )
    for source in qsa_sources:
        main_name, main_spec = qsa_by_role[QSA_MAIN][source]
        compressed_name, compressed_spec = qsa_by_role[QSA_COMPRESSED][source]
        raw_name, raw_spec = qsa_by_role[QSA_RAW][source]
        assert isinstance(main_spec, FullAttentionSpec)
        assert isinstance(compressed_spec, MLAAttentionSpec)
        assert isinstance(raw_spec, CircularBufferSpec)
        compress_ratio = get_kv_cache_compression_ratio(compressed_spec)
        same_block_table = owner_group_ids[main_name] == owner_group_ids[compressed_name]
        if same_block_table and compressed_spec.block_size != main_spec.block_size:
            raise ValueError(
                f"{compressed_name} logical block_size="
                f"{compressed_spec.block_size} does not match QSA main "
                f"block_size={main_spec.block_size} in their composite group"
            )
        if compressed_spec.block_size % compress_ratio:
            raise ValueError(
                f"{compressed_name} block_size={compressed_spec.block_size} is not divisible by ratio={compress_ratio}"
            )
        if raw_spec.block_size % compress_ratio:
            raise ValueError(
                f"{raw_name} circular capacity={raw_spec.block_size} is not aligned to ratio={compress_ratio}"
            )
    main_pages = [
        _qsa_main_page_bytes(spec) for _, spec in role_members[QSA_MAIN] if isinstance(spec, FullAttentionSpec)
    ]
    ple_specs = [spec for _, spec in role_members[PLE] if isinstance(spec, MambaSpec)]
    # r3 used to hold a separate V slab. V now occupies the tail of each
    # r2 physical page, so GDN state aliases the complete K+V footprint.
    # GBSA exposes K and V through 128-token views, including padding when
    # a larger state determines the r2 stride. Each K/V subpage must align.
    subpage_alignment = math.lcm(
        *(
            128 * spec.num_kv_heads * size * get_dtype_size(spec.dtype)
            for _, spec in role_members[QSA_MAIN]
            for size in (spec.head_size, spec.head_size_v)
        )
    )
    r2_bytes = max(max_ssm_bytes, *(k + v for k, v in main_pages), *(_state_bytes(spec, 0) for spec in ple_specs))
    page_sizes = {
        "r1": max(_state_bytes(spec, 0) for spec in gdn_specs),
        "r2": round_up(r2_bytes, subpage_alignment),
        "r4": max(_real_page_size_bytes(spec) for _, spec in role_members[QSA_RAW]),
        "r5": max(_real_page_size_bytes(spec) for _, spec in role_members[QSA_COMPRESSED]),
    }
    regions: list[SlabRegion] = []
    cursor = 0
    for name, page_size in page_sizes.items():
        cursor = round_up(cursor, alignment)
        regions.append(SlabRegion(name, cursor, page_size, num_blocks))
        cursor += page_size * num_blocks
    slot_backing_size = round_up(cursor, alignment)

    # Each QSA layer needs one slot. State groups reuse those slots with
    # separate scheduler block tables, so live groups own disjoint block IDs.
    slot_count = len(role_members[QSA_MAIN])
    owners: list[CacheOwner] = []
    for role in required:
        group_slots: dict[int, int] = {}
        for ordinal, (layer_name, spec) in enumerate(role_members[role]):
            group_id = owner_group_ids[layer_name]
            slot = group_slots.get(group_id, 0) if role in (GDN, PLE) else ordinal
            if slot >= slot_count:
                raise ValueError(f"Cache group {group_id} exceeds the {slot_count} QSA slots")
            owners.append(CacheOwner(layer_name, spec, role, slot, group_id))
            group_slots[group_id] = slot + 1
    owners.extend(
        CacheOwner(layer_name, spec, HIDDEN, slot, owner_group_ids[layer_name])
        for slot, (layer_name, spec) in enumerate(role_members[HIDDEN])
    )
    layout = SixRegionKVCacheLayout(
        regions=tuple(regions),
        owners=tuple(owners),
        slot_count=slot_count,
        slot_backing_size=slot_backing_size,
        alignment=alignment,
        num_blocks=num_blocks,
    )

    previous_end = 0
    for region in layout.regions:
        assert region.offset >= previous_end
        assert region.offset % alignment == 0
        assert region.end <= slot_backing_size
        previous_end = region.end
    for owner in layout.owners:
        if owner.role == HIDDEN:
            continue
        region = layout.region(layout.region_name_for_owner(owner.layer_name))
        dtypes = owner.spec.dtypes if isinstance(owner.spec, MambaSpec) else (owner.spec.dtype,)
        assert all(region.offset % get_dtype_size(dtype) == 0 for dtype in dtypes)
    for _, spec in role_members[QSA_MAIN]:
        assert isinstance(spec, FullAttentionSpec)
        assert sum(_qsa_main_page_bytes(spec)) <= layout.region("r2").page_size_bytes
    return layout


def make_slab_view(
    backing: torch.Tensor,
    *,
    dtype: torch.dtype,
    num_blocks: int,
    item_shape: tuple[int, ...],
    storage_offset: int,
    block_stride: int | None = None,
) -> torch.Tensor:
    """Create dense block contents with an explicit physical byte stride."""

    dtype_size = get_dtype_size(dtype)
    if storage_offset % dtype_size:
        raise ValueError("Slab cache offset must align to the view dtype")
    item_bytes = math.prod(item_shape) * dtype_size
    block_stride = item_bytes if block_stride is None else block_stride
    if block_stride < item_bytes or block_stride % dtype_size:
        raise ValueError("Slab block stride must fit one item and align to its dtype")
    required_bytes = (num_blocks - 1) * block_stride + item_bytes
    if storage_offset + required_bytes > backing.numel():
        raise ValueError("Slab cache view exceeds its backing allocation")
    inner_strides = tuple(math.prod(item_shape[index + 1 :]) for index in range(len(item_shape)))
    result = (
        backing[storage_offset : storage_offset + required_bytes]
        .view(dtype)
        .as_strided((num_blocks, *item_shape), (block_stride // dtype_size, *inner_strides))
    )
    return result


__all__ = [
    "GDN",
    "HIDDEN",
    "PLE",
    "QSA_COMPRESSED",
    "QSA_MAIN",
    "QSA_RAW",
    "SixRegionKVCacheLayout",
    "build_six_region_kv_cache_layout",
    "make_slab_view",
]
