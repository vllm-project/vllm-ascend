from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

import torch
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec

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
    """Six global contiguous slabs in every shared layer-slot backing."""

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
            PLE: "r6",
        }[self.owner(layer_name).role]


def _group_member_specs(group: KVCacheGroupSpec) -> dict[str, KVCacheSpec]:
    if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
        return group.kv_cache_spec.kv_cache_specs
    return {name: group.kv_cache_spec for name in group.layer_names}


def _classify_spec(layer_name: str, spec: KVCacheSpec) -> str | None:
    if isinstance(spec, HiddenStateCacheSpec):
        return HIDDEN
    if layer_name.endswith(".raw_key_cache") and isinstance(spec, AscendCircularBufferSpec):
        return QSA_RAW
    if layer_name.endswith(".compressed_key_cache") and isinstance(spec, MLAAttentionSpec) and spec.compress_ratio > 1:
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
    """Build a spec-driven six-slab layout, if the raw circular owner exists."""

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
        owners: dict[str, tuple[str, KVCacheSpec]] = {}
        for layer_name, spec in role_members[role]:
            source = _qsa_source_name(layer_name, role)
            if source in owners:
                raise ValueError(f"Duplicate {role} owner for source {source}")
            owners[source] = (layer_name, spec)
        qsa_by_role[role] = owners
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
    main_pages = [
        _qsa_main_page_bytes(spec) for _, spec in role_members[QSA_MAIN] if isinstance(spec, FullAttentionSpec)
    ]
    gdn_specs = [spec for _, spec in role_members[GDN] if isinstance(spec, MambaSpec)]
    max_ssm_bytes = max(_state_bytes(spec, 1) for spec in gdn_specs)
    for layer_name, spec in role_members[QSA_MAIN]:
        assert isinstance(spec, FullAttentionSpec)
        k_token_bytes = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
        if max_ssm_bytes % k_token_bytes:
            raise ValueError(
                f"GDN SSM page ({max_ssm_bytes} bytes) is not an integral "
                f"number of {layer_name} K tokens ({k_token_bytes} bytes/token)"
            )
        aligned_block_size = max_ssm_bytes // k_token_bytes
        target_block_size = max(spec.block_size, aligned_block_size)
        if target_block_size % 128:
            raise ValueError(
                f"{layer_name} aligned block_size={target_block_size} is not "
                "a multiple of the QSA kernel block size 128"
            )
        if spec.block_size != target_block_size:
            raise ValueError(
                f"{layer_name} block_size={spec.block_size} was not aligned "
                f"before KV-cache grouping; expected {target_block_size}"
            )
    for source in qsa_sources:
        main_name, main_spec = qsa_by_role[QSA_MAIN][source]
        compressed_name, compressed_spec = qsa_by_role[QSA_COMPRESSED][source]
        raw_name, raw_spec = qsa_by_role[QSA_RAW][source]
        assert isinstance(main_spec, FullAttentionSpec)
        assert isinstance(compressed_spec, MLAAttentionSpec)
        assert isinstance(raw_spec, AscendCircularBufferSpec)
        same_block_table = owner_group_ids[main_name] == owner_group_ids[compressed_name]
        if same_block_table and compressed_spec.block_size != main_spec.block_size:
            raise ValueError(
                f"{compressed_name} logical block_size="
                f"{compressed_spec.block_size} does not match QSA main "
                f"block_size={main_spec.block_size} in their composite group"
            )
        if compressed_spec.block_size % compressed_spec.compress_ratio:
            raise ValueError(
                f"{compressed_name} block_size={compressed_spec.block_size} "
                f"is not divisible by ratio={compressed_spec.compress_ratio}"
            )
        if raw_spec.block_size % compressed_spec.compress_ratio:
            raise ValueError(
                f"{raw_name} circular capacity={raw_spec.block_size} is not "
                f"aligned to ratio={compressed_spec.compress_ratio}"
            )
    main_pages = [
        _qsa_main_page_bytes(spec) for _, spec in role_members[QSA_MAIN] if isinstance(spec, FullAttentionSpec)
    ]
    ple_specs = [spec for _, spec in role_members[PLE] if isinstance(spec, MambaSpec)]
    page_sizes = (
        max(_state_bytes(spec, 0) for spec in gdn_specs),
        max(max(_state_bytes(spec, 1) for spec in gdn_specs), max(k_bytes for k_bytes, _ in main_pages)),
        max(v_bytes for _, v_bytes in main_pages),
        max(_real_page_size_bytes(spec) for _, spec in role_members[QSA_RAW]),
        max(_real_page_size_bytes(spec) for _, spec in role_members[QSA_COMPRESSED]),
        max(_state_bytes(spec, 0) for spec in ple_specs),
    )

    max_ssm_bytes = max(_state_bytes(spec, 1) for spec in gdn_specs)
    for layer_name, spec in role_members[QSA_MAIN]:
        assert isinstance(spec, FullAttentionSpec)
        k_token_bytes = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
        if max_ssm_bytes % k_token_bytes:
            raise ValueError(
                f"GDN SSM page ({max_ssm_bytes} bytes) is not an integral "
                f"number of {layer_name} K tokens ({k_token_bytes} bytes/token)"
            )
        aligned_block_size = max_ssm_bytes // k_token_bytes
        if spec.block_size < aligned_block_size or spec.block_size % 128:
            raise ValueError(
                f"{layer_name} block_size={spec.block_size} violates the "
                "Qwen3.5 hybrid alignment: "
                f"minimum={aligned_block_size}, kernel_multiple=128"
            )

    regions: list[SlabRegion] = []
    cursor = 0
    for index, page_size in enumerate(page_sizes, 1):
        cursor = round_up(cursor, alignment)
        regions.append(
            SlabRegion(
                name=f"r{index}",
                offset=cursor,
                page_size_bytes=page_size,
                num_blocks=num_blocks,
            )
        )
        cursor += page_size * num_blocks
    slot_backing_size = round_up(cursor, alignment)

    owners: list[CacheOwner] = []
    for role in required:
        owners.extend(
            CacheOwner(layer_name, spec, role, slot) for slot, (layer_name, spec) in enumerate(role_members[role])
        )
    owners.extend(
        CacheOwner(layer_name, spec, HIDDEN, slot) for slot, (layer_name, spec) in enumerate(role_members[HIDDEN])
    )
    slot_count = max(len(role_members[role]) for role in required)
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
        k_token_bytes = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
        assert layout.region("r2").page_size_bytes // k_token_bytes >= spec.block_size
    return layout


def make_contiguous_slab_view(
    backing: torch.Tensor,
    *,
    dtype: torch.dtype,
    num_blocks: int,
    item_shape: tuple[int, ...],
    storage_offset: int,
) -> torch.Tensor:
    """Create one contiguous [num_blocks, *item_shape] region view."""

    dtype_size = get_dtype_size(dtype)
    if storage_offset % dtype_size:
        raise ValueError("Slab cache offset must align to the view dtype")
    required_bytes = num_blocks * math.prod(item_shape) * dtype_size
    if storage_offset + required_bytes > backing.numel():
        raise ValueError("Slab cache view exceeds its backing allocation")
    result = backing[storage_offset : storage_offset + required_bytes].view(dtype).view(num_blocks, *item_shape)
    assert result.is_contiguous()
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
    "make_contiguous_slab_view",
]
