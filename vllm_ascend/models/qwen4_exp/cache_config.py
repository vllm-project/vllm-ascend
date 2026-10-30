"""QSA hybrid cache storage."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import regex as re
import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size

try:
    from vllm.v1.core.kv_cache_utils import may_override_num_blocks
except ImportError:
    from vllm.v1.core.kv_cache_planning import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
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
    """A state component with dense contents and an explicit byte block stride."""

    name: str
    offset: int
    page_size_bytes: int
    num_blocks: int
    block_stride_bytes: int | None = None

    @property
    def block_stride(self) -> int:
        return self.page_size_bytes if self.block_stride_bytes is None else self.block_stride_bytes

    @property
    def size(self) -> int:
        return (self.num_blocks - 1) * self.block_stride + self.page_size_bytes

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True)
class Qwen4KVCacheLayout:
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
        return [owner.layer_name for owner in self.owners if owner.role not in (HIDDEN, PLE) and owner.slot == slot]

    def region_name_for_owner(self, layer_name: str) -> str:
        return {
            QSA_MAIN: "r0",
            QSA_RAW: "r4",
            QSA_COMPRESSED: "r5",
            GDN: "r1",
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


def build_qwen4_kv_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
    num_blocks: int,
) -> Qwen4KVCacheLayout | None:
    """Build shared QSA/GDN slots; PLE and hidden states have private storage."""

    if num_blocks <= 0:
        raise ValueError("Qwen4 hybrid layout requires a positive block count")
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
        raise ValueError(f"Qwen4 QSA hybrid cache layout is incomplete: missing={missing}, unsupported={details}")

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
    conv_bytes = max(_state_bytes(spec, 0) for spec in gdn_specs)
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
    # Independent scheduler groups own disjoint block IDs. QSA uses the
    # whole page; GDN interprets the same bytes as adjacent conv and SSM.
    subpage_alignment = math.lcm(
        *(
            2 * 128 * spec.num_kv_heads * size * get_dtype_size(spec.dtype)
            for _, spec in role_members[QSA_MAIN]
            for size in (spec.head_size, spec.head_size_v)
        )
    )
    state_stride = round_up(max(conv_bytes + max_ssm_bytes, *(k + v for k, v in main_pages)), subpage_alignment)
    regions = [
        SlabRegion("r0", 0, state_stride, num_blocks),
        SlabRegion("r1", 0, conv_bytes, num_blocks, state_stride),
        SlabRegion("r2", conv_bytes, max_ssm_bytes, num_blocks, state_stride),
    ]
    cursor = state_stride * num_blocks
    for name, page_size in {
        "r4": max(_real_page_size_bytes(spec) for _, spec in role_members[QSA_RAW]),
        "r5": max(_real_page_size_bytes(spec) for _, spec in role_members[QSA_COMPRESSED]),
    }.items():
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
    layout = Qwen4KVCacheLayout(
        regions=tuple(regions),
        owners=tuple(owners),
        slot_count=slot_count,
        slot_backing_size=slot_backing_size,
        alignment=alignment,
        num_blocks=num_blocks,
    )

    assert layout.region("r1").offset + conv_bytes <= layout.region("r2").offset
    assert layout.region("r2").offset + max_ssm_bytes <= state_stride
    assert layout.region("r4").offset >= state_stride * num_blocks
    assert layout.region("r5").offset >= layout.region("r4").end
    for region in layout.regions:
        assert region.offset % alignment == 0
        assert region.end <= slot_backing_size
    for owner in layout.owners:
        if owner.role in (HIDDEN, PLE):
            continue
        region = layout.region(layout.region_name_for_owner(owner.layer_name))
        dtypes = owner.spec.dtypes if isinstance(owner.spec, MambaSpec) else (owner.spec.dtype,)
        assert all(region.offset % get_dtype_size(dtype) == 0 for dtype in dtypes)
    for _, spec in role_members[QSA_MAIN]:
        assert isinstance(spec, FullAttentionSpec)
        assert sum(_qsa_main_page_bytes(spec)) <= layout.region("r0").page_size_bytes
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


def make_qsa_kernel_page_cache(
    backing: torch.Tensor,
    *,
    dtype: torch.dtype,
    num_blocks: int,
    manager_block_size: int,
    kernel_block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    storage_offset: int,
    block_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave K/V per kernel page and expose two strided tensor views."""
    dtype_size = get_dtype_size(dtype)
    page_bytes = kernel_block_size * num_kv_heads * head_size * dtype_size
    pair_bytes = 2 * page_bytes
    if manager_block_size % kernel_block_size or block_stride % pair_bytes or head_size != head_size_v:
        raise ValueError("QSA kernel-page pools require equal K/V widths and whole-page strides")
    kernel_pages = manager_block_size // kernel_block_size
    # vLLM grows manager blocks to fit the full Mamba page. The resulting
    # K/V pairs must be contiguous so its block IDs and slots need no remap.
    if block_stride != kernel_pages * pair_bytes:
        raise ValueError("QSA manager page must match its aligned kernel K/V pairs")
    physical_page_count = num_blocks * kernel_pages
    key, value = (
        make_slab_view(
            backing,
            dtype=dtype,
            num_blocks=physical_page_count,
            item_shape=(kernel_block_size, num_kv_heads, width),
            storage_offset=storage_offset + offset,
            block_stride=pair_bytes,
        )
        for width, offset in ((head_size, 0), (head_size_v, page_bytes))
    )
    return key, value


def _prepare_qsa_composite_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> tuple[set[str], set[str], set[str]] | None:
    """Identify the three cache owners of every QSA source layer.

    The Qwen4 hybrid backing aliases main K/V and compressed-key pages by the
    same physical block ID, while the raw-key ring has one request-lifetime
    block.  Grouping must therefore express those lifetimes before the main
    block manager and admission planner see the specs.
    """

    raw = {
        name[: -len(".indexer.raw_key_cache")]: (name, spec)
        for name, spec in kv_cache_spec.items()
        if name.endswith(".indexer.raw_key_cache") and isinstance(spec, CircularBufferSpec)
    }
    if not raw:
        return None
    all_main = {
        name[: -len(".attn")]: (name, spec)
        for name, spec in kv_cache_spec.items()
        if name.endswith(".attn") and isinstance(spec, FullAttentionSpec) and not isinstance(spec, MLAAttentionSpec)
    }
    all_compressed = {
        name[: -len(".indexer.compressed_key_cache")]: (name, spec)
        for name, spec in kv_cache_spec.items()
        if name.endswith(".indexer.compressed_key_cache")
        and isinstance(spec, MLAAttentionSpec)
        and get_kv_cache_compression_ratio(spec) > 1
    }
    missing_main = set(raw) - set(all_main)
    missing_compressed = set(raw) - set(all_compressed)
    if missing_main or missing_compressed:
        raise ValueError(
            "QSA raw cache owners must have matching main/compressed owners: "
            f"missing_main={sorted(missing_main)}, "
            f"missing_compressed={sorted(missing_compressed)}"
        )
    main = {source: all_main[source] for source in raw}
    compressed = {source: all_compressed[source] for source in raw}
    if not any(isinstance(spec, MambaSpec) and len(spec.shapes) == 2 for spec in kv_cache_spec.values()):
        raise ValueError("Qwen4 QSA cache layout requires GDN state specs")

    main_names: set[str] = set()
    compressed_names: set[str] = set()
    raw_names: set[str] = set()
    for source in sorted(main):
        main_name, main_spec = main[source]
        compressed_name, compressed_spec = compressed[source]
        raw_name, raw_spec = raw[source]
        assert isinstance(main_spec, FullAttentionSpec)
        assert isinstance(compressed_spec, MLAAttentionSpec)
        assert isinstance(raw_spec, CircularBufferSpec)
        ratio = get_kv_cache_compression_ratio(compressed_spec)
        if main_spec.block_size != compressed_spec.block_size:
            raise ValueError(
                f"{main_name} block_size={main_spec.block_size} does not match "
                f"{compressed_name} block_size={compressed_spec.block_size}"
            )
        if main_spec.block_size % 128 or main_spec.block_size % ratio:
            raise ValueError(
                f"QSA block_size={main_spec.block_size} violates kernel/compression alignment (ratio={ratio})"
            )
        if raw_spec.block_size % ratio:
            raise ValueError(
                f"{raw_name} capacity={raw_spec.block_size} is not compression-ratio aligned (ratio={ratio})"
            )
        main_names.add(main_name)
        compressed_names.add(compressed_name)
        raw_names.add(raw_name)
    return main_names, compressed_names, raw_names


def _merge_qsa_composite_groups(
    groups: list[KVCacheGroupSpec],
    kv_cache_spec: dict[str, KVCacheSpec],
    main_names: set[str],
    compressed_names: set[str],
    raw_names: set[str],
) -> list[KVCacheGroupSpec]:
    """Give QSA aliases the block-table lifetimes their backing requires."""

    def merge_owners(
        current: list[KVCacheGroupSpec],
        owner_names: set[str],
        description: str,
        ordered_names: list[str] | None = None,
    ) -> list[KVCacheGroupSpec]:
        consumed: list[int] = []
        eagle = False
        for index, group in enumerate(current):
            members = set(group.layer_names)
            if not members & owner_names:
                continue
            if not members <= owner_names:
                raise ValueError(f"QSA {description} owners were mixed with another cache role")
            consumed.append(index)
            eagle = eagle or group.is_eagle_group
        if not consumed:
            raise ValueError(f"QSA {description} owners were not grouped")
        covered = set().union(*(set(current[index].layer_names) for index in consumed))
        if covered != owner_names:
            raise ValueError(f"QSA {description} grouping lost cache owners")
        ordered = ordered_names or [name for name in kv_cache_spec if name in owner_names]
        if set(ordered) != owner_names or len(ordered) != len(owner_names):
            raise ValueError(f"QSA {description} owner order does not match its cache owners")
        uniform = UniformTypeKVCacheSpecs.from_specs({name: kv_cache_spec[name] for name in ordered})
        if uniform is None:
            raise ValueError(f"QSA {description} owners do not have one lifetime")
        merged = KVCacheGroupSpec(
            ordered,
            uniform,
            is_eagle_group=eagle,
        )
        first = min(consumed)
        return [
            merged if index == first else group
            for index, group in enumerate(current)
            if index == first or index not in consumed
        ]

    result = merge_owners(
        groups,
        main_names | compressed_names,
        "main/compressed",
        [name for name in kv_cache_spec if name in main_names]
        + [name for name in kv_cache_spec if name in compressed_names],
    )
    return merge_owners(result, raw_names, "raw circular")


def _merge_mamba_state_groups(
    groups: list[KVCacheGroupSpec],
    slot_count: int,
) -> list[KVCacheGroupSpec]:
    """Pack compatible state layers into groups bounded by the QSA slot count.

    Each group gets a separate scheduler block table and reuses slot ordinals.
    Geometry and draft/target boundaries remain separate allocation domains.
    """
    buckets: dict[tuple, list[tuple[int, str, MambaSpec]]] = {}
    for index, group in enumerate(groups):
        spec = group.kv_cache_spec
        members = (
            spec.kv_cache_specs
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else {name: spec for name in group.layer_names}
        )
        if not all(isinstance(member, MambaSpec) for member in members.values()):
            continue
        for name, member in members.items():
            key = (member, group.is_eagle_group)
            buckets.setdefault(key, []).append((index, name, member))
    replacements: dict[int, list[KVCacheGroupSpec]] = {}
    drop: set[int] = set()
    for members in buckets.values():
        first = members[0][0]
        drop.update(index for index, _, _ in members)
        for start in range(0, len(members), slot_count):
            chunk = members[start : start + slot_count]
            specs = {name: spec for _, name, spec in chunk}
            uniform = UniformTypeKVCacheSpecs.from_specs(specs)
            assert uniform is not None
            replacements.setdefault(first, []).append(
                KVCacheGroupSpec(
                    list(specs),
                    chunk[0][2] if len(chunk) == 1 else uniform,
                    is_eagle_group=groups[first].is_eagle_group,
                )
            )
    result: list[KVCacheGroupSpec] = []
    for index, group in enumerate(groups):
        result.extend(replacements.get(index, []))
        if index not in drop:
            result.append(group)
    return result


def _private_page_bytes(owner: CacheOwner) -> int:
    # PLE has one dense state bank, independent of QSA slot count and page padding.
    return owner.spec.real_page_size_bytes if owner.role == PLE else owner.spec.page_size_bytes


def get_qwen4_exp_pool_bytes_per_block(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int | None:
    """Charge shared slots once, plus one private bank per PLE/hidden owner."""
    layout = build_qwen4_kv_cache_layout(kv_cache_groups, num_blocks=1)
    if layout is None:
        return None
    private_bytes = sum(_private_page_bytes(owner) for owner in layout.owners if owner.role in (HIDDEN, PLE))
    return layout.slot_count * layout.slot_backing_size + private_bytes


def get_qwen4_exp_max_memory_usage(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int | None:
    """Use the allocation divisor for admission across all group lifetimes."""
    bytes_per_block = get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)
    if bytes_per_block is None:
        return None
    required_blocks = sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), group.kv_cache_spec.page_size_bytes)
        for group in kv_cache_groups
    )
    return required_blocks * bytes_per_block


def get_qwen4_exp_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    get_default_groups: Callable[[VllmConfig, dict[str, KVCacheSpec]], list[KVCacheGroupSpec]],
) -> list[KVCacheGroupSpec] | None:
    """Build model-specific alias lifetimes on top of upstream grouping."""
    owners = _prepare_qsa_composite_groups(kv_cache_spec)
    if owners is None:
        return None
    groups = get_default_groups(vllm_config, kv_cache_spec)
    groups = _merge_qsa_composite_groups(groups, kv_cache_spec, *owners)
    groups = _merge_mamba_state_groups(groups, len(owners[0]))
    logger.info(
        "Using Qwen4 cache grouping: %d main/compressed owners, %d raw circular owners, %d total cache groups",
        len(owners[0]) + len(owners[1]),
        len(owners[2]),
        len(groups),
    )
    return groups


def get_qwen4_exp_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig | None:
    """Plan state slabs and block-packed K/V-or-SSM pages per ordinal slot.

    The main descriptor API cannot express that one logical QSA/GDN owner has
    two disjoint physical regions.  Descriptors therefore publish the anchor
    region and shared backing geometry; the Ascend runner reconstructs every
    role from the same capability-derived layout.  Different cache groups
    deliberately alias the same allocation because a physical block ID is
    owned by only one group at a time.
    """
    probe = build_qwen4_kv_cache_layout(
        kv_cache_groups,
        num_blocks=1,
    )
    if probe is None:
        return None

    private_owners = [owner for owner in probe.owners if owner.role in (HIDDEN, PLE)]
    private_bytes_per_block = sum(_private_page_bytes(owner) for owner in private_owners)
    slab_bytes_per_block = probe.slot_backing_size
    bytes_per_block = probe.slot_count * slab_bytes_per_block + private_bytes_per_block
    candidate = available_memory // bytes_per_block
    while candidate > 0:
        candidate_layout = build_qwen4_kv_cache_layout(
            kv_cache_groups,
            num_blocks=candidate,
        )
        assert candidate_layout is not None
        required = (
            candidate_layout.slot_count * candidate_layout.slot_backing_size + private_bytes_per_block * candidate
        )
        if required <= available_memory:
            break
        candidate -= 1
    num_blocks = may_override_num_blocks(vllm_config, candidate)
    layout = build_qwen4_kv_cache_layout(
        kv_cache_groups,
        num_blocks=num_blocks,
    )
    assert layout is not None
    backing_size = layout.slot_count * layout.slot_backing_size
    logger.info_once(
        "Qwen4 cache pool: %d blocks, shared page %d bytes, pool block %d bytes, available %d bytes",
        num_blocks,
        layout.region("r0").page_size_bytes,
        bytes_per_block,
        available_memory,
    )

    region_by_role = {
        QSA_MAIN: "r0",
        QSA_RAW: "r4",
        QSA_COMPRESSED: "r5",
        GDN: "r1",
    }
    tensors: list[KVCacheTensor] = []
    for role, region_name in region_by_role.items():
        owners_by_group: dict[int, list[CacheOwner]] = {}
        for owner in layout.owners:
            if owner.role == role:
                owners_by_group.setdefault(owner.group_id, []).append(owner)
        region = layout.region(region_name)
        for group_owners in owners_by_group.values():
            owners = sorted(group_owners, key=lambda owner: owner.slot)
            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=[owner.layer_name for owner in owners],
                    layer_stride=layout.slot_backing_size,
                    block_stride=region.block_stride,
                    offset=region.offset + owners[0].slot * layout.slot_backing_size,
                )
            )
    tensors.extend(
        KVCacheTensor(
            size=_private_page_bytes(owner) * num_blocks,
            layers=[owner.layer_name],
            layer_stride=_private_page_bytes(owner) * num_blocks,
            block_stride=_private_page_bytes(owner),
            offset=0,
        )
        for owner in private_owners
    )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(vllm_config.cache_config.prefix_cache_retention_interval),
    )
