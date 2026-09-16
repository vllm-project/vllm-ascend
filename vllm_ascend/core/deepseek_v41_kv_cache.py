# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache specs and layer-outermost hybrid allocation."""

from dataclasses import dataclass, replace

import torch
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)
from vllm_ascend.utils import vllm_version_is

STATE_RING_ROWS = 32


class _DeepseekV41CacheSpec:
    @property
    def cache_layout(self):
        return _CACHE_LAYOUT


@dataclass(frozen=True, kw_only=True)
class DeepseekV41FullSpec(_DeepseekV41CacheSpec, AscendMLAAttentionSpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and s.block_size == self.block_size
            and get_kv_cache_compression_ratio(s) in (1, 2)
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41IndexerSpec(_DeepseekV41CacheSpec, AscendMLAAttentionSpec):
    """INT8 index keys followed by FP16 scales inside each shared slot page."""

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and s.block_size == self.block_size
            and get_kv_cache_compression_ratio(s) in (1, 2)
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41SWASpec(_DeepseekV41CacheSpec, AscendSlidingWindowMLASpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41SWASpec) and s.sliding_window == self.sliding_window for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41DraftSWASpec(_DeepseekV41CacheSpec, AscendSlidingWindowMLASpec):
    """DSpark SWA owned by G12, aliasing target slots at distinct block IDs."""

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41DraftSWASpec)
            and s.block_size == self.block_size
            and s.sliding_window == self.sliding_window
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41CompressorStateSpec(_DeepseekV41CacheSpec, CircularBufferSpec):
    """One private FP32 KV/score ring page for each active request."""

    compress_ratio: int = 1
    head_size_v: int = 0

    @property
    def storage_block_size(self):
        return self.block_size


def is_deepseek_v41_cache_spec(spec):
    return isinstance(
        spec,
        (
            DeepseekV41FullSpec,
            DeepseekV41IndexerSpec,
            DeepseekV41SWASpec,
            DeepseekV41DraftSWASpec,
            DeepseekV41CompressorStateSpec,
        ),
    )


def _uniform(members, label):
    return UniformTypeKVCacheSpecs.from_specs(members)


@dataclass(frozen=True)
class CachePlacement:
    name: str
    offset: int
    page_size_bytes: int


@dataclass(frozen=True)
class CacheSlot:
    page_size_bytes: int
    placements: tuple[CachePlacement, ...]


def _layer_number(name):
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def _cache_plane_sizes(spec):
    rows = get_storage_block_size(spec) * spec.num_kv_heads
    key_bytes = rows * spec.head_size * spec.dtype.itemsize
    if isinstance(spec, DeepseekV41IndexerSpec):
        return key_bytes, rows * spec.scale_dim * spec.scale_dtype.itemsize
    return (key_bytes,)


def _draft_layer_number(name):
    return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])


def plan_cache_slots(specs):
    """Place source KV/index tuples, state and SWA in four shared layer slots.

    Sizes come from payloads, never previously padded specs. Different groups
    overlay a slot at distinct live block IDs; a source's KV and index share
    the same ID at disjoint offsets within its page.
    """
    full = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41FullSpec)), key=_layer_number)
    state = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41CompressorStateSpec)), key=_layer_number)
    swa = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41SWASpec)), key=_layer_number)
    draft = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41DraftSWASpec)), key=_draft_layer_number)

    slots = []
    for slot_idx, kv_name in enumerate(full):
        prefix = kv_name.rsplit(".", 1)[0]
        index_name = prefix + ".indexer.k_cache"
        index_spec = specs.get(index_name)
        kv_spec = specs[kv_name]
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + swa[slot_idx :: len(full)]
        kv_bytes = sum(_cache_plane_sizes(kv_spec))
        index_bytes = sum(_cache_plane_sizes(index_spec))
        capacity = max(kv_bytes + index_bytes, *(sum(_cache_plane_sizes(specs[n])) for n in aliases))
        if slot_idx < len(draft):
            draft_name = draft[slot_idx]
            aliases.append(draft_name)
        placements = [
            CachePlacement(kv_name, 0, kv_bytes),
            CachePlacement(index_name, kv_bytes, capacity - kv_bytes),
            *(CachePlacement(name, 0, capacity) for name in aliases),
        ]
        slots.append(CacheSlot(capacity, tuple(placements)))
    return tuple(slots)


def group_cache_specs(specs):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    if not any(is_deepseek_v41_cache_spec(s) for s in specs.values()):
        return None
    slots = plan_cache_slots(specs)
    padded = {
        p.name: replace(specs[p.name], page_size_padded=p.page_size_bytes) for slot in slots for p in slot.placements
    }
    full = {n: s for n, s in padded.items() if isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))}
    state = {n: s for n, s in padded.items() if isinstance(s, DeepseekV41CompressorStateSpec)}
    groups = [_uniform(full, "full"), _uniform(state, "state")]
    swa = sorted((n for n, s in padded.items() if isinstance(s, DeepseekV41SWASpec)), key=_layer_number)
    groups.extend(
        _uniform({n: padded[n] for n in swa[start : start + len(slots)]}, f"swa{start}")
        for start in range(0, len(swa), len(slots))
    )
    draft = sorted((n for n, s in padded.items() if isinstance(s, DeepseekV41DraftSWASpec)), key=_draft_layer_number)
    if draft:
        groups.append(_uniform({n: padded[n] for n in draft}, "dspark"))
    return groups


def make_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


def uses_deepseek_v41_cache_layout(groups):
    return any(
        is_deepseek_v41_cache_spec(s)
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        for s in g.kv_cache_spec.kv_cache_specs.values()
    )


def cache_slots_from_groups(groups):
    specs = {}
    for group in groups:
        for name in group.layer_names:
            specs[name] = group.kv_cache_spec.kv_cache_specs[name]
    return plan_cache_slots(specs)


def pool_bytes_per_block(groups):
    return sum(slot.page_size_bytes for slot in cache_slots_from_groups(groups))


def request_blocks(vllm_config, groups):
    # Different logical groups consume different IDs in one global block pool.
    total = 0
    for group in groups:
        spec = group.kv_cache_spec
        # The scheduler replaces uniform groups with a representative layer spec.
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        total += max(
            (s.max_memory_usage_bytes(vllm_config) + s.page_size_bytes - 1) // s.page_size_bytes for s in specs
        )
    return total


def allocate_cache_config(vllm_config, groups, available_memory):
    """Allocate four independent layer slots backed by one global block-ID pool."""
    slots = cache_slots_from_groups(groups)
    capacity = available_memory // sum(slot.page_size_bytes for slot in slots)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    tensors = []
    for slot in slots:
        layer_names = [placement.name for placement in slot.placements]
        size = num_blocks * slot.page_size_bytes
        if vllm_version_is("0.28.0"):
            tensors.append(
                KVCacheTensor(
                    size=size,
                    shared_by=layer_names,
                    block_stride=slot.page_size_bytes,
                )
            )
        else:
            tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=layer_names,
                    offset=0,
                    layer_stride=0,
                    block_stride=slot.page_size_bytes,
                )
            )
    return num_blocks, tensors


def reshape_cache(raw: torch.Tensor, spec, *, num_blocks, offset, block_stride):
    """Create typed per-page views using the containing slot's physical stride."""
    plane_sizes = _cache_plane_sizes(spec)
    storage_block_size = get_storage_block_size(spec)

    def view(dtype, width, byte_offset):
        dtype_size = dtype.itemsize
        storage_offset = raw.storage_offset() + byte_offset
        return torch.as_strided(
            raw.view(dtype),
            size=(num_blocks, storage_block_size, spec.num_kv_heads, width),
            stride=(block_stride // dtype_size, spec.num_kv_heads * width, width, 1),
            storage_offset=storage_offset // dtype_size,
        )

    key = view(spec.dtype, spec.head_size, offset)
    if isinstance(spec, DeepseekV41IndexerSpec):
        return key, view(spec.scale_dtype, spec.scale_dim, offset + plane_sizes[0])
    return key


def validate_cache_runtime(vllm_config):
    if vllm_config.speculative_config is not None:
        # DeepSeek V4.1's planes are always BF16. Pin the inherited DSV4 draft
        # backend to the same layout, including on hardware where auto is FP8.
        vllm_config.cache_config.cache_dtype = "bfloat16"


class DeepseekV41CacheLayout:
    """Keep grouping, allocation and admission on the same layer-slot layout."""

    include_private_groups_in_block_alignment = True
    group_specs = staticmethod(group_cache_specs)
    make_groups = staticmethod(make_cache_groups)
    allocate = staticmethod(allocate_cache_config)
    pool_bytes_per_block = staticmethod(pool_bytes_per_block)

    @staticmethod
    def max_memory_usage(vllm_config, groups):
        return (request_blocks(vllm_config, groups) + 1) * pool_bytes_per_block(groups)

    @staticmethod
    def max_concurrency(vllm_config, cache_config):
        return max(0, cache_config.num_blocks - 1) / request_blocks(vllm_config, cache_config.kv_cache_groups)


_CACHE_LAYOUT = DeepseekV41CacheLayout()
