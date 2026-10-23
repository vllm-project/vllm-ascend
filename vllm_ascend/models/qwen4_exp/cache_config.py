# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp: semantic groups, four LBNHC rows, two backing allocations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import NamedTuple

import torch
from vllm.config import VllmConfig
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
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
from vllm.v1.kv_cache_layout import KVCacheLayout

CACHE_ALIGNMENT = 16
TRANSFER_ALIGNMENT = 2 * 1024 * 1024
GDN_COUNT = 3
VALUE_ROW, KEY_ROW, CONV_ROW, INDEXER_ROW = range(4)


class LayerTuple(NamedTuple):
    attention: str
    indexer: str
    gdn: tuple[str, ...]
    ring: str


def is_qwen4_exp_cache(specs_or_groups: dict[str, KVCacheSpec] | list[KVCacheGroupSpec]) -> bool:
    items = specs_or_groups.values() if isinstance(specs_or_groups, dict) else specs_or_groups
    for item in items:
        spec = getattr(item, "kv_cache_spec", item)
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if any(getattr(member, "model_version", None) == "qwen4_exp" for member in specs):
            return True
    return False


def _layer_index(name: str) -> int:
    try:
        return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot determine Qwen4Exp layer index from {name}") from error


def _component_specs(spec: KVCacheSpec) -> tuple[tuple[torch.dtype, tuple[int, ...]], ...]:
    """Model binding order: K/V, Conv/SSM, or a single state."""
    if isinstance(spec, MambaSpec):
        if len(spec.shapes) != len(spec.dtypes) or len(spec.shapes) not in (1, 2):
            raise ValueError("Qwen4Exp Mamba cache requires one or two typed states")
        return tuple(zip(spec.dtypes, spec.shapes, strict=True))
    if type(spec) is FullAttentionSpec:
        if spec.tokens_per_state != 1 or spec.state_content_bytes is not None or spec.kv_quant_mode:
            raise ValueError("Qwen4Exp main cache requires dense, unquantized K/V")
        return (
            (spec.dtype, (spec.num_states, spec.num_kv_heads, spec.head_size)),
            (spec.dtype, (spec.num_states, spec.num_kv_heads, spec.head_size_v)),
        )
    if isinstance(spec, (MLAAttentionSpec, CircularBufferSpec, HiddenStateCacheSpec)):
        return ((spec.dtype, (spec.num_kv_heads, spec.num_states, spec.head_size)),)
    raise ValueError(f"Unsupported Qwen4Exp cache spec: {type(spec).__name__}")


def build_layer_tuples(
    specs: dict[str, KVCacheSpec], layer_types: tuple[str, ...]
) -> tuple[tuple[LayerTuple, ...], str, tuple[str, ...]]:
    """Build deterministic cycles from original model indices, never dict order."""
    period = GDN_COUNT + 1
    if not layer_types or len(layer_types) % period:
        raise ValueError("Qwen4Exp requires complete cycles of one QSA and three GDN layers")
    by_index: dict[int, list[str]] = {}
    ple_layer: str | None = None
    hidden_layers = []
    draft_by_prefix: dict[str, list[str]] = {}
    for name, spec in specs.items():
        _component_specs(spec)
        if isinstance(spec, HiddenStateCacheSpec):
            hidden_layers.append(name)
        elif isinstance(spec, MambaSpec) and len(spec.shapes) == 1:
            if ple_layer is not None:
                raise ValueError("Qwen4Exp requires exactly one PLE cache layer")
            ple_layer = name
        else:
            index = _layer_index(name)
            if ".mtp." in name or index >= len(layer_types):
                prefix = name.split(".indexer.", 1)[0].removesuffix(".attn")
                draft_by_prefix.setdefault(prefix, []).append(name)
            else:
                by_index.setdefault(index, []).append(name)
    if ple_layer is None:
        raise ValueError("Qwen4Exp requires exactly one PLE cache layer")

    cycle_layers = []
    for start in range(0, len(layer_types), period):
        qsa_indices = [
            i for i in range(start, start + period) if layer_types[i] in ("full_attention", "qwen_sparse_attention")
        ]
        gdn_indices = [i for i in range(start, start + period) if layer_types[i] == "linear_attention"]
        if len(qsa_indices) != 1 or len(gdn_indices) != GDN_COUNT:
            raise ValueError("Qwen4Exp cycle must contain one QSA and three GDN layers")
        gdn = []
        for index in gdn_indices:
            names = by_index.get(index, [])
            if len(names) != 1 or not isinstance(specs[names[0]], MambaSpec):
                raise ValueError(f"Missing or duplicate GDN layer at model index {index}")
            gdn.append(names[0])
        cycle_layers.append((by_index.get(qsa_indices[0], []), tuple(gdn)))
    # Draft QSA layers keep separate groups; their tuples have no GDN members.
    cycle_layers.extend(
        (draft_by_prefix[prefix], ()) for prefix in sorted(draft_by_prefix, key=lambda name: (_layer_index(name), name))
    )
    layer_tuples = []
    for names, gdn in cycle_layers:
        attention = [name for name in names if name.endswith(".attn") and type(specs[name]) is FullAttentionSpec]
        if len(attention) != 1:
            raise ValueError(f"Missing or duplicate QSA Attention: {names}")
        source = attention[0].removesuffix(".attn")
        indexer, ring = source + ".indexer.compressed_key_cache", source + ".indexer.raw_key_cache"
        if set(names) != {attention[0], indexer, ring}:
            raise ValueError(f"QSA Attention/Indexer/RingBuffer must match per layer: {source}")
        main, compressed, raw = (specs[name] for name in (attention[0], indexer, ring))
        if not isinstance(compressed, MLAAttentionSpec) or not isinstance(raw, CircularBufferSpec):
            raise ValueError(f"Invalid QSA Indexer/RingBuffer specs: {source}")
        ratio = compressed.tokens_per_state
        if (
            not isinstance(ratio, int)
            or ratio <= 1
            or main.block_size != compressed.block_size
            or main.block_size % ratio
            or raw.block_size <= 0
            or main.block_size % raw.block_size
        ):
            raise ValueError(f"Incompatible QSA token block coverage: {source}")
        layer_tuples.append(LayerTuple(attention[0], indexer, gdn, ring))
    topology = tuple(layer_tuples)
    assigned_layers = [name for item in topology for name in (item.attention, item.indexer, *item.gdn, item.ring)]
    assigned_layers += [ple_layer, *hidden_layers]
    if len(assigned_layers) != len(set(assigned_layers)) or set(assigned_layers) != set(specs):
        raise ValueError("Qwen4Exp layer tuples contain duplicate or missing cache layers")

    return topology, ple_layer, tuple(sorted(hidden_layers))


def get_qwen4_exp_kv_cache_groups(vllm_config: VllmConfig, specs: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    topology, ple_layer, hidden_layers = build_layer_tuples(
        specs, tuple(vllm_config.model_config.hf_text_config.layer_types)
    )
    main_tuples = [item for item in topology if item.gdn]
    draft_tuples = [item for item in topology if not item.gdn]
    group_layers = [
        ([name for item in main_tuples for name in (item.attention, item.indexer)], False, True),
        ([name for item in main_tuples for name in item.gdn] + [ple_layer], False, True),
        ([item.ring for item in main_tuples], False, False),
    ]
    if draft_tuples:
        group_layers += [
            ([name for item in draft_tuples for name in (item.attention, item.indexer)], True, True),
            ([item.ring for item in draft_tuples], True, False),
        ]
    group_layers.extend(([name], False, True) for name in sorted(hidden_layers))
    groups = []
    for names, is_draft, enable_kv_transfer in group_layers:
        members = {name: specs[name] for name in names}
        first = members[names[0]]
        # These fields determine checkpoint position, restore/recycle policy and
        # block-table demand. Backend, TP replication and payload shape do not.
        scheduling_fields = (
            "block_size",
            "prefix_cacheable",
            "prefix_replay_tokens",
            "mamba_cache_mode",
            "num_speculative_blocks",
            "num_prefill_checkpoint_blocks",
            "prefill_checkpoint_alignment",
            "sliding_window",
            "attention_chunk_size",
        )
        for name, spec in members.items():
            if any(getattr(spec, key, None) != getattr(first, key, None) for key in scheduling_fields):
                raise ValueError(f"Incompatible Qwen4Exp cache lifetimes: {name}")
        if isinstance(first, MambaSpec):
            if any(type(spec) is not MambaSpec for spec in members.values()):
                raise ValueError("Qwen4Exp GDN/PLE requires known Mamba scheduling semantics")
            # Deliberately local: do not loosen upstream uniformity or modify TP.
            uniform = UniformTypeKVCacheSpecs(block_size=first.block_size, kv_cache_specs=members)
        else:
            uniform = UniformTypeKVCacheSpecs.from_specs(members)
            if uniform is None:
                raise ValueError("Incompatible Qwen4Exp attention cache group")
        groups.append(
            KVCacheGroupSpec(
                layer_names=names,
                kv_cache_spec=uniform,
                is_eagle_group=is_draft,
                enable_kv_transfer=enable_kv_transfer,
            )
        )
    return groups


def _get_topology(groups: list[KVCacheGroupSpec]) -> tuple[tuple[LayerTuple, ...], str, tuple[str, ...]]:
    # PP retains each ordinary group's global member specs. Only layer_names
    # is projected; original model indices still define all cycle positions.
    specs = {name: spec for group in groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}
    main = {
        _layer_index(name): "linear_attention" if isinstance(spec, MambaSpec) else "qwen_sparse_attention"
        for group in groups
        if not group.is_eagle_group
        for name, spec in group.kv_cache_spec.kv_cache_specs.items()
        if not isinstance(spec, HiddenStateCacheSpec) and not (isinstance(spec, MambaSpec) and len(spec.shapes) == 1)
    }
    layer_types = tuple(main[index] for index in range(max(main) + 1))
    return build_layer_tuples(specs, layer_types)


def _iter_row_slots(groups: list[KVCacheGroupSpec]):
    """Yield (row, slot, layer, state index) in model binding order.

    Groups overlay their slots: attention and ring use one per cycle; GDN
    uses three. PLE occupies the third column of the first state cycle once.
    Draft groups have independent block ownership and restart at slot zero.
    This iterator is the sole layer-to-slot mapping, not an address table.
    """
    topology, ple, hidden = _get_topology(groups)
    local = _get_layer_specs(groups)
    main_slot = draft_slot = 0
    for item in topology:
        slot = main_slot if item.gdn else draft_slot
        for row, name, component in (
            (KEY_ROW, item.attention, 0),
            (VALUE_ROW, item.attention, 1),
            (INDEXER_ROW, item.indexer, 0),
            (VALUE_ROW, item.ring, 0),
        ):
            if name in local:
                yield row, slot, name, component
        for column, name in enumerate(item.gdn):
            if name in local:
                yield CONV_ROW, slot * GDN_COUNT + column, name, 0
                yield KEY_ROW, slot * GDN_COUNT + column, name, 1
        if item.gdn:
            main_slot += 1
        else:
            draft_slot += 1
    if ple in local:
        yield VALUE_ROW, GDN_COUNT - 1, ple, 0
    for row, name in enumerate(hidden, INDEXER_ROW + 1):
        if name in local:
            yield row, 0, name, 0


def _get_cache_layout(groups: list[KVCacheGroupSpec]) -> tuple[tuple[int, int], ...]:
    """Return (common page bytes, physical slots) per row, including extras."""
    specs = _get_layer_specs(groups)
    pages, slots = [CACHE_ALIGNMENT] * 4, [0] * 4
    for row, slot, name, component in _iter_row_slots(groups):
        while row >= len(pages):
            pages.append(CACHE_ALIGNMENT)
            slots.append(0)
        dtype, shape = _component_specs(specs[name])[component]
        pages[row] = max(pages[row], round_up(math.prod(shape) * get_dtype_size(dtype), CACHE_ALIGNMENT))
        slots[row] = max(slots[row], slot + 1)
    return tuple(zip(pages, slots, strict=True))


def _get_layer_specs(groups: list[KVCacheGroupSpec]) -> dict[str, KVCacheSpec]:
    specs = {}
    for group in groups:
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            raise ValueError("Worker Qwen4Exp groups must retain per-layer specs")
        for name in group.layer_names:
            if name in specs:
                raise ValueError(f"Duplicate Qwen4Exp cache layer: {name}")
            specs[name] = group.kv_cache_spec.kv_cache_specs[name]
    return specs


def project_qwen4_exp_cache_groups(
    groups: list[KVCacheGroupSpec], worker_specs: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
    """Keep global group IDs/topology, but compute storage from local specs."""
    if not worker_specs:
        return []
    global_specs = _get_layer_specs(groups)
    if not worker_specs.keys() <= global_specs.keys():
        raise ValueError("Unknown Qwen4Exp worker cache layers")
    projected = []
    for group in groups:
        names = [name for name in group.layer_names if name in worker_specs]
        # Empty groups retain their representative for global scheduler IDs.
        spec = group.kv_cache_spec
        if names:
            spec = replace(spec, kv_cache_specs={**spec.kv_cache_specs, **{name: worker_specs[name] for name in names}})
        projected.append(replace(group, layer_names=names, kv_cache_spec=spec))
    return projected


def get_qwen4_exp_pool_bytes_per_block(groups: list[KVCacheGroupSpec]) -> int:
    return sum(page * slots for page, slots in _get_cache_layout(groups))


def _allocation_overhead(vllm_config: VllmConfig, rows: tuple[tuple[int, int], ...]) -> int:
    if getattr(vllm_config, "kv_transfer_config", None) is None:
        return 0
    slots = [count for _, count in rows]
    return TRANSFER_ALIGNMENT * (int(any(slots[:3])) + int(bool(slots[3])) + sum(bool(s) for s in slots[4:]))


def get_qwen4_exp_allocation_overhead(vllm_config: VllmConfig, groups: list[KVCacheGroupSpec]) -> int:
    return _allocation_overhead(vllm_config, _get_cache_layout(groups))


def get_qwen4_exp_kv_cache_configs(
    vllm_config: VllmConfig,
    worker_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """Preserve fixed allocation overhead when equalizing worker block counts.

    Upstream re-plans with ``blocks * page_bytes``, which omits the two
    backing allocations' alignment reserve. Reuse its admission and replica
    helpers, but include that reserve at both initial planning and re-planning.
    """
    specs: dict[str, KVCacheSpec] = {}
    for local_specs in worker_specs:
        for name, spec in local_specs.items():
            if name in specs and specs[name] != spec:
                raise ValueError(f"Inconsistent Qwen4Exp cache spec across workers: {name}")
            specs[name] = spec
    kv_cache_utils.KVCacheSpecRegistry.check_kv_cache_spec_registry(specs)
    groups = get_qwen4_exp_kv_cache_groups(vllm_config, specs)
    projected = [project_qwen4_exp_cache_groups(groups, local) for local in worker_specs]
    pages = [get_qwen4_exp_pool_bytes_per_block(local) for local in projected]
    overheads = [get_qwen4_exp_allocation_overhead(vllm_config, local) for local in projected]
    override = vllm_config.cache_config.num_gpu_blocks_override
    budgets = (
        [override * page + overhead for page, overhead in zip(pages, overheads, strict=True)]
        if override is not None
        else available_memory
    )
    # Upstream admission excludes the permanently reserved null block.
    check_memory = [budget - page for budget, page in zip(budgets, pages, strict=True)]
    if vllm_config.model_config.original_max_model_len == -1:
        kv_cache_utils._auto_fit_max_model_len(vllm_config, projected, check_memory)
    for local, budget in zip(projected, check_memory, strict=True):
        kv_cache_utils._check_enough_kv_cache_memory(
            budget,
            partial(kv_cache_utils._max_memory_usage_bytes_from_groups, vllm_config, local),
            vllm_config.model_config.max_model_len,
            partial(kv_cache_utils._estimate_max_model_len_from_groups, vllm_config, local),
        )
    configs = [
        kv_cache_utils.get_kv_cache_config_from_groups(vllm_config, local, budget)
        for local, budget in zip(projected, budgets, strict=True)
    ]
    num_blocks = min(config.num_blocks for config in configs)
    for i, config in enumerate(configs):
        if config.num_blocks != num_blocks:
            configs[i] = kv_cache_utils.get_kv_cache_config_from_groups(
                vllm_config, projected[i], num_blocks * pages[i] + overheads[i]
            )
        configs[i].kv_tp_replicas = kv_cache_utils.kv_cache_groups_tp_replicas(
            projected[i],
            vllm_config.parallel_config.tensor_parallel_size,
            vllm_config.parallel_config.decode_context_parallel_size,
        )
    return configs


def get_qwen4_exp_kv_cache_config(
    vllm_config: VllmConfig, groups: list[KVCacheGroupSpec], available_memory: int
) -> KVCacheConfig:
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    if layout != KVCacheLayout.LBNHC:
        raise ValueError(f"Qwen4Exp packed cache requires LBNHC layout, got {layout.name}")
    rows = _get_cache_layout(groups)
    overhead = _allocation_overhead(vllm_config, rows)
    num_blocks = may_override_num_blocks(
        vllm_config, (available_memory - overhead) // sum(page * slots for page, slots in rows)
    )
    if num_blocks <= 0:
        raise ValueError("Insufficient memory for one Qwen4Exp packed cache block")
    tensors = _make_descriptors(groups, rows, num_blocks)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        kv_cache_layout=layout.name,
    )


def _make_descriptors(groups, rows, num_blocks) -> list[KVCacheTensor]:
    sizes = [num_blocks * page * slots for page, slots in rows]
    names = [[] for _ in rows]
    for row, _, name, _ in _iter_row_slots(groups):
        names[row].append(name)
    return [
        KVCacheTensor(
            size=sum(sizes[:3]) if row < INDEXER_ROW else sizes[row],
            offset=sum(sizes[:row]) if row < INDEXER_ROW else 0,
            layers=names[row],
            layer_stride=num_blocks * page,
            block_stride=page,
        )
        for row, (page, _) in enumerate(rows)
    ]


def split_tensor(backing: torch.Tensor, sizes: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """Split a contiguous byte backing into aligned, disjoint storage views."""
    if backing.ndim != 1 or backing.element_size() != 1 or not backing.is_contiguous():
        raise ValueError("Qwen4Exp requires a contiguous byte backing")
    if (
        not sizes
        or any(size < 0 or size % CACHE_ALIGNMENT for size in sizes)
        or sum(sizes) != backing.numel()
        or backing.data_ptr() % CACHE_ALIGNMENT
    ):
        raise ValueError("Qwen4Exp split sizes, boundaries or alignment are invalid")
    return backing.split(sizes)


def allocate_qwen4_exp_kv_cache_tensors(
    config: KVCacheConfig, allocator: Callable[[int, int], torch.Tensor]
) -> dict[str, tuple[torch.Tensor, ...]]:
    groups = config.kv_cache_groups
    specs = _get_layer_specs(groups)
    geometry = _get_cache_layout(groups)
    descriptors = config.kv_cache_tensors
    if config.kv_cache_layout != KVCacheLayout.LBNHC.name:
        raise ValueError("Qwen4Exp allocation requires LBNHC")
    if descriptors != _make_descriptors(groups, geometry, config.num_blocks):
        raise ValueError("Qwen4Exp descriptor size, offset, stride or layers do not match layout")
    row_sizes = tuple(config.num_blocks * page * slots for page, slots in geometry)
    rows = []
    storage_keys = set()
    # Exactly two allocations for the main model. Empty PP backings need none;
    # hidden-state caches each have their own allocation and lifecycle.
    for first, stop in ((0, 3), (3, 4), *((r, r + 1) for r in range(4, len(geometry)))):
        size = descriptors[first].size
        if not size:
            rows.extend([None] * (stop - first))
            continue
        backing = allocator(size, TRANSFER_ALIGNMENT)
        storage_key = backing.untyped_storage().data_ptr()
        if storage_key in storage_keys:
            raise ValueError("Qwen4Exp backings must be independent allocations")
        storage_keys.add(storage_key)
        rows.extend(split_tensor(backing, row_sizes[first:stop]))
    caches = {name: [None] * len(_component_specs(spec)) for name, spec in specs.items()}
    for row_id, slot, name, component in _iter_row_slots(groups):
        row = rows[row_id]
        descriptor = descriptors[row_id]
        dtype, shape = _component_specs(specs[name])[component]
        payload = math.prod(shape) * get_dtype_size(dtype)
        # storage_offset is absolute within storage, including allocator slack.
        caches[name][component] = row.as_strided(
            (config.num_blocks, payload),
            (descriptor.block_stride, 1),
            row.storage_offset() + slot * descriptor.layer_stride,
        )
    return {name: tuple(components) for name, components in caches.items()}


def reshape_qwen4_exp_kv_cache_tensors(
    config: KVCacheConfig, raw_caches: dict[str, tuple[torch.Tensor, ...]]
) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    caches = {}
    specs = _get_layer_specs(config.kv_cache_groups)
    for name, raw_components in raw_caches.items():
        spec = specs[name]
        views = []
        for raw, (dtype, shape) in zip(raw_components, _component_specs(spec), strict=True):
            # The per-block payload is contiguous; flatten/unflatten only its
            # inner dimensions. The outer row stride and storage are unchanged.
            views.append(raw.view(dtype).unflatten(1, shape))
        caches[name] = tuple(views) if isinstance(spec, MambaSpec) or type(spec) is FullAttentionSpec else views[0]
    return caches
