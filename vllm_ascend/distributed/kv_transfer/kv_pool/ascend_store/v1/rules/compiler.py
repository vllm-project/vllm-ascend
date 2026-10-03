"""Compile static AscendStore facts into directly callable KV rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import cast

import numpy as np

from ..backend import LayerwiseAccessKind, resolve_backend_spec
from ..topology import KVPoolTopology
from .identity import (
    BlockRows,
    KeyAxes,
    LayerwiseFullKey,
    LayerwisePartialKey,
    bind_checkpoint_rule,
    bind_chunk_rules,
    bind_key_rules,
    bind_store_ownership,
    resolve_store_pipeline_ranks,
)
from .memory import (
    BulkRangeBatch,
    KVMemoryRule,
    MemoryDataPlane,
    RangeBatch,
    bulk_arguments,
    gva_arguments,
    key_range_arguments,
    required_object_sizes,
)
from .reachability import HybridReachability, UnitaryReachability

RuleBinder = Callable[..., "KVPoolRules"]


@dataclass(frozen=True, slots=True)
class KVPoolRuleSpec:
    """Resolved configuration that selects one callable KV rule family."""

    topology: KVPoolTopology
    backend_name: str
    max_model_len: int
    use_layerwise: bool = False
    use_eagle: bool = False
    retention_interval: int | None = None


class KVPoolRules:
    """Configuration-selected callables plus one shared memory rule."""

    __slots__ = (
        "admit_store",
        "checkpoint_rows",
        "format_ranges",
        "group_ids",
        "load_keys",
        "load_rows",
        "load_selection",
        "lookup_chunks",
        "lookup_keys",
        "memory",
        "object_size",
        "partial_key",
        "resolve_lookup",
        "required_object_sizes",
        "requires_store_observation",
        "store_keys",
        "store_rows",
        "store_selection",
        "lookup_selection",
    )

    def __init__(
        self,
        *,
        group_ids: tuple[int, ...],
        lookup_chunks: Callable,
        lookup_selection: Callable,
        resolve_lookup: Callable,
        load_rows: Callable,
        load_selection: Callable,
        store_rows: Callable,
        store_selection: Callable,
        checkpoint_rows: Callable | None,
        lookup_keys: Callable,
        load_keys: Callable,
        store_keys: Callable,
        partial_key: Callable | None,
        requires_store_observation: bool,
        admit_store: Callable,
        memory: KVMemoryRule,
        object_size: Callable,
        format_ranges: Callable,
        required_object_sizes_rule: Callable | None,
    ) -> None:
        self.group_ids = group_ids
        self.lookup_chunks = lookup_chunks
        self.lookup_selection = lookup_selection
        self.resolve_lookup = resolve_lookup
        self.load_rows = load_rows
        self.load_selection = load_selection
        self.store_rows = store_rows
        self.store_selection = store_selection
        self.checkpoint_rows = checkpoint_rows
        self.lookup_keys = lookup_keys
        self.load_keys = load_keys
        self.store_keys = store_keys
        self.partial_key = partial_key
        self.requires_store_observation = requires_store_observation
        self.admit_store = admit_store
        self.memory = memory
        self.object_size = object_size
        self.format_ranges = format_ranges
        self.required_object_sizes = required_object_sizes_rule


def compile_kv_pool_rules(
    spec: KVPoolRuleSpec,
    *,
    layerwise_full_key: LayerwiseFullKey | None = None,
    layerwise_partial_key: LayerwisePartialKey | None = None,
) -> RuleBinder:
    """Select configuration-dependent rules and return the memory binder.

    The returned callable is the only intermediate value.  Once registered
    memory is supplied it returns a final ``KVPoolRules`` object; there is no
    public unbound/bound object hierarchy.
    """

    topology = spec.topology
    groups = topology.transfer_groups
    align_state_group_ids = frozenset(group.group_id for group in groups if group.uses_align_state)
    _validate_static_rules(spec, align_state_group_ids, layerwise_full_key)

    _, lookup_chunks, load_rows = bind_chunk_rules(topology)
    if len(groups) == 1 and not align_state_group_ids:
        reachability: UnitaryReachability | HybridReachability = UnitaryReachability(
            topology.transfer_group_ids[0],
            spec.max_model_len,
            topology.cache_transfer_granularity,
        )
    else:
        reachability = HybridReachability(
            groups,
            scheduler_block_size=topology.cache_transfer_granularity,
            hash_block_size=topology.hash_block_size,
            max_model_len=spec.max_model_len,
            use_eagle=spec.use_eagle,
            retention_interval=spec.retention_interval,
        )
    store_pipeline_ranks = resolve_store_pipeline_ranks(topology)
    load_keys, store_keys, lookup_keys, partial_key = bind_key_rules(
        topology,
        layerwise_full_key,
        layerwise_partial_key,
        store_pipeline_ranks,
    )
    ownership = bind_store_ownership(topology)
    checkpoint_rows = bind_checkpoint_rule(topology, ownership)
    if spec.use_layerwise:
        store_rows = load_rows if _is_layerwise_store_leader(topology) else _discard_store_rows
    else:
        store_rows = partial(_store_rows, load_rows, ownership)

    backend = resolve_backend_spec(spec.backend_name)
    format_ranges: Callable
    memory_data_plane: MemoryDataPlane
    required_object_sizes_rule: Callable | None = None
    if not spec.use_layerwise:
        format_ranges = _bulk_arguments
        memory_data_plane = "bulk"
    elif backend.layerwise_access is LayerwiseAccessKind.GVA:
        format_ranges = _gva_arguments
        memory_data_plane = "gva"
        required_object_sizes_rule = required_object_sizes
    else:
        format_ranges = _key_range_arguments
        memory_data_plane = "key_range"

    memory_parameters = {
        "group_ids": topology.transfer_group_ids,
        "block_sizes": {group.group_id: group.block_size for group in groups},
        "align_state_group_ids": align_state_group_ids,
        "physical_layers": {
            group.group_id: tuple(layer.physical_layer_id for layer in group.layers) for group in groups
        },
        "strided_slice_count": (topology.tp_partition.key_slices_per_rank if topology.tp_partition.tp_mismatch else 1),
        "consumer_pipeline_partitions": topology.consumer_pipeline_partitions,
        "store_pipeline_ranks": store_pipeline_ranks,
        "data_plane": memory_data_plane,
        "requires_global_offsets": (topology.pp_size > 1 or (topology.dcp_size > 1 and topology.put_step > 1)),
    }
    admit_store = cast(Callable, _missing_objects if backend.requires_exists_before_put else _all_objects)
    return partial(
        _bind_rules,
        group_ids=topology.transfer_group_ids,
        lookup_chunks=lookup_chunks,
        lookup_selection=reachability.select_for_lookup,
        resolve_lookup=reachability.resolve_available_end,
        load_rows=load_rows,
        load_selection=reachability.select_for_load,
        store_rows=store_rows,
        store_selection=reachability.select_for_store,
        checkpoint_rows=checkpoint_rows,
        lookup_keys=lookup_keys,
        load_keys=load_keys,
        store_keys=store_keys,
        partial_key=partial_key,
        requires_store_observation=backend.requires_exists_before_put,
        admit_store=admit_store,
        format_ranges=format_ranges,
        required_object_sizes_rule=required_object_sizes_rule,
        memory_parameters=memory_parameters,
    )


def _bind_rules(
    base_addresses: Mapping[int, Sequence[int]],
    block_lengths: Mapping[int, Sequence[int]],
    block_strides: Mapping[int, Sequence[int]],
    layer_entry_offsets: Mapping[int, Sequence[int]],
    *,
    object_sizes: Mapping[int, int] | None = None,
    object_offsets: Mapping[int, int] | None = None,
    group_ids: tuple[int, ...],
    lookup_chunks: Callable,
    lookup_selection: Callable,
    resolve_lookup: Callable,
    load_rows: Callable,
    load_selection: Callable,
    store_rows: Callable,
    store_selection: Callable,
    checkpoint_rows: Callable | None,
    lookup_keys: Callable,
    load_keys: Callable,
    store_keys: Callable,
    partial_key: Callable | None,
    requires_store_observation: bool,
    admit_store: Callable,
    format_ranges: Callable,
    required_object_sizes_rule: Callable | None,
    memory_parameters: dict,
) -> KVPoolRules:
    memory = KVMemoryRule(
        base_addresses=base_addresses,
        block_lengths=block_lengths,
        block_strides=block_strides,
        layer_entry_offsets=layer_entry_offsets,
        object_sizes=object_sizes,
        object_offsets=object_offsets,
        **memory_parameters,
    )
    data_plane = memory_parameters["data_plane"]
    object_size_by_group = {
        group_id: (
            int(object_sizes[group_id])
            if data_plane == "gva" and object_sizes is not None
            else int(sum(block_lengths[group_id]))
        )
        for group_id in group_ids
    }
    return KVPoolRules(
        group_ids=group_ids,
        lookup_chunks=lookup_chunks,
        lookup_selection=lookup_selection,
        resolve_lookup=resolve_lookup,
        load_rows=load_rows,
        load_selection=load_selection,
        store_rows=store_rows,
        store_selection=store_selection,
        checkpoint_rows=checkpoint_rows,
        lookup_keys=lookup_keys,
        load_keys=load_keys,
        store_keys=store_keys,
        partial_key=partial_key,
        requires_store_observation=requires_store_observation,
        admit_store=admit_store,
        memory=memory,
        object_size=partial(_object_size, object_size_by_group),
        format_ranges=format_ranges,
        required_object_sizes_rule=required_object_sizes_rule,
    )


def _object_size(sizes: Mapping[int, int], group_id: int) -> int:
    return sizes[group_id]


def _store_rows(
    block_rows: Callable,
    ownership: Callable,
    group_id: int,
    end_token: int,
    block_hashes,
    block_ids,
    *,
    start_token: int = 0,
    mask=None,
) -> BlockRows:
    rows = block_rows(
        group_id,
        end_token,
        block_hashes,
        block_ids,
        start_token=start_token,
        mask=mask,
    )
    return ownership(group_id, rows)


def _discard_store_rows(
    group_id: int,
    end_token: int,
    block_hashes,
    block_ids,
    *,
    start_token: int = 0,
    mask=None,
) -> BlockRows:
    del group_id, end_token, block_hashes, block_ids, start_token, mask
    empty: np.ndarray = np.empty(0, dtype=np.uint64)
    empty.flags.writeable = False
    return empty, empty, (), empty


def _is_layerwise_store_leader(topology: KVPoolTopology) -> bool:
    if topology.dcp_size <= 1:
        return topology.tp_rank % topology.put_step == 0
    dcp_rank = topology.transfer_groups[0].key_metadata.dcp_rank
    head = topology.tp_rank // topology.put_step
    peers = tuple(
        rank
        for rank in range(head * topology.put_step, (head + 1) * topology.put_step)
        if rank % topology.dcp_size == dcp_rank
    )
    return not peers or topology.tp_rank == min(peers)


def _all_objects(exists: Sequence[bool] | None = None):
    del exists
    return None


def _missing_objects(exists: Sequence[bool]) -> np.ndarray:
    return ~np.asarray(exists, dtype=np.bool_)


def _bulk_arguments(
    key_axes: KeyAxes,
    ranges: BulkRangeBatch,
    *,
    object_bases=None,
    selected_objects: Sequence[bool] | None = None,
):
    del object_bases
    return bulk_arguments(key_axes, ranges, selected_objects)


def _gva_arguments(
    key_axes: KeyAxes,
    ranges: RangeBatch,
    *,
    object_bases,
    selected_objects: Sequence[bool] | None = None,
):
    del key_axes
    return gva_arguments(ranges, object_bases, selected_objects)


def _key_range_arguments(
    key_axes: KeyAxes,
    ranges: RangeBatch,
    *,
    object_bases=None,
    selected_objects: Sequence[bool] | None = None,
):
    del object_bases
    return key_range_arguments(key_axes, ranges, selected_objects)


def _validate_static_rules(
    spec: KVPoolRuleSpec,
    align_state_group_ids: frozenset[int],
    layerwise_full_key: LayerwiseFullKey | None,
) -> None:
    topology = spec.topology
    groups = topology.transfer_groups
    if topology.tp_partition.tp_mismatch and len(groups) != 1:
        raise ValueError("TP-mismatched transfer requires exactly one transferable KV cache group")
    if topology.tp_partition.tp_mismatch and align_state_group_ids:
        raise ValueError("Mamba align-state transfer cannot be composed with TP mismatch")
    if topology.tp_partition.tp_mismatch and spec.use_layerwise:
        raise ValueError("Layerwise transfer cannot be composed with TP mismatch")
    partitions = topology.consumer_pipeline_partitions
    if topology.tp_partition.tp_mismatch and partitions is not None and len(partitions) > 1:
        raise ValueError("Consumer pipeline projection cannot be composed with TP mismatch")
    if spec.use_layerwise and partitions is not None and len(partitions) > 1:
        raise ValueError("Layerwise transfer cannot be composed with consumer pipeline projection")
    if spec.use_layerwise and align_state_group_ids:
        raise ValueError("Layerwise transfer does not support Mamba align-state groups")
    if spec.use_layerwise and layerwise_full_key is None:
        raise ValueError("Layerwise rules require the Backend-bound full-key function")
