"""Evaluate bound rules into the argument shape required by each Backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ...rules import KVPoolRules
from ..batch import KVGroupBatch, KVTransferBatch, TransferSource


@dataclass(frozen=True, slots=True)
class RuleRanges:
    """Key-aligned local ranges evaluated for one Runtime submission."""

    keys: list[str]
    addresses: list[list[int]]
    sizes: list[list[int]]
    offsets: list[list[int]]
    sources: tuple[TransferSource, ...]


@dataclass(frozen=True, slots=True)
class RuleGVA:
    """Flattened global and local ranges evaluated for one GVA copy."""

    remote_addresses: np.ndarray
    local_addresses: np.ndarray
    sizes: np.ndarray
    sources: tuple[TransferSource, ...]


def materialize_rule_ranges(
    rules: KVPoolRules,
    batch: KVTransferBatch,
    *,
    layer_id: int | None,
    store: bool,
) -> RuleRanges:
    """Evaluate only the current request rows and execution fence."""

    keys: list[str] = []
    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    sources: list[TransferSource] = []
    memory_rule = rules.memory.store_partial if store else rules.memory.partial
    for group in batch.groups:
        ranges = memory_rule(
            group.group_id,
            group.block_ids,
            group.token_counts,
            layer_id=layer_id,
        )
        materialized = rules.format_ranges(
            group.key_axes,
            ranges,
            object_bases=None,
            selected_objects=group.selection,
        )
        group_keys, group_addresses, group_sizes, *group_offsets = materialized
        keys.extend(group_keys)
        addresses.extend(group_addresses)
        sizes.extend(group_sizes)
        offsets.extend(group_offsets[0] if group_offsets else ([] for _ in group_keys))
        sources.extend(_selected_sources(group, layer_id))
    return RuleRanges(keys, addresses, sizes, offsets, tuple(sources))


def materialize_rule_gva(
    rules: KVPoolRules,
    batch: KVTransferBatch,
    sessions: Mapping[str, tuple[int, int] | None],
    resolved_bases: dict[KVGroupBatch, np.ndarray],
    *,
    layer_id: int,
    store: bool,
) -> RuleGVA:
    """Combine dynamic GVA bases with static offsets at the Backend boundary."""

    remote_parts = []
    local_parts = []
    size_parts = []
    sources: list[TransferSource] = []
    memory_rule = rules.memory.store_partial if store else rules.memory.partial
    for group in batch.groups:
        ranges = memory_rule(
            group.group_id,
            group.block_ids,
            group.token_counts,
            layer_id=layer_id,
        )
        selected_sources = _selected_sources(group, layer_id)
        object_bases = resolved_bases.get(group)
        if object_bases is None:
            bases = []
            for source in selected_sources:
                session = sessions.get(source.key)
                if session is None:
                    raise RuntimeError(f"GVA session for {source.key!r} is unavailable")
                base, object_size = session
                if object_size != group.object_size:
                    raise RuntimeError(f"GVA session for {source.key!r} has an unexpected object size")
                bases.append(base)
            object_bases = np.asarray(bases, dtype=np.uint64)
            object_bases.flags.writeable = False
            resolved_bases[group] = object_bases
        remote, local, sizes = rules.format_ranges(
            group.key_axes,
            ranges,
            object_bases=object_bases,
            selected_objects=group.selection,
        )
        remote_parts.append(remote)
        local_parts.append(local)
        size_parts.append(sizes)
        sources.extend(selected_sources)
    return RuleGVA(
        _concatenate_gva_parts(remote_parts),
        _concatenate_gva_parts(local_parts),
        _concatenate_gva_parts(size_parts),
        tuple(sources),
    )


def merge_rule_key_ranges(ranges: RuleRanges) -> RuleRanges:
    """Coalesce repeated object keys before one KeyRange Backend call."""

    if len(set(ranges.keys)) == len(ranges.keys):
        return ranges
    key_indices: dict[str, int] = {}
    keys: list[str] = []
    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    for key, row_addresses, row_sizes, row_offsets in zip(
        ranges.keys,
        ranges.addresses,
        ranges.sizes,
        ranges.offsets,
        strict=True,
    ):
        index = key_indices.get(key)
        if index is None:
            index = len(keys)
            key_indices[key] = index
            keys.append(key)
            addresses.append([])
            sizes.append([])
            offsets.append([])
        addresses[index].extend(row_addresses)
        sizes[index].extend(row_sizes)
        offsets[index].extend(row_offsets)
    return RuleRanges(keys, addresses, sizes, offsets, ranges.sources)


def _selected_sources(group: KVGroupBatch, layer_id: int | None) -> tuple[TransferSource, ...]:
    if group.selection is None:
        indices = range(group.object_count)
    else:
        indices = np.flatnonzero(group.selection).tolist()
    return tuple(group.source(index, layer_id) for index in indices)


def _concatenate_gva_parts(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)
