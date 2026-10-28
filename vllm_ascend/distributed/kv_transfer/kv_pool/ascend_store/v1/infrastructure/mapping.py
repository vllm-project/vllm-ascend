"""Bind one group's Key/Local tables and layer indices; build Remote layouts separately.

Segments follow layer/name/component/element order. Within one PP partition (DCP rank fixed):

    segment              0    1    2  |  3    4
    layer                [ layer A ]    [layer B]
    Local[head 0].bases   a0   a1   a2    a3   a4
    Local[head 1].bases   b0   b1   b2    b3   b4
    shared geometry      block_strides / lengths
    Layer[PP]            A -> [0:3]      B -> [3:5]
    Remote[PP].offsets   byte offset of each segment within the remote object

KV storage remains caller-owned and must outlive all bound addresses.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from itertools import accumulate
from typing import TYPE_CHECKING, TypeAlias, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import get_storage_block_size

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheSpec

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata

    from .configuration.topology import KVPoolGroupTopology, KVPoolLayerTopology, KVPoolTopology


_ALL_SEGMENTS = slice(None)

PartitionId = tuple[int, int, int]  # PP, actual DCP and head/TP partition ranks within one group binding.
UInt64Array: TypeAlias = NDArray[np.uint64]
KeyTable = dict[PartitionId, tuple[str, str]]
KeyBuilder: TypeAlias = Callable[["KeyMetadata"], tuple[str, str]]


class LocalSegmentData(TypedDict):
    """Block-zero addresses and byte geometry in segment order.

    Uniform strides/lengths are stored as singletons shared by all segments.
    Empty objects have empty arrays.
    """

    bases: UInt64Array
    block_strides: UInt64Array
    lengths: UInt64Array


class RemoteLayout(TypedDict):
    """Object size and byte offsets over the common flat Local segment order."""

    size_bytes: int
    offsets: UInt64Array


LocalTable = dict[PartitionId, LocalSegmentData]
LayerTable = dict[int, dict[int, slice]]  # PP rank -> physical layer -> segment slice.
RemoteTable = dict[int, RemoteLayout]  # One layout shared by all head partitions in a PP partition.


# ========================
# Common binding
# ========================


def build_mappings(
    topology: KVPoolTopology,
    kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
    *,
    group_id: int,
    kv_cache_layout: str = "NHD",
    key_builder: KeyBuilder | None = None,
    layerwise: bool = False,
) -> tuple[KeyTable, LocalTable, LayerTable | None]:
    """Bind local partitions from Tensor views without copying KV data.

    Return Key/Local tables and layer indices (None for Bulk). Treat layer indices as read-only.
    key_builder maps each partition's KeyMetadata to (prefix, suffix), defaulting to PoolKey affixes.
    """
    group = next((group for group in topology.transfer_groups if group.group_id == group_id), None)
    if group is None:
        raise KeyError(group_id)
    metadata = group.key_metadata
    head_partition_count = topology.tp_partition.key_slices_per_rank if topology.tp_partition.tp_mismatch else 1
    first_head_rank = metadata.head_or_tp_rank * head_partition_count
    # HND/LBHNC caches arrive as per-layer head-major views.
    head_axis = 1 if kv_cache_layout in ("HND", "LBHNC") else 2
    token_axis = 2 if head_axis == 1 else 1
    builder = _default_key_builder if key_builder is None else key_builder

    keys: KeyTable = {}
    local: LocalTable = {}
    layer_slices: LayerTable | None = {} if layerwise else None
    for pp_rank, layers in select_layers(group, topology.consumer_pipeline_partitions).items():
        # DCP tokens are already sliced upstream; PCP does not change local selection.
        partitions = tuple(
            (pp_rank, metadata.dcp_rank, first_head_rank + head_partition_index)
            for head_partition_index in range(head_partition_count)
        )
        for partition_id in partitions:
            keys[partition_id] = builder(
                replace(metadata, pp_rank=pp_rank, head_or_tp_rank=partition_id[2], kv_cache_group_id=group_id)
            )

        bases: list[list[int]] = [[] for _ in partitions]
        strides: list[int] = []
        lengths: list[int] = []
        pp_layer_slices: dict[int, slice] | None = {} if layerwise else None
        for layer in layers:
            layer_segment_start = len(lengths)
            for name in layer.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                cache = kv_caches[name]
                tensors = (cache,) if isinstance(cache, torch.Tensor) else cache
                for tensor in tensors:
                    if not tensor.numel():
                        continue
                    scale = _kernel_blocks_per_group(tensor, spec, token_axis)
                    blocks = tensor.unflatten(0, (-1, scale))
                    element_bytes = tensor.element_size()
                    stride = blocks.stride(0) * element_bytes
                    head_views = split_head_partitions(blocks[0], head_partition_count, head_axis)
                    # Equal head views share shape/strides, so their segments differ only by address offset.
                    head_offsets = tuple(view.data_ptr() - head_views[0].data_ptr() for view in head_views)
                    for part in contiguous_subviews(head_views[0]):
                        address = part.data_ptr()
                        strides.append(stride)
                        lengths.append(part.numel() * element_bytes)
                        for partition_bases, offset in zip(bases, head_offsets, strict=True):
                            partition_bases.append(address + offset)

            if pp_layer_slices is not None:
                pp_layer_slices[layer.physical_layer_id] = slice(layer_segment_start, len(lengths))

        block_strides = _compact_uniform_values(strides)
        segment_lengths = _compact_uniform_values(lengths)
        if layer_slices is not None and pp_layer_slices is not None:
            layer_slices[pp_rank] = pp_layer_slices
        for head_partition_index, partition_id in enumerate(partitions):
            local[partition_id] = {
                "bases": _readonly_u64(bases[head_partition_index]),
                "block_strides": block_strides,
                "lengths": segment_lengths,
            }
    return keys, local, layer_slices


def _default_key_builder(metadata: KeyMetadata) -> tuple[str, str]:
    # Defer the canonical codec's Worker dependencies until binding.
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey

    return PoolKey(metadata, "").to_string(), ""


def build_lookup_keys(topology: KVPoolTopology, group_id: int) -> tuple[tuple[str, str], ...]:
    """Bind the complete canonical remote family, independently of local tensors.

    Lookup requires every PP/DCP/head partition, including partitions not owned
    by this Worker. Align-state groups use TP ranks instead of attention heads.
    """
    group = next(group for group in topology.transfer_groups if group.group_id == group_id)
    partitions = topology.consumer_pipeline_partitions or ()
    pp_size = len(partitions) if len(partitions) > 1 else topology.pp_size
    head_count = topology.tp_size if group.uses_align_state else topology.tp_partition.key_rank_count
    return tuple(
        _default_key_builder(replace(group.key_metadata, pp_rank=pp, dcp_rank=dcp, head_or_tp_rank=head))
        for pp in range(pp_size)
        for dcp in range(topology.dcp_size)
        for head in range(head_count)
    )


def select_layers(
    group: KVPoolGroupTopology, target_layer_counts: Sequence[int] | None
) -> dict[int, list[KVPoolLayerTopology]]:
    """Assign local layers to resolved PP spans, preserving actual rank numbers."""
    pp_counts = target_layer_counts or ()
    pp_stops = tuple(accumulate(pp_counts)) if len(pp_counts) > 1 else ()
    layers_by_pp: dict[int, list[KVPoolLayerTopology]] = {}
    for layer in group.layers:
        pp_rank = bisect_right(pp_stops, layer.physical_layer_id) if pp_stops else group.key_metadata.pp_rank
        if pp_stops and pp_rank == len(pp_stops):
            raise ValueError("Target PP layer spans do not cover the group's layers")
        layers_by_pp.setdefault(pp_rank, []).append(layer)
    return layers_by_pp


def _kernel_blocks_per_group(tensor: torch.Tensor, spec: KVCacheSpec, token_axis: int) -> int:
    if isinstance(spec, MambaSpec):
        return 1
    # Match the runner's storage-row units, not logical tokens: compressed MLA
    # can store fewer rows per manager block. Tensor rows expose the kernel size.
    storage_rows = get_storage_block_size(spec)
    scale, remainder = divmod(storage_rows, tensor.shape[token_axis])
    if remainder or scale <= 0:
        raise ValueError("Kernel rows must divide the group's storage block size")
    return scale


def split_head_partitions(tensor: torch.Tensor, head_partition_count: int, head_axis: int) -> tuple[torch.Tensor, ...]:
    if head_partition_count == 1 or not tensor.numel():
        return (tensor,) * head_partition_count
    # Unflatten rejects head counts that cannot be divided equally.
    return tensor.unflatten(head_axis, (head_partition_count, -1)).unbind(head_axis)


def contiguous_subviews(view: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield contiguous subviews in logical element order without copying data."""
    if view.numel() == 0:
        return
    if view.is_contiguous():
        yield view
        return
    for child in view.unbind(0):
        yield from contiguous_subviews(child)


# ========================
# Remote layouts
# ========================


def build_key_range_remote(local: LocalTable) -> RemoteTable:
    """Pack each PP payload tightly, sharing offsets across head partitions."""
    remote: RemoteTable = {}
    for (pp_rank, _, _), segments in local.items():
        if pp_rank in remote:
            continue
        boundaries = _segment_byte_boundaries(segments["lengths"], len(segments["bases"]))
        remote[pp_rank] = {"size_bytes": int(boundaries[-1]), "offsets": _readonly_u64(boundaries[:-1])}
    return remote


def build_gva_remote(topology: KVPoolTopology, local: LocalTable, layer_slices: LayerTable) -> RemoteTable:
    """Reserve global layers and aligned CP slots once per PP, without resolving GVA bases.

    Input tables must come from the same layerwise binding of one group. Runtime
    object bases remain owned by Backend sessions and are not part of this layout.
    """
    remote: RemoteTable = {}
    for (pp_rank, _, _), segments in local.items():
        if pp_rank in remote:
            continue
        boundaries = _segment_byte_boundaries(segments["lengths"], len(segments["bases"]))
        layer_spans = layer_slices[pp_rank]
        if topology.pp_partition.global_layer_count is not None:
            layer_sizes = {int(boundaries[span.stop]) - int(boundaries[span.start]) for span in layer_spans.values()}
            if len(layer_sizes) > 1:
                raise ValueError("Global layer spans require uniform per-layer payloads")
        size, origin = _gva_object_span(topology, int(boundaries[-1]), len(layer_spans))
        offsets = boundaries[:-1]
        if origin:
            offsets += np.uint64(origin)
        remote[pp_rank] = {"size_bytes": size, "offsets": _readonly_u64(offsets)}
    return remote


def _segment_byte_boundaries(lengths: UInt64Array, segment_count: int) -> UInt64Array:
    if len(lengths) == 1:
        boundaries: UInt64Array = np.arange(segment_count + 1, dtype=np.uint64)
        boundaries *= lengths[0]
        return boundaries
    boundaries = np.empty(segment_count + 1, dtype=np.uint64)
    boundaries[0] = 0
    np.cumsum(lengths, out=boundaries[1:])
    return boundaries


def _gva_object_span(topology: KVPoolTopology, payload_bytes: int, layer_count: int) -> tuple[int, int]:
    shard_bytes, origin = payload_bytes, 0
    global_layers = topology.pp_partition.global_layer_count
    if global_layers is not None:
        reserved_layers = max(global_layers, topology.stage_layer_count, layer_count)
        if topology.stage_layer_start < 0 or topology.stage_layer_start + layer_count > reserved_layers:
            raise ValueError("Stage payload exceeds the reserved global layer span")
        bytes_per_layer = payload_bytes // layer_count
        shard_bytes = reserved_layers * bytes_per_layer
        origin = topology.stage_layer_start * bytes_per_layer
    count = topology.cp_partition.object_shard_count
    alignment = topology.cp_partition.shard_alignment_bytes
    shard_index = topology.object_shard_index
    if count <= 0 or alignment <= 0 or not 0 <= shard_index < count:
        raise ValueError("Invalid remote shard count, alignment or index")
    shard_stride = ((shard_bytes + alignment - 1) // alignment) * alignment
    return count * shard_stride, origin + shard_index * shard_stride


# ========================
# Runtime mapping
# ========================


def map_keys(key_parts: tuple[str, str], hashes: Sequence[bytes | str]) -> tuple[str, ...]:
    """Insert block hashes into an object's wire name."""
    prefix, suffix = key_parts
    return tuple(prefix + (value.hex() if isinstance(value, bytes) else value) + suffix for value in hashes)


def local_addresses(
    segments: LocalSegmentData, block_ids: Sequence[int] | UInt64Array, segment_range: slice = _ALL_SEGMENTS
) -> UInt64Array:
    """Evaluate [block, segment] addresses; allocation bounds remain caller-owned."""
    ids = np.asarray(block_ids, dtype=np.uint64)
    if ids.ndim != 1:
        raise ValueError("Local block IDs must be one-dimensional")
    strides = segments["block_strides"]
    if len(strides) != 1:
        strides = strides[segment_range]
    # Keep uniform strides scalar so the multiplication allocates one offset
    # per block, not a full block-by-segment temporary array.
    return ids[:, None] * strides + segments["bases"][segment_range]


def local_lengths(segments: LocalSegmentData, segment_range: slice = _ALL_SEGMENTS) -> UInt64Array:
    """Select lengths: keep uniform values as singletons, but return an empty array for empty selections."""
    lengths = segments["lengths"]
    if len(lengths) == 1:
        return lengths if len(segments["bases"][segment_range]) else lengths[:0]
    return lengths[segment_range]


# ========================
# Read-only arrays
# ========================


def _compact_uniform_values(values: Sequence[int]) -> UInt64Array:
    # Allocate only the retained singleton, not a slice holding a full array alive.
    if values and all(value == values[0] for value in values):
        return _readonly_u64((values[0],))
    return _readonly_u64(values)


def _readonly_u64(values: Sequence[int] | UInt64Array) -> UInt64Array:
    array = np.asarray(values, dtype=np.uint64)
    array.flags.writeable = False
    return array
