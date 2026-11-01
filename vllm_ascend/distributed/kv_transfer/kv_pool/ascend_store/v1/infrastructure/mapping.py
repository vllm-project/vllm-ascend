"""Map logical token positions to block hashes and caller-supplied block IDs.

Layout binding defines partition Keys, Local addresses, layer indices and Remote offsets.
Objects retain layer order and independent-storage declaration order. Within a storage,
segment order follows Block 0 / head partition 0 addresses, with padding omitted.
Each segment retains its own object-coordinate strides. Within one PP partition (DCP rank fixed):

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
from dataclasses import dataclass, field, replace
from heapq import merge
from itertools import accumulate, product
from typing import TYPE_CHECKING, NamedTuple, TypeAlias, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import get_storage_block_size

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheSpec

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata

    from .topology import KVPoolGroupTopology, KVPoolLayerTopology, KVPoolTopology


# ========================
# Logical block identities
# ========================


BlockPair: TypeAlias = tuple[int, bytes | str, int]


class TokenChunk(NamedTuple):
    start_token: int
    token_count: int
    block_hash: bytes | str


@dataclass(frozen=True, slots=True)
class BlockHashMapping:
    """Align content hashes and caller-supplied block IDs by logical token position.

    Hashes describe the full prefix; block IDs may retain only its suffix.
    Bound token geometry determines their alignment, independently of selection
    policy, transfer direction and physical memory addresses.
    """

    block_size: int
    hash_block_size: int
    _hashes_per_block: int = field(init=False, repr=False)
    _terminal_hash_offset: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.hash_block_size <= 0 or self.block_size % self.hash_block_size:
            raise ValueError("Block size must be a positive multiple of the hash block size")
        hashes_per_block = self.block_size // self.hash_block_size
        object.__setattr__(self, "_hashes_per_block", hashes_per_block)
        object.__setattr__(self, "_terminal_hash_offset", hashes_per_block - 1)

    def pair_block_ids(
        self,
        hashes: Sequence[bytes | str],
        block_ids: Sequence[int],
        *,
        start_token: int,
        end_token: int,
        tail_boundary: int | None = None,
    ) -> Iterator[BlockPair]:
        if not hashes or end_token <= 0:
            return
        hashes_per_block, terminal_hash_offset = self._hashes_per_block, self._terminal_hash_offset
        hash_count = len(hashes) // hashes_per_block
        block_count = -(-end_token // self.block_size)
        if tail_boundary is not None:
            if tail_boundary < end_token or -(-tail_boundary // self.block_size) != block_count:
                raise ValueError(f"Tail-key boundary {tail_boundary} does not identify the tail ending at {end_token}")
        else:
            block_count = min(block_count, hash_count)
        block_offset = max(block_count - len(block_ids), 0)
        for logical_index in range(max(start_token // self.block_size, block_offset, 0), block_count):
            if tail_boundary is not None and logical_index == block_count - 1:
                value = self.hash_at_boundary(tail_boundary, hashes)
            elif logical_index < hash_count:
                value = hashes[logical_index * hashes_per_block + terminal_hash_offset]
            else:
                continue
            yield logical_index, value, block_ids[logical_index - block_offset]

    def iter_hash_ranges(
        self,
        hashes: Sequence[bytes | str],
        *,
        start_token: int,
        end_token: int,
        fine_grained: bool = False,
    ) -> Iterator[TokenChunk]:
        """Enumerate content identities without requiring an allocated local block table."""
        if not hashes or end_token <= 0:
            return
        if fine_grained:
            for index in range(
                max(start_token // self.hash_block_size, 0), min(len(hashes), end_token // self.hash_block_size)
            ):
                boundary = (index + 1) * self.hash_block_size
                start = (-(-boundary // self.block_size) - 1) * self.block_size
                yield TokenChunk(start, boundary - start, hashes[index])
        else:
            hashes_per_block, terminal_hash_offset = self._hashes_per_block, self._terminal_hash_offset
            count = min(len(hashes) // hashes_per_block, -(-end_token // self.block_size))
            for index in range(max(start_token // self.block_size, 0), count):
                start = index * self.block_size
                yield TokenChunk(
                    start,
                    min(self.block_size, end_token - start),
                    hashes[index * hashes_per_block + terminal_hash_offset],
                )

    def hash_at_boundary(self, boundary: int, hashes: Sequence[bytes | str]) -> bytes | str:
        if boundary <= 0 or boundary % self.hash_block_size:
            raise ValueError(f"Token boundary {boundary} is not aligned to the hash block size")
        index = boundary // self.hash_block_size - 1
        if index >= len(hashes):
            raise ValueError(f"Token boundary {boundary} has no corresponding block hash")
        return hashes[index]


# ========================
# Layout tables
# ========================


_ALL_SEGMENTS = slice(None)

PartitionId = tuple[int, int, int]  # PP, actual DCP and head/TP partition ranks within one group binding.
UInt64Array: TypeAlias = NDArray[np.uint64]
KeyTable = dict[PartitionId, tuple[str, str]]
KeyBuilder: TypeAlias = Callable[["KeyMetadata"], tuple[str, str]]


class _Layout(NamedTuple):
    """Byte geometry: [logical block, head partition, *content axes].

    The first two axes locate objects and never fold into their contents.
    Normalized content axes descend by stride, ending in contiguous bytes.
    """

    offset: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]


_GeometryCache: TypeAlias = dict[tuple[_Layout, ...], tuple[_Layout, ...]]


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
# Key and local binding
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
    head_partition_count = (
        topology.config.tp_partition.key_slices_per_rank if topology.config.tp_partition.tp_mismatch else 1
    )
    first_head_rank = metadata.head_or_tp_rank * head_partition_count
    # HND/LBHNC caches arrive as per-layer head-major views.
    head_axis = 1 if kv_cache_layout in ("HND", "LBHNC") else 2
    builder = _default_key_builder if key_builder is None else key_builder

    keys: KeyTable = {}
    local: LocalTable = {}
    layer_slices: LayerTable | None = {} if layerwise else None
    geometry_cache: _GeometryCache = {}
    for pp_rank, layers in select_layers(group, topology.config.pp_partition.target_layer_counts).items():
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
            for origin, segments in _layer_layout(
                layer, group.kv_cache_spec, kv_caches, head_axis, head_partition_count, geometry_cache
            ):
                for segment in segments:
                    strides.append(segment.strides[0])
                    lengths.append(segment.shape[-1])
                    for head, partition_bases in enumerate(bases):
                        partition_bases.append(origin + segment.offset + head * segment.strides[1])

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
    partitions = topology.config.pp_partition.target_layer_counts or ()
    pp_size = len(partitions) if len(partitions) > 1 else topology.config.pp_size
    head_count = topology.config.tp_size if group.uses_align_state else topology.config.tp_partition.key_rank_count
    return tuple(
        _default_key_builder(replace(group.key_metadata, pp_rank=pp, dcp_rank=dcp, head_or_tp_rank=head))
        for pp in range(pp_size)
        for dcp in range(topology.config.cp_partition.dcp_size)
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


def _layer_layout(
    layer: KVPoolLayerTopology,
    group_spec: KVCacheSpec,
    kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
    head_axis: int,
    head_partition_count: int,
    geometry_cache: _GeometryCache,
) -> Iterator[tuple[int, tuple[_Layout, ...]]]:
    """Bind storage origins to reusable layouts, without expanding each component first."""
    token_axis = 2 if head_axis == 1 else 1
    storage_layouts: dict[int, tuple[int, list[_Layout]]] = {}
    for name in layer.layer_names:
        spec = group_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[name]
        cache = kv_caches[name]
        tensors = (cache,) if isinstance(cache, torch.Tensor) else cache
        for tensor in tensors:
            if not tensor.numel():
                continue
            scale = _kernel_blocks_per_group(tensor, spec, token_axis)
            if tensor.shape[0] % scale:
                raise ValueError("Kernel block count must divide into complete logical blocks")
            element_bytes = tensor.element_size()
            shape = (tensor.shape[0] // scale, head_partition_count, scale, *tensor.shape[1:], element_bytes)
            byte_strides = tuple(step * element_bytes for step in tensor.stride())
            strides = (scale * byte_strides[0], 0, *byte_strides, 1)
            if head_partition_count > 1:
                axis = head_axis + 2
                heads, remainder = divmod(shape[axis], head_partition_count)
                if remainder:
                    raise ValueError("Head count must divide into equal partitions")
                shape = (*shape[:axis], heads, *shape[axis + 1 :])
                strides = (strides[0], heads * strides[axis], *strides[2:])
            base = tensor.data_ptr()
            storage = tensor.untyped_storage().data_ptr()
            if storage not in storage_layouts:
                storage_layouts[storage] = (base, [])
            origin, layouts = storage_layouts[storage]
            layouts.append(_Layout(base - origin, shape, strides))
    # Independent allocations keep declaration order; absolute addresses never enter the layout cache.
    for origin, layouts in storage_layouts.values():
        geometry = tuple(layouts)
        if geometry not in geometry_cache:
            geometry_cache[geometry] = tuple(_storage_segments(layouts))
        yield origin, geometry_cache[geometry]


def _fold_layout(layout: _Layout) -> _Layout:
    axes = sorted((step, size) for size, step in zip(layout.shape[2:], layout.strides[2:], strict=True) if size > 1)
    shape, strides = [1], [1]
    for step, size in axes:
        # Flatten adjacent index dimensions; only stride-one repetitions form contiguous bytes.
        if step == shape[-1] * strides[-1]:
            shape[-1] *= size
        else:
            shape.append(size)
            strides.append(step)
    return _Layout(
        layout.offset,
        (*layout.shape[:2], *reversed(shape)),
        (*layout.strides[:2], *reversed(strides)),
    )


def _join_layouts(left: _Layout, right: _Layout) -> _Layout | None:
    """Add a translated-component axis or extend one content axis, preserving object coordinates."""
    if left.shape[:2] != right.shape[:2] or left.strides[:2] != right.strides[:2]:
        return None
    distance = right.offset - left.offset
    if left.shape == right.shape and left.strides == right.strides:
        combined = _Layout(
            left.offset, (*left.shape[:2], 2, *left.shape[2:]), (*left.strides[:2], distance, *left.strides[2:])
        )
    else:
        # A missing content axis has extent one, allowing another component to extend an existing repetition.
        left_axes = dict(zip(left.strides[2:], left.shape[2:], strict=True))
        right_axes = dict(zip(right.strides[2:], right.shape[2:], strict=True))
        changed = [
            step for step in left_axes.keys() | right_axes.keys() if left_axes.get(step, 1) != right_axes.get(step, 1)
        ]
        if len(changed) != 1:
            return None
        step = changed[0]
        if distance != step * left_axes.get(step, 1):
            return None
        left_axes[step] = left_axes.get(step, 1) + right_axes.get(step, 1)
        combined = _Layout(left.offset, (*left.shape[:2], *left_axes.values()), (*left.strides[:2], *left_axes.keys()))
    combined = _fold_layout(combined)
    # Keep expansion address-ordered. Interleaving repetitions retain their original ordered streams.
    span = 1
    for size, step in zip(reversed(combined.shape[2:]), reversed(combined.strides[2:]), strict=True):
        if step < span:
            return None
        span += (size - 1) * step
    return combined


def _expand_layout(layout: _Layout) -> Iterator[_Layout]:
    # Only content gaps expand. Object coordinates stay symbolic, and bytes stay contiguous.
    shape = (*layout.shape[:2], layout.shape[-1])
    strides = (*layout.strides[:2], 1)
    steps = layout.strides[2:-1]
    for point in product(*(range(size) for size in layout.shape[2:-1])):
        offset = sum(index * step for index, step in zip(point, steps, strict=True))
        yield _Layout(layout.offset + offset, shape, strides)


def _storage_segments(layouts: list[_Layout]) -> Iterator[_Layout]:
    """Express component unions as content axes, then emit their contiguous byte intervals."""
    joined: list[_Layout] = []
    for current in sorted((_fold_layout(layout) for layout in layouts), key=lambda layout: layout.offset):
        while joined:
            combined = _join_layouts(joined[-1], current)
            if combined is None:
                break
            joined.pop()
            current = combined
        joined.append(current)
    if len(joined) == 1 and len(joined[0].shape) == 3:
        yield joined[0]
        return
    # A single augmented layout can also have adjacent intervals across repetition boundaries.
    ordered = iter(merge(*(_expand_layout(layout) for layout in joined)))
    current = next(ordered)
    length = current.shape[-1]
    for following in ordered:
        if (
            current.offset + length == following.offset
            and current.strides == following.strides
            and current.shape[:2] == following.shape[:2]
        ):
            length += following.shape[-1]
        else:
            yield current if length == current.shape[-1] else current._replace(shape=(*current.shape[:2], length))
            current = following
            length = current.shape[-1]
    yield current if length == current.shape[-1] else current._replace(shape=(*current.shape[:2], length))


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
        if topology.config.pp_partition.global_layer_count is not None:
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
    global_layers = topology.config.pp_partition.global_layer_count
    if global_layers is not None:
        reserved_layers = max(global_layers, topology.stage_layer_count, layer_count)
        if topology.stage_layer_start < 0 or topology.stage_layer_start + layer_count > reserved_layers:
            raise ValueError("Stage payload exceeds the reserved global layer span")
        bytes_per_layer = payload_bytes // layer_count
        shard_bytes = reserved_layers * bytes_per_layer
        origin = topology.stage_layer_start * bytes_per_layer
    count = topology.config.cp_partition.object_shard_count
    alignment = topology.config.cp_partition.shard_alignment_bytes
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
