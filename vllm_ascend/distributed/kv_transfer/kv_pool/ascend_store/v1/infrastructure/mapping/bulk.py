"""Bind local cache tensors and produce complete-object transfer arguments.

    registration + topology
              |
              v
    canonical prefixes + target PP layer spans
              |
              v
        TP mismatch?
          +-- no ----------------> contiguous.bind_segments --------+
          |                                                         |
          +-- yes --> inspect_entries                               |
                        | whole_entries_contiguous                  |
                        +-- True -> contiguous.bind_segments -------+
                        +-- False                                   |
                              |                                     |
                              v                                     |
                       compile_geometries -> materialize_segments --+
                                                                    |
                                                                    v
                                     select PP columns + bind local prefixes

An Entry is a registered nonempty tensor/view containing cache blocks. One block
may need several local segments; gaps between selected bytes are not payload.
One object key = (group, target PP, DCP, head/TP) prefix + block hash.
"Complete" covers that key's assigned bytes, not all model layers/heads;
Load, Store and Layerwise retain the same object identity and size.

*_by_key_prefix fields align in target-PP/local-head order; local DCP is fixed.
Lookup also includes nonlocal prefixes. physical_layer_ids are absolute IDs;
layer_segment_boundaries_by_key_prefix uses segment-array indices, starting at
zero and ending after the last layer. Block sizes count tokens, not bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from . import contiguous, head_slicing, layer_partitioning
from .types import BulkTransferArguments, KeysByPrefix, LocalMemorySegments, UInt64Array

if TYPE_CHECKING:
    import torch

    from ..configuration.topology import KVPoolGroupTopology, KVPoolTopology


class BulkMapping:
    """Own one original group's local transfer layout and global Lookup key prefixes.

    Registration arrays must have matching Entry order. Callers supply supported
    topology, select writers and manage KV allocation, readiness and completion.
    Each group's hashes, block IDs and masks are independent of other groups.
    """

    __slots__ = (
        "group_id",
        "group_block_size",
        "hash_block_size",
        "physical_layer_ids",
        "local_group_block_bytes",
        "lookup_key_prefixes",
        "key_prefixes",
        "physical_layer_ids_by_key_prefix",
        "object_sizes_bytes_by_key_prefix",
        "local_memory_segments_by_key_prefix",
        "layer_segment_boundaries_by_key_prefix",
    )

    def __init__(
        self,
        topology: KVPoolTopology,
        entry_addresses_by_group: Mapping[int, Sequence[int]],
        entry_block_payload_bytes_by_group: Mapping[int, Sequence[int]],
        entry_block_strides_bytes_by_group: Mapping[int, Sequence[int]],
        layer_entry_boundaries_by_group: Mapping[int, Sequence[int]],
        *,
        group_id: int | None = None,
        kv_cache_layout: str = "NHD",
        kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]] | None = None,
    ) -> None:
        # The canonical codec imports Worker dependencies; defer it until binding.
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey

        # ====================================================
        # Bind registration and resolve target PP layers
        # ====================================================

        groups = topology.transfer_groups
        group: KVPoolGroupTopology | None
        if group_id is None:
            if len(groups) != 1:
                raise ValueError("Select an original group_id when binding multiple transferable cache groups")
            group = groups[0]
        else:
            group = next((group for group in groups if group.group_id == group_id), None)
            if group is None:
                raise KeyError(f"Unknown transferable KV cache group {group_id}")
        target_pp_layer_counts = topology.consumer_pipeline_partitions or ()
        split_pipeline = len(target_pp_layer_counts) > 1

        self.group_id = group.group_id
        self.group_block_size = group.block_size
        self.hash_block_size = topology.hash_block_size
        self.physical_layer_ids = tuple(layer.physical_layer_id for layer in group.layers)
        if self.group_block_size <= 0:
            raise ValueError("Group block size must be positive")
        entry_addresses = np.array(entry_addresses_by_group[self.group_id], dtype=np.uint64)
        entry_block_payload_bytes = np.array(entry_block_payload_bytes_by_group[self.group_id], dtype=np.uint64)
        group_block_strides_bytes = np.array(entry_block_strides_bytes_by_group[self.group_id], dtype=np.uint64)
        if not len(entry_addresses):
            raise ValueError("KV cache group has no registered memory entries")
        for array in (entry_addresses, entry_block_payload_bytes, group_block_strides_bytes):
            array.flags.writeable = False
        self.local_group_block_bytes = int(entry_block_payload_bytes.sum())

        metadata = group.key_metadata
        # Lookup covers remote stages too; Consumer PP names objects, not execution ranks.
        key_pp_size = len(target_pp_layer_counts) if split_pipeline else topology.pp_size
        key_head_count = topology.tp_size if group.uses_align_state else topology.tp_partition.key_rank_count
        self.lookup_key_prefixes = tuple(
            PoolKey(replace(metadata, pp_rank=pp, dcp_rank=dcp, head_or_tp_rank=head), "").to_string()
            for pp in range(key_pp_size)
            for dcp in range(topology.dcp_size)
            for head in range(key_head_count)
        )
        pp_layer_partitions = (
            layer_partitioning.partition_layers(
                self.physical_layer_ids, layer_entry_boundaries_by_group[self.group_id], target_pp_layer_counts
            )
            if split_pipeline
            else ()
        )
        if len(pp_layer_partitions) == 1 and pp_layer_partitions[0].pp_rank == metadata.pp_rank:
            pp_layer_partitions = ()
        entry_segment_boundaries = None

        # ====================================================
        # Bind whole entries or head slices exactly once
        # ====================================================
        #
        # Bind all local entries before selecting layer spans, so PP spans
        # can share NumPy slices without recompiling head geometry per PP span.

        head_slices_per_rank = 1
        segments_by_head_slice: tuple[LocalMemorySegments, ...]
        if topology.tp_partition.tp_mismatch:
            head_slices_per_rank = topology.tp_partition.key_slices_per_rank
            head_major = kv_cache_layout in ("HND", "LBHNC")
            entry_layouts, whole_entries_contiguous = head_slicing.inspect_entries(
                group, entry_block_payload_bytes, group_block_strides_bytes, head_slices_per_rank, head_major, kv_caches
            )
            if whole_entries_contiguous:
                segments_by_head_slice = (
                    contiguous.bind_segments(entry_addresses, entry_block_payload_bytes, group_block_strides_bytes),
                )
            else:
                entry_segment_boundaries = [0]
                geometry = head_slicing.compile_geometries(
                    entry_layouts, head_slices_per_rank, head_major, entry_segment_boundaries
                )
                segments_by_head_slice = head_slicing.materialize_segments(
                    entry_addresses, group_block_strides_bytes, entry_layouts, head_slices_per_rank, geometry
                )
        else:
            segments_by_head_slice = (
                contiguous.bind_segments(entry_addresses, entry_block_payload_bytes, group_block_strides_bytes),
            )
        layer_segment_boundaries = (
            tuple(layer_entry_boundaries_by_group[self.group_id])
            if entry_segment_boundaries is None
            else tuple(entry_segment_boundaries[index] for index in layer_entry_boundaries_by_group[self.group_id])
        )

        # ====================================================
        # Bind local prefixes to complete object layouts
        # ====================================================

        # Replica ranks already share metadata. The caller selects Store writers.
        first_head_rank = metadata.head_or_tp_rank * head_slices_per_rank
        prefixes_per_pp = topology.dcp_size * key_head_count
        local_prefix_index_in_pp = metadata.dcp_rank * key_head_count + first_head_rank
        if not pp_layer_partitions:
            prefix_start = metadata.pp_rank * prefixes_per_pp + local_prefix_index_in_pp
            self.key_prefixes = self.lookup_key_prefixes[prefix_start : prefix_start + head_slices_per_rank]
            self.physical_layer_ids_by_key_prefix = (self.physical_layer_ids,) * head_slices_per_rank
            self.object_sizes_bytes_by_key_prefix = (
                self.local_group_block_bytes // head_slices_per_rank,
            ) * head_slices_per_rank
            self.local_memory_segments_by_key_prefix = segments_by_head_slice
            self.layer_segment_boundaries_by_key_prefix = (layer_segment_boundaries,) * head_slices_per_rank
            return

        # ====================================================
        # Select target PP layers for each key prefix
        # ====================================================

        prefixes: list[str] = []
        layers_by_prefix: list[tuple[int, ...]] = []
        object_sizes_bytes: list[int] = []
        segments_by_prefix: list[LocalMemorySegments] = []
        layer_boundaries_by_prefix: list[tuple[int, ...]] = []
        for pp_partition in pp_layer_partitions:
            pp_segments_by_head_slice = (
                segments_by_head_slice
                if len(pp_layer_partitions) == 1
                else layer_partitioning.slice_layer_segments(
                    segments_by_head_slice, pp_partition, entry_segment_boundaries
                )
            )
            object_size_bytes = (
                int(entry_block_payload_bytes[pp_partition.entry_start_index : pp_partition.entry_end_index].sum())
                // head_slices_per_rank
            )
            prefix_start = pp_partition.pp_rank * prefixes_per_pp + local_prefix_index_in_pp
            prefixes.extend(self.lookup_key_prefixes[prefix_start : prefix_start + head_slices_per_rank])
            if len(pp_layer_partitions) == 1:
                boundaries = layer_segment_boundaries
            else:
                segment_start = layer_segment_boundaries[pp_partition.local_layer_start_index]
                boundaries = tuple(
                    boundary - segment_start
                    for boundary in layer_segment_boundaries[
                        pp_partition.local_layer_start_index : pp_partition.local_layer_end_index + 1
                    ]
                )
            for segments in pp_segments_by_head_slice:
                layers_by_prefix.append(pp_partition.physical_layer_ids)
                object_sizes_bytes.append(object_size_bytes)
                segments_by_prefix.append(segments)
                layer_boundaries_by_prefix.append(boundaries)
        self.key_prefixes = tuple(prefixes)
        self.physical_layer_ids_by_key_prefix = tuple(layers_by_prefix)
        self.object_sizes_bytes_by_key_prefix = tuple(object_sizes_bytes)
        self.local_memory_segments_by_key_prefix = tuple(segments_by_prefix)
        self.layer_segment_boundaries_by_key_prefix = tuple(layer_boundaries_by_prefix)

    # =======================================
    # Runtime keys and transfer arguments
    # =======================================

    def lookup_keys(self, hashes: Sequence[bytes | str]) -> KeysByPrefix:
        """Return all required object keys in target PP, DCP, head, row order."""
        return self._map_keys(self.lookup_key_prefixes, hashes)

    def keys(self, hashes: Sequence[bytes | str]) -> KeysByPrefix:
        """Return local keys in [key prefix][input row] order."""
        return self._map_keys(self.key_prefixes, hashes)

    def arguments(
        self,
        keys: Sequence[Sequence[str]],
        block_ids: Sequence[int] | UInt64Array,
        *,
        selected_keys: Sequence[bool] | None = None,
    ) -> BulkTransferArguments:
        """Select complete payloads without clipping to logical token coverage.

        Input keys retain every row, including unselected rows. The optional
        flat mask is key-prefix-major, then input-row-major. Repeated IDs and input
        order are preserved. Callers ensure allocation bounds and object validity.
        """
        ids = np.asarray(block_ids, dtype=np.uint64)
        if ids.ndim != 1:
            raise ValueError("Block IDs must be one-dimensional rows")
        row_count = len(ids)
        if len(keys) != len(self.local_memory_segments_by_key_prefix) or any(
            len(prefix_keys) != row_count for prefix_keys in keys
        ):
            raise ValueError("Keys must align local key prefixes and Block ID rows")
        selections = None
        if selected_keys is not None:
            selections = np.asarray(selected_keys, dtype=np.bool_)
            if selections.ndim != 1 or selections.size != len(keys) * row_count:
                raise ValueError("Key selection must be a flat key-prefix-major row mask")
            selections = selections.reshape(len(keys), row_count)

        backend_keys: list[str] = []
        backend_addresses: list[list[int]] = []
        backend_lengths: list[list[int]] = []
        if not row_count:
            return backend_keys, backend_addresses, backend_lengths

        for key_prefix_index, (prefix_keys, segments) in enumerate(
            zip(keys, self.local_memory_segments_by_key_prefix, strict=True)
        ):
            selected_rows = None
            if selections is not None:
                selected_count = np.count_nonzero(selections[key_prefix_index])
                if not selected_count:
                    continue
                if selected_count != row_count:
                    selected_rows = np.flatnonzero(selections[key_prefix_index])
            selected_block_ids = ids if selected_rows is None else ids[selected_rows]
            backend_keys.extend(prefix_keys if selected_rows is None else (prefix_keys[row] for row in selected_rows))
            addresses = selected_block_ids[:, None] * segments.group_block_strides_bytes + segments.block_zero_addresses
            lengths = np.broadcast_to(segments.segment_lengths_bytes, segments.block_zero_addresses.shape).tolist()
            backend_addresses.extend(addresses.tolist())
            # Lists belong to individual Backend rows, even when their lengths match.
            backend_lengths.extend(lengths.copy() for _ in selected_block_ids)
        return backend_keys, backend_addresses, backend_lengths

    @staticmethod
    def _map_keys(prefixes: tuple[str, ...], hashes: Sequence[bytes | str]) -> KeysByPrefix:
        hash_strings = tuple(value if isinstance(value, str) else value.hex() for value in hashes)
        return tuple(tuple(prefix + value for value in hash_strings) for prefix in prefixes)
