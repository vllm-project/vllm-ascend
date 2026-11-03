# SPDX-License-Identifier: Apache-2.0
"""Accepted state descriptors extending vLLM's TP-sharded Store layout.

Reuse the upstream key format, shard ownership and segment-template arrays.
Only physical state registration and exact boundary-block addressing differ
from the full-attention implementation. No transport or coordination lives
in this module. The caller supplies pinned accepted blocks, not live state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from math import prod
from typing import TYPE_CHECKING

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    KeyMetadata,
    TPShardedStoreLayout,
)

from .cache_schema import RegisteredState
from .layout import TensorLayout
from .state_transfer import ComponentLayout, plan_component_transfer

if TYPE_CHECKING:
    import torch


class AcceptedStateStoreLayout(TPShardedStoreLayout):
    """One native state group with compatible semantic membership on peers.

    Wire component order is (layer name, component name). Each component spec
    declares its wire axis.
    Different native group partitions are not interchangeable: the caller must
    negotiate a compatible Store grouping before constructing this layout.
    Store namespace must also bind the complete semantic schema and model's
    accepted-boundary dependency contract.
    """

    store_format = "accepted-state-layout-v1"

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int,
        local_tp_size: int,
        store_tp_size: int,
        tp_rank: int,
        num_state_heads: int,
        components: Sequence[RegisteredState],
    ) -> None:
        if (
            local_tp_size <= 0
            or store_tp_size <= 0
            or store_tp_size % local_tp_size
            or num_state_heads <= 0
            or num_state_heads % store_tp_size
            or not 0 <= tp_rank < local_tp_size
        ):
            raise ValueError("Store shards must evenly partition local state heads")
        if self.store_format not in metadata.store_namespace:
            raise ValueError("accepted states require an explicit compatible Store namespace")
        self.cache_namespace = metadata.store_namespace
        self.components = tuple(sorted(components, key=lambda c: (c.layer_name, c.spec.name)))
        if not self.components or len({c.group_id for c in self.components}) != 1:
            raise ValueError("one state layout describes exactly one native group")
        identities = [(c.layer_name, c.spec.name) for c in self.components]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate state component")
        semantic_components = []
        for component in self.components:
            layout = component.checkpoint_layout
            wire_axis = component.spec.wire_shard_axis
            if type(wire_axis) is not int or not 0 <= wire_axis < len(layout.tensor.shape):
                raise ValueError("wire shard axis is outside the state shape")
            semantic_components.append(
                dict(
                    layer=component.layer_name,
                    component=component.spec.name,
                    dtype=layout.dtype,
                    projections=layout.projection_sizes,
                    dimensions=layout.logical_shape[1:],
                    wire_shard_axis=wire_axis,
                )
            )
            if (
                layout.replicated
                or layout.tp_size != local_tp_size
                or layout.tp_rank != tp_rank
                or component.block_tokens != block_size
                or any(size % store_tp_size for size in layout.projection_sizes)
            ):
                raise ValueError("state component is incompatible with this Store partition")
        schema_hash = hashlib.sha256(
            json.dumps(
                semantic_components,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        metadata = replace(
            metadata,
            store_namespace=f"{metadata.store_namespace}@state_schema:{schema_hash}",
        )
        super().__init__(
            metadata,
            block_size,
            hash_block_size,
            local_tp_size,
            store_tp_size,
            tp_rank,
            num_state_heads,
        )
        self._num_registered_blocks = 0

    def register_kv_caches(
        self,
        kv_caches: Sequence[torch.Tensor],
        num_blocks: int,
    ) -> None:
        if (
            num_blocks <= 0
            or len(kv_caches) != len(self.components)
            or {id(t) for t in kv_caches} != {id(c.tensor) for c in self.components}
        ):
            raise ValueError("registration does not match described state tensors")
        templates = []
        for shard in self.local_shard_ids:
            bases: list[int] = []
            strides: list[int] = []
            sizes: list[int] = []
            for component in self.components:
                tensor = component.tensor
                if tensor.shape[0] < num_blocks:
                    raise ValueError("state tensor does not cover registered block IDs")
                source = component.checkpoint_layout
                wire_axis = component.spec.wire_shard_axis
                shape = list(source.logical_shape[1:])
                shape.insert(wire_axis, sum(source.projection_sizes) // self.store_tp_size)
                wire_shape = tuple(shape)
                wire_strides = tuple(prod(wire_shape[axis + 1 :]) for axis in range(len(wire_shape)))
                wire = ComponentLayout(
                    TensorLayout(wire_shape, wire_strides, source.tensor.itemsize),
                    source.dtype,
                    source.projection_sizes,
                    wire_axis,
                    self.store_tp_size,
                    shard,
                )
                copies = sorted(
                    plan_component_transfer(source, wire),
                    key=lambda c: c.destination_offset,
                )
                offset = 0
                first_segment = len(bases)
                block_stride = tensor.stride(0) * tensor.element_size()
                for copy in copies:
                    if copy.destination_offset != offset:
                        raise ValueError("Store object contains a hole or overlapping segment")
                    address = tensor.data_ptr() + copy.source_offset
                    # Sorting into wire order may make independently sliced
                    # projections contiguous again. Merge within this tensor,
                    # never across separately registered allocations.
                    if len(bases) > first_segment and bases[-1] + sizes[-1] == address:
                        sizes[-1] += copy.num_bytes
                    else:
                        bases.append(address)
                        strides.append(block_stride)
                        sizes.append(copy.num_bytes)
                    offset += copy.num_bytes
                if offset != prod(shape) * source.tensor.itemsize:
                    raise ValueError("Store object does not cover the whole logical shard")
            templates.append((bases, strides, sizes))
        self._set_segment_templates(templates)
        self._num_registered_blocks = num_blocks

    def prepare_boundary_values(
        self,
        block_ids: Sequence[int],
        shard_ids: Sequence[int],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Build SDK pointer lists from exact pinned state block IDs.

        The same ordered segments pack PUTs and scatter GETs. Installation
        readiness and initialization of nonpersistent scratch remain the
        receiving worker/model adapter's responsibility.
        """
        if len(block_ids) != len(shard_ids):
            raise ValueError("each boundary block needs a Store shard")
        if not block_ids:
            return [], []
        if any(type(b) is not int or not 0 < b < self._num_registered_blocks for b in block_ids):
            raise ValueError("invalid, null or unregistered state block")
        first = self.local_shard_ids[0]
        if any(type(s) is not int or not first <= s < first + self.shards_per_rank for s in shard_ids):
            raise ValueError("Store shard is not owned by this worker")
        local_shards = np.asarray(shard_ids, dtype=np.int64) - first
        blocks = np.asarray(block_ids, dtype=np.uint64)
        addresses = self._shard_addr_bases[local_shards] + blocks[:, None] * self._shard_block_strides[local_shards]
        return addresses.tolist(), self._shard_sizes[local_shards].tolist()

    def prepare_values(
        self,
        chunks: Sequence[tuple[int, int]],
        block_ids: list[int],
        shard_ids: Sequence[int],
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        """Resolve receive slots from the core-owned, stable allocation snapshot.

        Source saving must still use exact BoundaryPut IDs. Receiving follows
        native StoreLayout: a full aligned chunk maps to its allocated column.
        The scheduler has checked exclusive destination ownership before enqueue.
        """
        for start, _ in chunks:
            if start < 0 or start % self.block_size:
                raise ValueError("state receive needs aligned block starts")
        # The native scheduler already established exclusive target ownership.
        # Reuse the upstream full-block/shard checks and template addressing.
        return super().prepare_values(chunks, block_ids, shard_ids)

    def initialize_received_scratch(self, block_ids: Sequence[int]) -> bool:
        """Initialize only regions omitted by the accepted-history wire format.

        Return whether device work was queued, so the receiver can fence its
        stream before reporting completion. Persistent history is not modified.
        """
        device_work = False
        for block_id in dict.fromkeys(block_ids):
            for component in self.components:
                spec = component.spec
                if spec.history_axis is None:
                    continue
                slot = component.slot(block_id)
                count = slot.shape[spec.history_axis] - spec.history_rows
                if count:
                    slot.narrow(spec.history_axis, spec.history_rows, count).zero_()
                    device_work |= slot.device.type != "cpu"
        return device_work
