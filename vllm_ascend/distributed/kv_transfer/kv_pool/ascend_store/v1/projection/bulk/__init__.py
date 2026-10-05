"""Registration binders for the explicit AscendStore Bulk variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from ...topology import KVPoolTopology
from .consumer_pipeline import ConsumerPipelineBulkProjection, bind_consumer_pipeline_bulk_projection
from .hybrid import HybridBulkProjection, bind_hybrid_bulk_projection
from .ordinary import OrdinaryBulkProjection, bind_ordinary_bulk_projection
from .tp_mismatch import TPMismatchBulkProjection, bind_tp_mismatch_bulk_projection

BulkProjection: TypeAlias = (
    OrdinaryBulkProjection | TPMismatchBulkProjection | ConsumerPipelineBulkProjection | HybridBulkProjection
)


@dataclass(frozen=True, slots=True)
class BulkProjectionBinder:
    """Bind one already-selected Bulk variant when KV memory is registered."""

    topology: KVPoolTopology
    max_model_len: int
    use_eagle: bool = False
    retention_interval: int | None = None

    def __post_init__(self) -> None:
        groups = self.topology.transfer_groups
        if not groups:
            raise ValueError("Bulk projection requires at least one transferable cache group")
        partitions = self.topology.consumer_pipeline_partitions
        if self.topology.tp_partition.tp_mismatch:
            if len(groups) != 1 or len(self.topology.groups) != 1 or groups[0].uses_align_state:
                raise ValueError("TP mismatch requires one dense transferable cache group")
            if partitions is not None and len(partitions) > 1:
                raise ValueError("Consumer pipeline projection cannot be composed with TP mismatch")
        if self.topology.dcp_size > 1 and self.topology.pcp_size > 1:
            raise ValueError("Combined DCP and PCP Bulk Store lacks a proven identical-key writer partition")
        if self.topology.dcp_size > 1 and self.topology.put_step > self.topology.dcp_size:
            raise ValueError("DCP Bulk Store with remaining same-key TP replicas lacks a proven writer partition")

    def bind(
        self,
        base_addresses: Mapping[int, Sequence[int]],
        block_lengths: Mapping[int, Sequence[int]],
        block_strides: Mapping[int, Sequence[int]],
        layer_entry_offsets: Mapping[int, Sequence[int]],
        *,
        object_sizes: Mapping[int, int] | None = None,
        object_offsets: Mapping[int, int] | None = None,
    ) -> BulkProjection:
        del object_sizes, object_offsets
        common = (
            self.topology,
            self.max_model_len,
            base_addresses,
            block_lengths,
            block_strides,
            layer_entry_offsets,
        )
        groups = self.topology.transfer_groups
        if self.topology.tp_partition.tp_mismatch:
            return bind_tp_mismatch_bulk_projection(*common)
        partitions = self.topology.consumer_pipeline_partitions
        if len(groups) == 1 and not groups[0].uses_align_state and partitions is not None and len(partitions) > 1:
            return bind_consumer_pipeline_bulk_projection(*common)
        if len(groups) == 1 and not groups[0].uses_align_state:
            return bind_ordinary_bulk_projection(*common)
        return bind_hybrid_bulk_projection(
            *common,
            use_eagle=self.use_eagle,
            retention_interval=self.retention_interval,
        )


def compile_bulk_projection_binder(
    topology: KVPoolTopology,
    max_model_len: int,
    *,
    use_eagle: bool = False,
    retention_interval: int | None = None,
) -> BulkProjectionBinder:
    return BulkProjectionBinder(topology, max_model_len, use_eagle, retention_interval)


__all__ = (
    "BulkProjectionBinder",
    "BulkProjection",
    "ConsumerPipelineBulkProjection",
    "HybridBulkProjection",
    "OrdinaryBulkProjection",
    "TPMismatchBulkProjection",
    "compile_bulk_projection_binder",
)
