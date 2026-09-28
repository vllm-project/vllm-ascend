"""Project semantic KV chunks onto existing Worker-local Block IDs."""

from __future__ import annotations

from collections.abc import Sequence

from ..elements import KVBlockAssignment, KVBlockAssignmentBatch, KVChunkBatch
from ..topology import KVPoolTopology


class KVBlockProjection:
    """Align semantic chunks with the Block IDs supplied for one scheduler step."""

    def __init__(self, topology: KVPoolTopology) -> None:
        self.group_ids = topology.transfer_group_ids
        self._minimum_block_ids = {
            group.group_id: 0 if topology.tp_partition.tp_mismatch or not group.uses_align_state else 1
            for group in topology.groups
            if group.group_id in topology.transfer_group_ids
        }

    def project(
        self,
        batches: tuple[KVChunkBatch, ...],
        block_ids_by_group: tuple[tuple[int, ...], ...],
    ) -> tuple[KVBlockAssignmentBatch, ...]:
        group_ids = tuple(batch.group_id for batch in batches)
        if group_ids != self.group_ids:
            raise ValueError(f"KV chunk groups {group_ids} do not match compiled groups {self.group_ids}")
        return tuple(self._project_batch(batch, block_ids_by_group[batch.group_id]) for batch in batches)

    def _project_batch(self, batch: KVChunkBatch, block_ids: Sequence[int]) -> KVBlockAssignmentBatch:
        block_offset = max(batch.logical_block_count - len(block_ids), 0)
        assignments = []
        minimum_block_id = self._minimum_block_ids[batch.group_id]
        for chunk in batch.chunks:
            local_block_index = chunk.block_index - block_offset
            if 0 <= local_block_index < len(block_ids):
                block_id = block_ids[local_block_index]
                if block_id >= minimum_block_id:
                    assignments.append(KVBlockAssignment(chunk, block_id))
        return KVBlockAssignmentBatch(batch.group_id, tuple(assignments))
