"""Resolve semantic chunks to their request-local KV cache Blocks."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol.transfer import CheckpointStoreCommand
from ..representation import KVBlockAssignment, KVBlockAssignmentBatch, KVChunkBatch
from ..spec.topology import KVPoolTopology


class LocalBlockResolution:
    """Resolve semantic chunks to the local Block IDs that currently contain them."""

    def __init__(
        self,
        group_ids: tuple[int, ...],
        minimum_block_ids: dict[int, int],
        block_sizes: dict[int, int],
    ) -> None:
        self.group_ids = group_ids
        self._minimum_block_ids = minimum_block_ids
        self._block_sizes = block_sizes

    def resolve(
        self,
        batches: tuple[KVChunkBatch, ...],
        block_ids_by_group: tuple[tuple[int, ...], ...],
    ) -> tuple[KVBlockAssignmentBatch, ...]:
        group_ids = tuple(batch.group_id for batch in batches)
        if group_ids != self.group_ids:
            raise ValueError(f"KV chunk groups {group_ids} do not match compiled groups {self.group_ids}")
        return tuple(self.resolve_batch(batch, block_ids_by_group[batch.group_id]) for batch in batches)

    def resolve_batch(self, batch: KVChunkBatch, block_ids: Sequence[int]) -> KVBlockAssignmentBatch:
        if batch.group_id not in self._minimum_block_ids:
            raise ValueError(f"Unknown transferable cache group {batch.group_id}")
        block_offset = max(batch.logical_block_count - len(block_ids), 0)
        assignments = []
        minimum_block_id = self._minimum_block_ids[batch.group_id]
        for chunk in batch.chunks:
            local_block_index = chunk.block_index - block_offset
            if 0 <= local_block_index < len(block_ids):
                block_id = block_ids[local_block_index]
                if block_id >= minimum_block_id:
                    assignments.append(KVBlockAssignment(chunk, block_id, self._block_sizes[batch.group_id]))
        return KVBlockAssignmentBatch(batch.group_id, tuple(assignments))


class CheckpointBlockResolution:
    """Resolve Mamba checkpoints from exact sources and companion chunks from Block Tables."""

    def __init__(
        self,
        local_blocks: LocalBlockResolution,
        minimum_block_ids: dict[int, int],
        block_sizes: dict[int, int],
    ) -> None:
        self._local_blocks = local_blocks
        self._minimum_block_ids = minimum_block_ids
        self._block_sizes = block_sizes

    def resolve(
        self,
        batches: tuple[KVChunkBatch, ...],
        command: CheckpointStoreCommand,
    ) -> tuple[KVBlockAssignmentBatch, ...]:
        checkpoint_batches = batches[: len(command.sources)]
        companion_batches = batches[len(command.sources) :]
        assignments = tuple(
            self._resolve_checkpoint(batch, source.group_id, source.block_id)
            for batch, source in zip(checkpoint_batches, command.sources, strict=True)
        )
        companions = tuple(
            self._with_full_block_extent(
                self._local_blocks.resolve_batch(batch, command.block_ids_by_group[batch.group_id])
            )
            for batch in companion_batches
        )
        return assignments + companions

    def _resolve_checkpoint(self, batch: KVChunkBatch, group_id: int, block_id: int) -> KVBlockAssignmentBatch:
        if batch.group_id != group_id or len(batch.chunks) != 1:
            raise ValueError(f"Checkpoint source for group {group_id} does not match its semantic chunk")
        if block_id < self._minimum_block_ids[group_id]:
            return KVBlockAssignmentBatch(group_id, ())
        return KVBlockAssignmentBatch(
            group_id,
            (KVBlockAssignment(batch.chunks[0], block_id, self._block_sizes[group_id]),),
        )

    def _with_full_block_extent(self, batch: KVBlockAssignmentBatch) -> KVBlockAssignmentBatch:
        block_size = self._block_sizes[batch.group_id]
        assignments = tuple(
            KVBlockAssignment(assignment.chunk, assignment.block_id, block_size) for assignment in batch.assignments
        )
        return KVBlockAssignmentBatch(batch.group_id, assignments)


def compile_block_resolutions(topology: KVPoolTopology) -> tuple[LocalBlockResolution, CheckpointBlockResolution]:
    """Compile normal and checkpoint Block rules from the static group topology."""

    transfer_groups = tuple(group for group in topology.groups if group.group_id in topology.transfer_group_ids)
    minimum_block_ids = {
        group.group_id: 0 if topology.tp_partition.tp_mismatch or not group.uses_align_state else 1
        for group in transfer_groups
    }
    block_sizes = {group.group_id: group.block_size for group in transfer_groups}
    local_blocks = LocalBlockResolution(topology.transfer_group_ids, minimum_block_ids, block_sizes)
    return local_blocks, CheckpointBlockResolution(local_blocks, minimum_block_ids, block_sizes)
