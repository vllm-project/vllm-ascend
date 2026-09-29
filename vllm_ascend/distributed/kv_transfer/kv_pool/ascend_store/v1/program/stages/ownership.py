"""Select the Store assignments owned by this participant."""

from __future__ import annotations

from dataclasses import dataclass

from ..representation import KVBlockAssignment, KVBlockAssignmentBatch
from ..spec.topology import KVPoolTopology


@dataclass(frozen=True, slots=True)
class _GroupStoreOwnership:
    shard_rank: int
    shard_count: int

    def select(self, assignments: tuple[KVBlockAssignment, ...]) -> tuple[KVBlockAssignment, ...]:
        if self.shard_count <= 1:
            return assignments
        return tuple(item for index, item in enumerate(assignments) if index % self.shard_count == self.shard_rank)


class StoreOwnershipSelection:
    """Select Store assignments owned by this participant before physical fanout."""

    def __init__(
        self,
        group_ids: tuple[int, ...],
        ownership_by_group: dict[int, _GroupStoreOwnership],
    ) -> None:
        self.group_ids = group_ids
        self._ownership_by_group = ownership_by_group

    def select(self, batches: tuple[KVBlockAssignmentBatch, ...]) -> tuple[KVBlockAssignmentBatch, ...]:
        group_ids = tuple(batch.group_id for batch in batches)
        if group_ids != self.group_ids:
            raise ValueError(f"KV block assignment groups {group_ids} do not match compiled groups {self.group_ids}")
        return tuple(self.select_batch(batch) for batch in batches)

    def select_batch(self, batch: KVBlockAssignmentBatch) -> KVBlockAssignmentBatch:
        try:
            ownership = self._ownership_by_group[batch.group_id]
        except KeyError as error:
            raise ValueError(f"Unknown transferable cache group {batch.group_id}") from error
        return KVBlockAssignmentBatch(batch.group_id, ownership.select(batch.assignments))


def compile_store_ownership(
    topology: KVPoolTopology,
    align_state_group_ids: frozenset[int],
) -> StoreOwnershipSelection:
    """Compile each group's Store replication rule into one deterministic shard selection."""

    ownership_by_group = {
        group_id: _compile_group_ownership(topology, group_id in align_state_group_ids)
        for group_id in topology.transfer_group_ids
    }
    return StoreOwnershipSelection(topology.transfer_group_ids, ownership_by_group)


def _compile_group_ownership(
    topology: KVPoolTopology,
    uses_align_state: bool,
) -> _GroupStoreOwnership:
    replica_count = topology.put_step
    if topology.tp_partition.tp_mismatch or topology.dcp_size > 1 or uses_align_state:
        replica_count = 1
    return _GroupStoreOwnership(
        shard_rank=topology.pcp_rank * replica_count + topology.tp_rank % replica_count,
        shard_count=topology.pcp_size * replica_count,
    )
