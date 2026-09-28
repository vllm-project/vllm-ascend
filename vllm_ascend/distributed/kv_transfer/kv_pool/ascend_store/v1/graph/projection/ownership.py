"""Select Store block assignments owned by this KV Pool participant."""

from __future__ import annotations

from dataclasses import dataclass

from ..elements import KVBlockAssignment, KVBlockAssignmentBatch
from ..topology import KVPoolGroupTopology, KVPoolTopology


@dataclass(frozen=True, slots=True)
class _StoreOwnership:
    tp_rank: int
    pcp_rank: int
    pcp_size: int
    replica_count: int

    @classmethod
    def compile(cls, topology: KVPoolTopology, group: KVPoolGroupTopology) -> _StoreOwnership:
        replica_count = topology.put_step
        if topology.tp_partition.tp_mismatch or topology.dcp_size > 1 or group.uses_align_state:
            replica_count = 1
        return cls(topology.tp_rank, topology.pcp_rank, topology.pcp_size, replica_count)

    def select(self, assignments: tuple[KVBlockAssignment, ...]) -> tuple[KVBlockAssignment, ...]:
        shard_rank = self.pcp_rank * self.replica_count + self.tp_rank % self.replica_count
        shard_count = self.pcp_size * self.replica_count
        if shard_count <= 1:
            return assignments
        return tuple(item for index, item in enumerate(assignments) if index % shard_count == shard_rank)


class StoreOwnershipProjection:
    """Project Store assignments onto the subset owned by this participant."""

    def __init__(self, topology: KVPoolTopology) -> None:
        groups_by_id = {group.group_id: group for group in topology.groups}
        self.group_ids = topology.transfer_group_ids
        self._ownership = {
            group_id: _StoreOwnership.compile(topology, groups_by_id[group_id]) for group_id in self.group_ids
        }

    def project(self, batches: tuple[KVBlockAssignmentBatch, ...]) -> tuple[KVBlockAssignmentBatch, ...]:
        group_ids = tuple(batch.group_id for batch in batches)
        if group_ids != self.group_ids:
            raise ValueError(f"KV block assignment groups {group_ids} do not match compiled groups {self.group_ids}")
        return tuple(
            KVBlockAssignmentBatch(batch.group_id, self._ownership[batch.group_id].select(batch.assignments))
            for batch in batches
        )
