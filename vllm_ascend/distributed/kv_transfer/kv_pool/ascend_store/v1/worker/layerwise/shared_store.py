"""Require all elected context-shard writers before publishing one shared GVA key."""

from __future__ import annotations

from dataclasses import dataclass

from ...infrastructure.configuration.topology import KVPoolTopology
from ...infrastructure.execution.collective import StoreCollective


@dataclass(frozen=True, slots=True)
class GVAWriteOutcome:
    # A normal copy completion is local evidence, not whole-object readiness.
    completed: frozenset[str]
    regions: dict[str, tuple[int, int]]


class SharedGVAWrites:
    """Worker policy over opaque object keys, separate from byte geometry and leases.

    Every TP rank participates once per Store, even a non-writer or failed opener.
    Only the elected ranks for this head contribute required shards. Failure or a
    missing shard prevents publication; no majority or first-writer rule is valid.
    """

    def __init__(self, topology: KVPoolTopology, collective: StoreCollective) -> None:
        self._collective = collective
        first_rank = (topology.tp_rank // topology.put_step) * topology.put_step
        leaders: dict[int, int] = {}
        for rank in range(first_rank, first_rank + topology.put_step):
            leaders.setdefault(rank % topology.dcp_size, rank)
        self._writers = tuple(leaders.values())

    def complete_keys(self, local: GVAWriteOutcome) -> set[str]:
        outcomes = self._collective.all_gather(local)
        writers = tuple(outcomes[rank] for rank in self._writers)
        complete = set(local.completed)
        for writer in writers:
            # A peer's earlier exists hit can refer to an evicted generation.
            # It cannot certify a shard in this newly allocated object.
            complete.intersection_update(writer.completed)
        # GVA is globally addressed: matching keys alone cannot prove that an
        # expired/reallocated object received every writer's bytes.
        return {
            key for key in complete if len({writer.regions[key] for writer in writers if key in writer.regions}) == 1
        }
