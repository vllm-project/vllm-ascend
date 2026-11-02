# SPDX-License-Identifier: Apache-2.0
"""A received snapshot becomes runnable after local state restoration completes."""

from __future__ import annotations

from copy import deepcopy

from .protocol import LayerLayout, coalesce_transfers, plan_reads, validate_wire
from .topology import Replica

_retained_imports = {}


class ImportSession:
    """Shared P/D and pool import lifecycle; no scheduler block allocation.

    read_sync must return only after its reads drain. An exception is uncertain,
    even for an API named synchronous. restore_local must update only the given
    DP/request's Resident/Hot/LRU and bind its current graph metadata; it returns
    a completion query for those writes. Target+MTP restoration is not supplied
    by this control module and must not be inferred from a successful byte copy.
    """

    def __init__(
        self,
        manifest,
        destination,
        destination_blocks,
        start_token,
        *,
        replica: Replica,
        tp_rank,
        request_id,
        owners,
        identity,
    ):
        validate_wire(manifest)
        manifest = deepcopy(manifest)
        if not identity or manifest["identity"] != identity:
            raise ValueError("Import model identity differs from the local checkpoint/configuration")
        if type(tp_rank) is not int or not 0 <= tp_rank < replica.tp_size:
            raise ValueError("Invalid local TP shard")
        producer = Replica(**manifest["producer"])
        if (
            producer.tp_size != replica.tp_size
            or manifest["world_size"] != replica.tp_size
            or manifest["rank"] != tp_rank
        ):
            raise ValueError("Import cannot reshard TP or substitute the consumer DP for the producer DP")
        if not request_id or not owners:
            raise ValueError("Import needs its local request identity and strong destination owners")
        source = [LayerLayout(**x) for x in manifest["layers"]]
        if any(x.model_role != "target" for x in source):
            raise ValueError("MTP import requires a qualified draft state restorer")
        source_blocks = {int(k): list(v) for k, v in manifest["blocks"].items()}
        self.role_descriptors = tuple(
            coalesce_transfers(
                plan_reads(
                    source,
                    destination,
                    source_blocks,
                    destination_blocks,
                    start_token,
                    manifest["ticket"]["num_tokens"],
                    with_roles=True,
                )
            )
        )
        self.descriptors = tuple(item[:3] for item in self.role_descriptors)
        self.manifest = manifest
        self.request_key = replica.request_key(request_id)
        self.owners = tuple(owners)
        self.state = "acquired"
        self.source_released = False
        self._ready = None

    def copy(self, read_sync):
        if self.state != "acquired":
            raise RuntimeError("Snapshot copy cannot be repeated or follow cancellation")
        self.state = "reading"
        # A caller dropping its failed Future must not free partially written
        # Host/graph owners. Only close after explicit drain can remove this pin.
        _retained_imports[id(self)] = self
        try:
            read_sync(self.descriptors)
        except BaseException:
            self.state = "uncertain"
            raise
        self.state = "copied"

    def restore(self, restore_local):
        if self.state != "copied":
            raise RuntimeError("Restore requires a complete, uncancelled byte copy")
        self.state = "restoring"
        try:
            ready = restore_local(self.request_key)
            if not callable(ready):
                raise ValueError("Local restoration must return its actual completion query")
            self._ready = ready
        except BaseException:
            self.state = "restore_uncertain"
            raise

    def runnable(self):
        if self.state == "restoring" and self._ready() is True:
            self.state = "ready"
        return self.state == "ready"

    def cancel(self):
        if self.state in ("reading", "uncertain", "restore_uncertain", "closed"):
            raise RuntimeError("An outstanding read must be drained before cancellation can release owners")
        if self.state == "restoring" and self._ready() is not True:
            raise RuntimeError("Local graph/cache writes must drain before cancellation")
        self.state = "cancelled"

    def confirm_failed_read_drained(self, drain_transport):
        if self.state != "uncertain" or drain_transport() is not True:
            raise RuntimeError("Transport teardown must explicitly confirm outstanding DMA drained")
        self.state = "cancelled"

    def confirm_failed_restore_drained(self, drain_local):
        if self.state != "restore_uncertain" or drain_local() is not True:
            raise RuntimeError("Local teardown must confirm partial graph/cache restoration drained")
        self.state = "cancelled"

    def release_source(self, acknowledge):
        if self.state in ("acquired", "reading", "uncertain"):
            raise RuntimeError("Cannot ACK an uncertain DMA")
        if not self.source_released:
            acknowledge()
            self.source_released = True

    def close(self):
        if self.state not in ("cancelled", "ready") or not self.source_released:
            raise RuntimeError("Keep destination owners until source ACK and local work complete")
        self.owners = ()
        self.state = "closed"
        _retained_imports.pop(id(self), None)
