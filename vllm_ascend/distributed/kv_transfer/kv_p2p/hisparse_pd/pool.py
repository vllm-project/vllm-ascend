# SPDX-License-Identifier: Apache-2.0
"""Producer-owned immutable snapshots shared through the P/D read protocol.

This catalog retains upstream block owners; it allocates no KV blocks. It is
not an AscendStore persistence adapter. The producer must stay alive until the
snapshot is retired and every acquired read is known drained.
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass

from .protocol import LayerLayout, digest, plan_reads, validate_layouts, validate_source_range, validate_wire
from .topology import Replica


@dataclass(frozen=True)
class PoolKey:
    model_identity: str
    layout_digest: str
    prefix_hash: str
    num_tokens: int
    tp_rank: int
    tp_size: int

    def __post_init__(self):
        if not all(isinstance(x, str) and x for x in (self.model_identity, self.layout_digest, self.prefix_hash)):
            raise ValueError("Pool identity requires model, byte layout and upstream prefix hash")
        if any(type(x) is not int for x in (self.num_tokens, self.tp_rank, self.tp_size)):
            raise ValueError("Pool token and TP geometry must be integer")
        if self.num_tokens <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("Invalid pool token or TP geometry")

    @classmethod
    def from_layouts(cls, identity, prefix_hash, num_tokens, rank, tp_size, layouts):
        validate_layouts(layouts)
        return cls(
            identity, digest(sorted(layer.signature() for layer in layouts)), prefix_hash, num_tokens, rank, tp_size
        )


class SharedSnapshotPool:
    """Multi-reader implementation of the P/D acquire/release control API.

    Each instance belongs to one producer DP and TP worker. Producer DP is
    deliberately absent from PoolKey: consumers may discover a matching shard
    on any configured producer replica. Reader IDs must include engine, DP,
    request, TP and an attempt nonce, rather than a bare request ID.
    """

    def __init__(self, producer: Replica, *, tp_rank=0, max_snapshots=128, max_readers_per_snapshot=4096):
        if max_snapshots <= 0 or max_readers_per_snapshot <= 0:
            raise ValueError("Pool bounds must be positive")
        self.producer = producer
        if type(tp_rank) is not int or not 0 <= tp_rank < producer.tp_size:
            raise ValueError("Invalid owning TP rank")
        self.tp_rank = tp_rank
        self.max_snapshots = max_snapshots
        self.max_readers = max_readers_per_snapshot
        self._lock = threading.RLock()
        self._stage_claim = None
        self._entries = {}
        self._keys = {}
        self._finished = set()
        self._released = {}
        self._closing = False

    def publish_snapshot(self, key, request_id, manifest, *, owners, ready):
        """Retain owners now; expose a hit only after the real completion fence."""
        validate_wire(manifest)
        layouts = [LayerLayout(**value) for value in manifest["layers"]]
        expected = PoolKey.from_layouts(
            manifest["identity"],
            key.prefix_hash,
            manifest["ticket"]["num_tokens"],
            manifest["rank"],
            manifest["world_size"],
            layouts,
        )
        if expected != key or key.tp_size != self.producer.tp_size or key.tp_rank != self.tp_rank:
            raise ValueError("Pool namespace or shard geometry differs from the manifest")
        if manifest.get("producer") != self.producer.ticket_identity():
            raise ValueError("Snapshot belongs to another producer DP")
        if any(layer.model_role != "target" for layer in layouts):
            raise ValueError("Draft snapshot import requires a qualified MTP restoration policy")
        blocks = {int(group): values for group, values in manifest["blocks"].items()}
        plan_reads(layouts, layouts, blocks, blocks, 0, key.num_tokens)
        if not request_id or not owners or not callable(ready):
            raise ValueError("Snapshot needs its upstream request, strong owners and completion fence")
        lease = manifest["ticket"]["lease_id"]
        if not isinstance(lease, str) or not lease:
            raise ValueError("Snapshot needs an immutable lease generation")
        with self._lock:
            self._prune_released()
            if self._closing or key in self._keys or lease in self._entries or lease in self._released:
                raise ValueError("Pool is closing or snapshot already exists")
            if len(self._entries) >= self.max_snapshots:
                raise RuntimeError("Retire a snapshot before exceeding the pool bound")
            self._entries[lease] = dict(
                key=key,
                request_id=request_id,
                manifest=deepcopy(manifest),
                owners=tuple(owners),
                ready=ready,
                readers=set(),
                acknowledged=set(),
                stage_positions={},
                retired=False,
            )
            self._keys[key] = lease

    def lookup(self, key):
        """Return discovery metadata, never unleased byte pointers."""
        with self._lock:
            entry = self._entries.get(self._keys.get(key))
            if entry is None or self._closing or entry["retired"] or entry["ready"]() is not True:
                return None
            return dict(lease_id=entry["manifest"]["ticket"]["lease_id"], producer=self.producer.ticket_identity())

    def acquire(self, lease_id, consumer, identity, rank, world_size):
        with self._lock:
            entry = self._entries.get(lease_id)
            if (
                entry is None
                or self._closing
                or (entry["retired"] and consumer not in entry["readers"])
                or entry["ready"]() is not True
            ):
                return {"status": "pending"}
            manifest = entry["manifest"]
            if (identity, rank, world_size) != (manifest["identity"], manifest["rank"], manifest["world_size"]):
                raise ValueError("Model or TP shard mismatch")
            if not isinstance(consumer, str) or not consumer or consumer in entry["acknowledged"]:
                raise ValueError("Reader identity is empty or already completed")
            if consumer not in entry["readers"] and len(entry["readers"] | entry["acknowledged"]) >= self.max_readers:
                raise RuntimeError("Retire this generation before exceeding its reader bound")
            entry["readers"].add(consumer)
            return {"status": "ready", "manifest": deepcopy(manifest)}

    def lookup_wire(self, message):
        validate_wire(message)
        found = self.lookup(PoolKey(**message["key"]))
        return {"status": "ready", **found} if found else {"status": "miss"}

    def release(self, lease_id, consumer, *, drained):
        with self._lock:
            self._prune_released()
            entry = self._entries.get(lease_id)
            if entry is None:
                if drained is True and consumer in self._released.get(lease_id, ((), 0))[0]:
                    return
                raise ValueError("Unknown snapshot generation")
            if drained is not True:
                raise ValueError("A timeout or cancellation does not prove DMA drained")
            if consumer in entry["acknowledged"]:
                return
            if consumer not in entry["readers"]:
                raise ValueError("Only an acquired reader may release")
            if self._stage_claim == (lease_id, consumer):
                raise ValueError("Producer stage is outstanding; snapshot reader remains retained")
            entry["readers"].remove(consumer)
            entry["acknowledged"].add(consumer)

    def validate_source_claim(self, message):
        with self._lock:
            entry = self._entries.get(message["lease_id"])
            if entry is None or message["consumer"] not in entry["readers"]:
                raise ValueError("Stage requires an acquired snapshot reader")
            validate_source_range(entry["manifest"], message)

    def stage_position(self, lease_id, consumer):
        with self._lock:
            entry = self._entries[lease_id]
            if consumer not in entry["readers"]:
                raise ValueError("Stage requires an acquired snapshot reader")
            return entry["stage_positions"].get(consumer)

    def record_stage_release(self, lease_id, consumer, position):
        with self._lock:
            if self._stage_claim != (lease_id, consumer):
                raise ValueError("Only the owning stage may advance its reader")
            self._entries[lease_id]["stage_positions"][consumer] = position

    def active(self):
        with self._lock:
            return sum(len(entry["readers"]) for entry in self._entries.values())

    def retire(self, lease_id):
        with self._lock:
            entry = self._entries[lease_id]
            entry["retired"] = True
            if self._keys.get(entry["key"]) == lease_id:
                del self._keys[entry["key"]]

    def take_finished(self):
        """Complete the original request only after eviction and all read ACKs."""
        with self._lock:
            for lease, entry in list(self._entries.items()):
                if (
                    entry["retired"]
                    and not entry["readers"]
                    and (self._stage_claim is None or self._stage_claim[0] != lease)
                    and entry["ready"]() is True
                ):
                    self._released[lease] = frozenset(entry["acknowledged"]), time.monotonic() + 600
                    self._finished.add(entry["request_id"])
                    del self._entries[lease]
            self._prune_released()
            retained_requests = {entry["request_id"] for entry in self._entries.values()}
            result = self._finished - retained_requests
            self._finished.intersection_update(retained_requests)
            return result

    def begin_shutdown(self):
        with self._lock:
            if self._stage_claim is not None or any(
                entry["readers"] or entry["ready"]() is not True for entry in self._entries.values()
            ):
                raise RuntimeError("Keep source owners and control server alive until all DMA drains")
            self._closing = True
            for lease in list(self._entries):
                self.retire(lease)

    def _prune_released(self):
        for lease, (_, expiry) in list(self._released.items()):
            if expiry <= time.monotonic():
                del self._released[lease]
        while len(self._released) > 4096:
            del self._released[next(iter(self._released))]
