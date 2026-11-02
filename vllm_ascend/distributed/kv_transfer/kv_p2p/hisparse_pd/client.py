# SPDX-License-Identifier: Apache-2.0
"""DP-aware discovery/acquisition shared by P/D tickets and pooled prefixes."""

from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from dataclasses import asdict

from .protocol import SOURCE_TRANSPORT, VERSION, digest, validate_wire
from .transport import acquire, rpc


class SnapshotClient:
    """Resolve only configured producers; preserve their existing TP shards."""

    def __init__(self, directory, replica, tp_rank, identity, *, timeout=120):
        if type(tp_rank) is not int or not 0 <= tp_rank < replica.tp_size or not identity:
            raise ValueError("Client requires an explicit local TP shard and model identity")
        if not 0 < timeout <= 3600:
            raise ValueError("Invalid control timeout")
        self.directory, self.replica, self.tp_rank = directory, replica, tp_rank
        self.identity, self.timeout = identity, timeout
        self._lock = threading.RLock()
        self._pending_claims = {}

    @property
    def pending_claims(self):
        with self._lock:
            return deepcopy(self._pending_claims)

    def lookup(self, producer, key):
        if (key.model_identity, key.tp_rank, key.tp_size) != (self.identity, self.tp_rank, self.replica.tp_size):
            raise ValueError("Pool lookup must use this client's model and TP shard")
        endpoint = self.directory.resolve(producer, self.replica, self.tp_rank)
        result = rpc(
            endpoint,
            dict(version=VERSION, source_transport=SOURCE_TRANSPORT, op="lookup", key=asdict(key)),
            self.timeout,
        )
        if result.get("status") == "miss":
            return None
        if result.get("status") != "ready" or result.get("producer") != producer:
            raise ValueError("Snapshot discovery returned a different producer")
        return result

    def acquire(self, producer, lease_id, request_id, cancelled):
        endpoint = self.directory.resolve(producer, self.replica, self.tp_rank)
        consumer = digest((self.replica.request_key(request_id), self.tp_rank, uuid.uuid4().hex))
        message = dict(
            version=VERSION,
            source_transport=SOURCE_TRANSPORT,
            lease_id=lease_id,
            consumer=consumer,
            identity=self.identity,
            rank=self.tp_rank,
            world_size=self.replica.tp_size,
        )
        with self._lock:
            if len(self._pending_claims) >= 4096:
                raise RuntimeError("Reconcile uncertain acquisitions before opening additional readers")
            self._pending_claims[consumer] = (endpoint, message, "acquiring")
        # Any lost acquire reply can still own a source reader. Errors retain
        # that claim; the engine must reconcile it before producer teardown.
        try:
            manifest = acquire(endpoint, message, self.timeout, cancelled)
            validate_wire(manifest)
            if (
                manifest.get("producer") != producer
                or manifest["ticket"]["lease_id"] != lease_id
                or (manifest["identity"], manifest["rank"], manifest["world_size"])
                != (self.identity, self.tp_rank, self.replica.tp_size)
            ):
                raise ValueError("Acquired manifest differs from its configured producer/model/TP shard")
        except BaseException:
            with self._lock:
                self._pending_claims[consumer] = (endpoint, message, "failed")
            raise
        with self._lock:
            del self._pending_claims[consumer]
        return endpoint, message, manifest

    def reconcile_failed_acquire(self, consumer):
        """No data read was exposed for these attempts; explicitly reconcile ACKs."""
        with self._lock:
            endpoint, message, state = self._pending_claims[consumer]
            if state != "failed":
                raise RuntimeError("An acquisition still running cannot be reconciled")
            self._pending_claims[consumer] = endpoint, message, "reconciling"
        try:
            result = rpc(endpoint, dict(message, op="release", drained=True), self.timeout)
            if result.get("status") != "released":
                raise ValueError("Failed acquisition's source ownership remains unconfirmed")
        except BaseException:
            with self._lock:
                self._pending_claims[consumer] = endpoint, message, "failed"
            raise
        with self._lock:
            del self._pending_claims[consumer]
