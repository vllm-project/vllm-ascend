# SPDX-License-Identifier: Apache-2.0
"""Common P/D and shared-snapshot byte reader over the existing transport."""

from __future__ import annotations

from .protocol import STAGE_BYTES, validate_wire
from .transport import prepare_stage, rpc


class StagedTransfer:
    """SOURCE uses producer HBM staging; INDEXER key/scale use direct reads.

    read_sync is the backend's real drained MemFabric read. This object does
    not allocate or restore Resident/Hot pages, graph metadata or MTP state.
    Uncertain prepare/read/release exceptions propagate to ImportSession,
    which retains destination owners and forbids acknowledging the root lease.
    """

    def __init__(self, endpoint, message, manifest, read_sync, *, timeout=120, max_descriptors=1024):
        validate_wire(manifest)
        if (
            manifest.get("stage_capacity") != STAGE_BYTES
            or type(manifest.get("stage_address")) is not int
            or not 0 < manifest["stage_address"] < 2**64 - STAGE_BYTES
        ):
            raise ValueError("Acquired producer must declare its real bounded HBM stage")
        if not 0 < timeout <= 3600 or type(max_descriptors) is not int or not 0 < max_descriptors <= 65536:
            raise ValueError("Invalid bounded transport limits")
        self.endpoint, self.message, self.manifest = endpoint, dict(message), manifest
        self.read_sync, self.timeout, self.max_descriptors = read_sync, timeout, max_descriptors

    def copy(self, session):
        if session.manifest != self.manifest:
            raise ValueError("Read session differs from its acquired immutable manifest")
        session.copy(lambda _: self._copy(session.role_descriptors))

    def _copy(self, descriptors):
        pending = []
        for index, (destination, source, length, role, layer_key) in enumerate(descriptors):
            if role == "hisparse_source":
                if pending:
                    self.read_sync(self.manifest["session"], pending)
                    pending = []
                self._source(index, layer_key[0], destination, source, length)
            else:
                pending.append((destination, source, length))
                if len(pending) == self.max_descriptors:
                    self.read_sync(self.manifest["session"], pending)
                    pending = []
        if pending:
            self.read_sync(self.manifest["session"], pending)

    def _source(self, index, layer, destination, source, length):
        for offset in range(0, length, STAGE_BYTES):
            count = min(length - offset, STAGE_BYTES)
            chunk = dict(self.message, chunk_id=f"{index}:{offset}", layer=layer, source=source + offset, length=count)
            prepared = prepare_stage(self.endpoint, chunk, self.timeout)
            if prepared["stage_address"] != self.manifest["stage_address"]:
                raise ValueError("Prepared producer stage differs from the acquired allocation")
            self.read_sync(self.manifest["session"], [(destination + offset, prepared["stage_address"], count)])
            response = rpc(
                self.endpoint,
                dict(chunk, op="stage_release", generation=prepared["generation"], drained=True),
                self.timeout,
            )
            if (
                response.get("status") != "stage_released"
                or response.get("generation") != prepared["generation"]
                or response.get("chunk_id") != chunk["chunk_id"]
            ):
                raise ValueError("Stage release remains unconfirmed; retain source ownership")

    def acknowledge(self):
        result = rpc(self.endpoint, dict(self.message, op="release", drained=True), self.timeout)
        if result.get("status") != "released":
            raise ValueError("Source lease release remains unconfirmed")
