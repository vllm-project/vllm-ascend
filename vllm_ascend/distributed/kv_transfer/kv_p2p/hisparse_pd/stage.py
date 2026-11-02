# SPDX-License-Identifier: Apache-2.0
"""One explicitly leased producer HBM slot; no timeout can recycle it."""

from __future__ import annotations

from .protocol import SOURCE_TRANSPORT, STAGE_BYTES, VERSION, LeaseRegistry


class StageSlot:
    def __init__(self, registry: LeaseRegistry, address: int, capacity: int, copy_source, copy_packed=None):
        if type(address) is not int or address <= 0 or type(capacity) is not int or capacity != STAGE_BYTES:
            raise ValueError("Producer staging requires one real 16 MiB HBM allocation")
        if address + capacity >= 2**64 or not callable(copy_source):
            raise ValueError("Invalid producer stage allocation or copy callback")
        self.registry = registry
        self.address, self.capacity = address, capacity
        self.copy_source = copy_source
        self.copy_packed = copy_packed
        self._lock = registry._lock
        self._current = None
        self._generation = 0
        self._released = {}
        self._closing = False

    @staticmethod
    def _key(message):
        chunk = message.get("chunk_id")
        if not isinstance(chunk, str) or not 0 < len(chunk) <= 128:
            raise ValueError("Invalid producer stage chunk identity")
        return tuple(
            message[key]
            for key in (
                "lease_id",
                "consumer",
                "identity",
                "rank",
                "world_size",
                "layer",
                "source",
                "length",
            )
        ) + (tuple(tuple(pair) for pair in message.get("segments", [])), chunk)

    @staticmethod
    def _position(chunk):
        pieces = chunk.split(":")
        if len(pieces) != 2 or any(not item.isascii() or not item.isdecimal() for item in pieces):
            raise ValueError("Stage chunk identity must be descriptor_index:byte_offset")
        position = tuple(int(item) for item in pieces)
        if any(str(value) != text for value, text in zip(position, pieces)) or position[1] % STAGE_BYTES:
            raise ValueError("Stage chunk identity is not canonical")
        return position

    def _validate_source_claim(self, message):
        segments = message.get("segments")
        if "segments" not in message:
            self._validate_source_claim(message)
            return
        if not callable(self.copy_packed) or not isinstance(segments, list) or not 1 <= len(segments) <= 256:
            raise ValueError("Packed stage requires 1..256 explicit SOURCE segments")
        if type(message.get("source")) is not int or type(message.get("length")) is not int:
            raise ValueError("Packed stage addresses and lengths must be integers")
        total = 0
        for pair in segments:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Invalid packed SOURCE segment")
            source, length = pair
            self.registry.validate_source_claim({**message, "source": source, "length": length})
            total += length
        if total != message["length"] or total > STAGE_BYTES or message["source"] != segments[0][0]:
            raise ValueError("Packed SOURCE byte count differs from its bounded slot")

    def prepare(self, message):
        with self._lock:
            if self._closing:
                raise ValueError("Producer stage is closing")
            self._validate_source_claim(message)
            return self._prepare_validated(message)

    def _prepare_validated(self, message):
        # Caller holds the registry lock and has validated all source extents.
        key = self._key(message)
        position = self._position(key[-1])
        if self._current is not None:
            if self._current["key"] != key:
                if self._current["key"][-1] == key[-1] and self._current["key"][:2] == key[:2]:
                    raise ValueError("Stage chunk identity was reused for different bytes")
                return {"status": "pending"}
            if self._current["state"] != "ready":
                raise ValueError("Producer stage copy is uncertain; slot remains retained")
            return dict(self._current["response"])
        previous = self.registry.stage_position(key[0], key[1])
        if previous is not None and position <= previous:
            raise ValueError("Stage chunk has already been released")
        self._generation += 1
        response = {
            "status": "ready",
            "version": VERSION,
            "source_transport": SOURCE_TRANSPORT,
            "stage_address": self.address,
            "capacity": self.capacity,
            "generation": self._generation,
            "chunk_id": key[-1],
            "length": message["length"],
        }
        self.registry._stage_claim = key[:2]
        self._current = {"key": key, "state": "copying", "response": response}
        try:
            # Reserve before copying. The callback sets the control thread's
            # device/stream and synchronizes H2D before this reply is sent.
            if "segments" in message:
                self.copy_packed(message["layer"], message["segments"])
            else:
                self.copy_source(message["layer"], message["source"], message["length"])
        except BaseException as error:
            self._current["state"] = "poisoned"
            raise ValueError("Producer stage copy failed; slot and source lease remain retained") from error
        self._current["state"] = "ready"
        return dict(response)

    def release(self, message):
        with self._lock:
            # Exact immutable chunk keys and generations retain the validated
            # range. Check lease ownership without rescanning every fragment.
            self.registry.stage_position(message["lease_id"], message["consumer"])
            key = self._key(message)
            generation = message.get("generation")
            if type(generation) is not int or message.get("drained") is not True:
                raise ValueError("Stage release requires an exact generation and drained read")
            chunk_identity = key[0], key[1], key[-1]
            if self._released.get(chunk_identity) == (key, generation):
                return {"status": "stage_released", "generation": generation, "chunk_id": key[-1]}
            if (
                self._current is None
                or self._current["key"] != key
                or self._current["response"]["generation"] != generation
                or self._current["state"] != "ready"
            ):
                raise ValueError("Only this stage generation's owning drained consumer may release it")
            self._released[chunk_identity] = key, generation
            # A constant-size per-reader high-water mark forbids reuse even after
            # the bounded release-reply history below has been pruned.
            self.registry.record_stage_release(key[0], key[1], self._position(key[-1]))
            while len(self._released) > 4096:
                del self._released[next(iter(self._released))]
            self._current = None
            self.registry._stage_claim = None
            return {"status": "stage_released", "generation": generation, "chunk_id": key[-1]}

    def advance(self, message):
        """Release a completed generation and prepare its successor atomically.

        Both operations retain their existing checks. Validate the successor
        before mutating the old slot, so a malformed range cannot release it.
        A lost reply never permits another consumer to overwrite the new slot.
        """
        with self._lock:
            if self._closing:
                raise ValueError("Producer stage is closing")
            previous, following = message["released"], message["next"]
            for name in ("lease_id", "consumer", "identity", "rank", "world_size"):
                if previous[name] != following[name] or message[name] != previous[name]:
                    raise ValueError("Stage advance must retain the same owning request and rank")
            self._validate_source_claim(following)
            if self._position(following["chunk_id"]) <= self._position(previous["chunk_id"]):
                raise ValueError("Stage advance requires a later chunk")
            released = self.release(previous)
            prepared = self._prepare_validated(following)
            return {"status": "stage_advanced", "released": released, "prepared": prepared}

    def begin_shutdown(self):
        with self._lock:
            self._closing = True
            if self._current is not None:
                raise RuntimeError("Producer stage is still owned by a peer; drain it before shutdown")
