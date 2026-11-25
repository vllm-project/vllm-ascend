# SPDX-License-Identifier: Apache-2.0
"""Track NOVA exact-window ownership across scheduler steps."""


class NovaRequestState:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("NOVA exact-window capacity must be positive")
        self.capacity = capacity
        self._slots = {}
        self._lengths = {}
        self._free = set(range(capacity))

    def release(self, request_ids):
        for request_id in request_ids:
            slot = self._slots.pop(request_id, None)
            self._lengths.pop(request_id, None)
            if slot is not None:
                self._free.add(slot)

    def missing(self, request_ids, computed_tokens):
        return {
            request_id
            for request_id, computed in zip(request_ids, computed_tokens, strict=True)
            if int(computed) > 0 and (request_id not in self._slots or self._lengths.get(request_id) != int(computed))
        }

    def prepare(self, request_ids, computed_tokens, scheduled_tokens):
        slots = []
        for request_id, computed, scheduled in zip(request_ids, computed_tokens, scheduled_tokens, strict=True):
            computed, scheduled = int(computed), int(scheduled)
            if computed < 0 or scheduled < 1:
                raise ValueError("NOVA request lengths must be nonnegative with a positive scheduled length")
            slot = self._slots.get(request_id)
            if computed == 0:
                if slot is None:
                    if not self._free:
                        raise RuntimeError("NOVA exact-window capacity exhausted")
                    slot = min(self._free)
                    self._free.remove(slot)
                    self._slots[request_id] = slot
            elif slot is None or self._lengths.get(request_id) != computed:
                raise RuntimeError("NOVA exact history is unavailable; the request must be recomputed from zero")
            self._lengths[request_id] = computed + scheduled
            slots.append(slot)
        return slots
