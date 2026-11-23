# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request incarnation identities for one worker's resident sparse KV state."""

from collections.abc import Callable

MAX_OWNER_ID = (1 << 63) - 1


class RequestIdentityRegistry:
    """Assign positive int64 owners, invalidating resident state before wrap.

    The runner owns this registry and shares it with target and draft metadata.
    Removing a local request releases its string key. Owners are not recycled
    within an allocation cycle: a resident row or an outstanding plan can still
    refer to a retired owner. Before wrap, ``on_wrap`` must drain outstanding
    work and invalidate every resident owner reference, including draft layers.
    Owners live at wrap stay reserved for the whole next cycle, even if their
    requests finish, since those requests can populate resident rows again.
    ``on_wrap`` may be omitted only when no resident state uses this registry.
    Only request-state removal, not removal from the scheduled batch, ends an
    incarnation. The runner's input preparation serializes all access.
    """

    def __init__(self, *, on_wrap: Callable[[], None] | None = None) -> None:
        self._owner_ids: dict[str, int] = {}
        self._next_owner_id = 1
        self._on_wrap = on_wrap
        self._reserved_owner_ids: set[int] = set()

    def get_or_create(self, req_id: str) -> int:
        if not isinstance(req_id, str):
            raise TypeError("Sparse KV offload request IDs must be strings")
        owner_id = self._owner_ids.get(req_id)
        if owner_id is not None:
            return owner_id
        while True:
            if self._next_owner_id > MAX_OWNER_ID:
                if len(self._owner_ids) >= MAX_OWNER_ID:
                    raise RuntimeError("Sparse KV offload has no free request owner IDs")
                if self._on_wrap is not None:
                    self._on_wrap()
                self._reserved_owner_ids = set(self._owner_ids.values())
                # Keep zero padding and the negative dummy/sentinel domain out.
                self._next_owner_id = (self._next_owner_id - 1) % MAX_OWNER_ID + 1
            owner_id = self._next_owner_id
            self._next_owner_id += 1
            if owner_id not in self._reserved_owner_ids:
                break
        self._owner_ids[req_id] = owner_id
        return owner_id

    def remove(self, req_id: str) -> None:
        self._owner_ids.pop(req_id, None)

    @staticmethod
    def dummy_owner_id(row: int) -> int:
        # Zero is padding and -1 is the native planner's empty-row sentinel.
        return -2 - row
