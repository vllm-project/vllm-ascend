# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request incarnation identities for one worker's resident sparse KV state."""

MAX_OWNER_ID = (1 << 63) - 1


class RequestIdentityRegistry:
    """Assign non-recycled int64 owners until the resident state is destroyed.

    The runner owns this registry and shares it with target and draft metadata.
    Removing a local request releases its string key, but never recycles its
    owner: a resident row or an outstanding plan can still refer to that owner.
    Only request-state removal, not removal from the scheduled batch, ends an
    incarnation. The runner's input preparation serializes all access.
    """

    def __init__(self) -> None:
        self._owner_ids: dict[str, int] = {}
        self._next_owner_id = 1

    def get_or_create(self, req_id: str) -> int:
        if not isinstance(req_id, str):
            raise TypeError("Sparse KV offload request IDs must be strings")
        owner_id = self._owner_ids.get(req_id)
        if owner_id is not None:
            return owner_id
        if self._next_owner_id > MAX_OWNER_ID:
            raise OverflowError("Sparse KV offload request owner IDs exhausted int64")
        owner_id = self._next_owner_id
        self._next_owner_id += 1
        self._owner_ids[req_id] = owner_id
        return owner_id

    def remove(self, req_id: str) -> None:
        self._owner_ids.pop(req_id, None)

    @staticmethod
    def dummy_owner_id(row: int) -> int:
        # Zero is padding and -1 is the native planner's empty-row sentinel.
        return -2 - row
