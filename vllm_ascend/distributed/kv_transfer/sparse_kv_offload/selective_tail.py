# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class _Tail:
    slot: int
    block: int
    first: int


class SelectiveTailSlots:
    """Keep tail storage bound to a request across batch reordering."""

    def __init__(self, capacity: int, block_size: int):
        self.capacity = capacity
        self.block_size = block_size
        self.requests: dict[str, _Tail] = {}

    def release(self, req_ids) -> None:
        for req_id in req_ids:
            self.requests.pop(req_id, None)

    def row(self, req_id: str, position: int) -> tuple[int, int]:
        block, offset = divmod(position, self.block_size)
        tail = self.requests.get(req_id)
        if tail is None:
            used = {entry.slot for entry in self.requests.values()}
            slot = next((i for i in range(self.capacity) if i not in used), None)
            if slot is None:
                raise RuntimeError("Selective tail slots exhausted")
            tail = self.requests[req_id] = _Tail(slot, block, offset)
        if tail.block != block:
            tail.block = block
            tail.first = offset
        else:
            tail.first = min(tail.first, offset)
        return tail.slot, tail.first
