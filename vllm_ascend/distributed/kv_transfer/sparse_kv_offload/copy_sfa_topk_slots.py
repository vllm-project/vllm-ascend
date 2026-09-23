# SPDX-License-Identifier: Apache-2.0
"""Request-owned fused_copy_sfa top-k row slots shared by the PD connector and runner."""

from __future__ import annotations

COPY_SFA_POOL_PADDING_ROWS = 2


def copy_sfa_pool_capacity(max_num_seqs: int) -> int:
    """Match the runner-owned fused_copy_sfa row arena: one row per request plus padding."""
    return max_num_seqs + COPY_SFA_POOL_PADDING_ROWS


def copy_sfa_tail_geometry(kv_tokens: int, block_size: int) -> tuple[int, int]:
    """Return ``(tail_tokens, tail_block_index)`` for a finished prefill prefix.

    The circular tail only stores the incomplete last block. A 128-aligned
    prefix has nothing to prefetch.
    """
    if kv_tokens <= 0 or block_size <= 0:
        return 0, 0
    tail_tokens = kv_tokens % block_size
    if tail_tokens == 0:
        return 0, 0
    return tail_tokens, kv_tokens // block_size


def copy_sfa_prefill_dest_geometry(
    kv_tokens: int,
    block_size: int,
    hot_tokens: int,
) -> tuple[bool, int, int]:
    """Return ``(dense, tail_tokens, tail_block_index)`` for a finished prefill.

    ``dense=True`` when the whole prompt fits the decode row's hot region
    (``kv_tokens <= hot_tokens``): blocks ``[0, ceil(kv_tokens / block_size))``
    are D2D'd to row offsets ``b * block_size`` so the request can decode in
    the -3 non-offload state. This also covers 128-aligned prompts that the
    circular-tail path would skip entirely. Otherwise the circular tail only
    prefetches the incomplete last block; a block-aligned ``kv_tokens`` keeps
    ``(False, 0, 0)`` and the hot region arrives via the decode-side -2 init.
    """
    if kv_tokens <= hot_tokens:
        return True, 0, 0
    tail_tokens, tail_block_index = copy_sfa_tail_geometry(kv_tokens, block_size)
    return False, tail_tokens, tail_block_index


class CopySfaTopkSlotAllocator:
    """Bind a stable top-k row to a request from PD alloc until it finishes."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"fused_copy_sfa topk slot capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._free: list[int] = list(range(capacity))
        self._req_to_slot: dict[str, int] = {}

    def bind(self, req_id: str) -> int:
        slot = self._req_to_slot.get(req_id)
        if slot is not None:
            return slot
        if not self._free:
            raise RuntimeError(f"fused_copy_sfa topk slot pool exhausted (capacity={self.capacity})")
        slot = self._free.pop(0)
        self._req_to_slot[req_id] = slot
        return slot

    def get(self, req_id: str) -> int | None:
        return self._req_to_slot.get(req_id)

    def release(self, req_id: str) -> int | None:
        slot = self._req_to_slot.pop(req_id, None)
        if slot is not None:
            self._free.append(slot)
        return slot

    def bound_slots(self) -> dict[str, int]:
        return dict(self._req_to_slot)
