# SPDX-License-Identifier: Apache-2.0
"""Request-owned fused_copy_sfa top-k row slots shared by the PD connector and runner."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

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


def prepare_copy_sfa_dummy_slots(slots: np.ndarray, generations: np.ndarray, padded_reqs: int) -> None:
    """Fill CPU buffer views with private rows and inactive generations.

    Pass the full slot buffer: its length is the real-row pool capacity.
    """
    slots[:padded_reqs] = np.arange(padded_reqs, dtype=np.int32) + len(slots)
    generations[:padded_reqs] = -1


def prepare_copy_sfa_request_slots(
    *,
    req_ids: Sequence[str],
    live_req_ids: Collection[str],
    slots: np.ndarray,
    generations: np.ndarray,
    request_slots: dict[str, int],
    slot_generations: dict[int, int],
    last_prefixes: dict[int, int],
    generation: int,
    prebound_slots: dict[str, int],
    computed_tokens: np.ndarray | None,
    padded_reqs: int,
    block_size: int,
    hot_tokens: int,
    dummy: bool,
) -> tuple[dict[str, int], int, bool, dict[int, tuple[int, int]]]:
    """Prepare CPU row ownership and return updated state plus rollback work.

    ``req_ids`` contains the rows in this batch. Pass the full CPU slot and
    generation buffer views, whose length is the real-row pool capacity.
    Slot generations and prefix history update in place. The result contains
    the retained request mapping, generation counter, tail-restore flag and
    dense fills as ``slot -> (batch row, computed tokens)``. The caller owns
    these buffers/maps and submits any returned dense fills to the manager.
    """
    capacity = len(slots)
    prepare_copy_sfa_dummy_slots(slots, generations, padded_reqs)
    restore_tails = False
    dense_fills: dict[int, tuple[int, int]] = {}
    if not dummy:
        # PD binds rows at alloc time. Keep those reservations even when the
        # request is waiting for KV and is not in the current decode batch.
        request_slots = {
            req: slot for req, slot in request_slots.items() if req in live_req_ids or req in prebound_slots
        }
        used = set(request_slots.values()) | set(prebound_slots.values())
        available = iter(slot for slot in range(capacity) if slot not in used)
        for row, req in enumerate(req_ids):
            if req not in request_slots:
                slot = prebound_slots[req] if req in prebound_slots else next(available)
                request_slots[req] = slot
                generation += 1
                slot_generations[slot] = generation
                # Fresh short rows get their full KV via the prefill-end
                # D2D in exec_kv; only rollbacks need the host-pool fill.
            slot = request_slots[req]
            slots[row] = slot
            generations[row] = slot_generations[slot]
        if computed_tokens is not None:
            for row in range(len(req_ids)):
                slot = int(slots[row])
                prefix = (int(computed_tokens[row]) // block_size) * block_size
                last_prefix = last_prefixes.get(slot)
                if last_prefix is not None and prefix < last_prefix:
                    if prebound_slots:
                        restore_tails = True
                    # A rollback across the hot boundary leaves a sparse-layout
                    # row under a dense (-3) reader: refill the whole row.
                    if hot_tokens and last_prefix >= hot_tokens > prefix:
                        dense_fills[slot] = (row, int(computed_tokens[row]))
                last_prefixes[slot] = prefix
    return request_slots, generation, restore_tails, dense_fills


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
