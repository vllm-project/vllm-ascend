# SPDX-License-Identifier: Apache-2.0
"""Request-owned fused_copy_sfa top-k row slots shared by the PD connector and runner."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

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


@dataclass
class _CopySfaRequestState:
    slot: int
    generation: int
    last_prefix: int | None = None


class CopySfaRequestStates:
    """Request-owned history; batch buffers are views of these bindings.

    PD reservations remain authoritative. Generations invalidate the existing
    target/MTP LIM banks when a slot acquires a new owner. Prefix history belongs
    to the same request incarnation, never to a reusable physical slot.
    """

    def __init__(self) -> None:
        self._requests: dict[str, _CopySfaRequestState] = {}
        self._generation = 0
        self._retired_requests: set[str] = set()

    def remove_request(self, req_id: str) -> None:
        self._requests.pop(req_id, None)
        self._retired_requests.add(req_id)

    def prepare(
        self,
        *,
        req_ids: Sequence[str],
        live_req_ids: Collection[str],
        slots: np.ndarray,
        generations: np.ndarray,
        prebound_slots: dict[str, int],
        computed_tokens: np.ndarray | None,
        padded_reqs: int,
        block_size: int,
        hot_tokens: int,
        dummy: bool,
    ) -> tuple[bool, dict[int, tuple[int, int]]]:
        capacity = len(slots)
        if not 0 <= len(req_ids) <= padded_reqs <= capacity or len(generations) != capacity:
            raise ValueError("fused_copy_sfa request rows exceed the slot buffer capacity")
        if block_size <= 0:
            raise ValueError("fused_copy_sfa block size must be positive")
        prepare_copy_sfa_dummy_slots(slots, generations, padded_reqs)
        if dummy:
            return False, {}
        if len(set(req_ids)) != len(req_ids):
            raise ValueError("fused_copy_sfa batch contains duplicate request IDs")
        # Worker connector cleanup runs after forward. The runner can therefore
        # see an already-finished owner's reservation alongside its replacement.
        # Forget retirement only once the connector drops it, or this ID is
        # explicitly admitted again with fresh request state.
        self._retired_requests.intersection_update(prebound_slots)
        self._retired_requests.difference_update(req_ids)
        prebound_slots = {req: slot for req, slot in prebound_slots.items() if req not in self._retired_requests}
        live = set(live_req_ids) | set(prebound_slots)
        if not set(req_ids) <= live:
            raise ValueError("fused_copy_sfa batch contains a request without a live owner")
        if computed_tokens is not None and len(computed_tokens) < len(req_ids):
            raise ValueError("fused_copy_sfa computed-token rows do not cover the batch")
        # Keep live batch bindings and PD reservations awaiting their next decode.
        retained = {req: state for req, state in self._requests.items() if req in live}
        owners: dict[int, str] = {}
        for req, slot in prebound_slots.items():
            if not 0 <= slot < capacity:
                raise ValueError(f"fused_copy_sfa invalid PD slot {slot} for {req}")
            if slot in owners:
                raise RuntimeError(f"fused_copy_sfa slot {slot} reserved by two requests")
            owners[slot] = req
        for req, state in retained.items():
            reserved = prebound_slots.get(req, state.slot)
            if reserved != state.slot:
                raise RuntimeError(f"fused_copy_sfa binding changed for live request {req}: {state.slot} -> {reserved}")
            if state.slot in owners and owners[state.slot] != req:
                raise RuntimeError(f"fused_copy_sfa slot {state.slot} still owned by {req}")
            owners[state.slot] = req
        available = iter(slot for slot in range(capacity) if slot not in owners)
        pending: dict[str, _CopySfaRequestState] = {}
        generation = self._generation
        for req in req_ids:
            if req in retained:
                continue
            slot = prebound_slots.get(req)
            if slot is None:
                slot = next(available, None)
            if slot is None:
                raise RuntimeError(f"fused_copy_sfa topk slot pool exhausted (capacity={capacity})")
            generation += 1
            if generation > np.iinfo(generations.dtype).max:
                raise RuntimeError("fused_copy_sfa request generation exhausted")
            pending[req] = _CopySfaRequestState(slot, generation)
        # Publish bindings only after validating the entire ownership transition.
        retained.update(pending)
        self._requests = retained
        self._generation = generation
        restore_tails = False
        dense_fills: dict[int, tuple[int, int]] = {}
        for row, req in enumerate(req_ids):
            state = retained[req]
            slots[row] = state.slot
            generations[row] = state.generation
            if computed_tokens is None:
                continue
            prefix = (int(computed_tokens[row]) // block_size) * block_size
            if state.last_prefix is not None and prefix < state.last_prefix:
                if prebound_slots:
                    restore_tails = True
                if hot_tokens and state.last_prefix >= hot_tokens > prefix:
                    dense_fills[state.slot] = (row, int(computed_tokens[row]))
            state.last_prefix = prefix
        return restore_tails, dense_fills


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
