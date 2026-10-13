# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 KV access legality, independent of cache values and checksums.

States follow debug_trace.pptx. A loaded tail retains its valid length: loading
19 tokens into a 128-token block does not prohibit a legal append at offset 19.
The worker adapter supplies execution plans and optional operator arguments.
Values and content checksums are deliberately outside this L1 state machine.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockState(str, Enum):
    UNKNOWN = "BLOCK_UNKNOWN"
    INVALID = "INVALIDATE_BLOCK"
    PARTIAL = "BLOCK_SLOT_PARTIAL"
    SEALED_LOAD = "BLOCK_SEALED_LOAD"
    SEALED_FILL = "BLOCK_SEALED_FILL"


@dataclass
class Block:
    pool_id: str
    epoch: str
    size: int
    state: BlockState = BlockState.UNKNOWN
    valid_end: int = 0
    owners: set[str] = field(default_factory=set)


class KVGuard:
    """One process-local, bounded state table shared by worker and receiver.

    Identity is (pool, group, physical block, allocation epoch). Allocating the
    same address again replaces its state. Pending loads are associated with
    a request, never applied to another request merely reusing an address.
    """

    def __init__(self, emit: Any):
        self.emit = emit
        self.blocks: dict[tuple[int, int], Block] = {}
        self.requests: dict[str, list[dict[str, Any]]] = {}
        self.pending_loads: dict[str, dict[str, Any]] = {}
        self.group_sizes: dict[int, int] = {}
        self.tainted_nulls: set[int] = set()
        self._lock = threading.RLock()

    def configure(self, group_sizes: dict[int, int]) -> None:
        self.group_sizes = group_sizes

    def _transition(self, key: tuple[int, int], block: Block, state: BlockState, valid_end: int, action: str) -> None:
        previous = block.state
        block.state, block.valid_end = state, valid_end
        # PARTIAL advances in memory; logging every decoded token defeats L1.
        if previous != state:
            self.emit(
                "guard.state",
                group_id=key[0],
                block_id=key[1],
                pool_id=block.pool_id,
                alloc_epoch=block.epoch,
                previous_state=previous.value,
                state=state.value,
                valid_end=valid_end,
                action=action,
            )

    def _violation(self, key: tuple[int, int], block: Block | None, operation: str, reason: str, **fields: Any) -> None:
        self.emit(
            "guard.violation",
            operation=operation,
            reason=reason,
            group_id=key[0],
            block_id=key[1],
            pool_id=block.pool_id if block else None,
            alloc_epoch=block.epoch if block else None,
            state=block.state.value if block else BlockState.UNKNOWN.value,
            valid_end=block.valid_end if block else 0,
            owner_request_ids=sorted(block.owners) if block else [],
            **fields,
        )

    def bind(self, context: dict[str, Any]) -> None:
        with self._lock:
            for request_id in context.get("finished_request_ids", ()):
                for ref in self.requests.pop(request_id, ()):
                    block = self.blocks.get((ref["group_id"], ref["block_id"]))
                    if block is not None:
                        block.owners.discard(request_id)
                self.pending_loads.pop(request_id, None)
            for request in context.get("requests", ()):
                request_id = request["request_id"]
                refs = request.get("block_refs", ())
                new_keys = {(ref["group_id"], ref["block_id"]) for ref in refs}
                for ref in self.requests.get(request_id, ()):
                    key = ref["group_id"], ref["block_id"]
                    if key not in new_keys and key in self.blocks:
                        self.blocks[key].owners.discard(request_id)
                self.requests[request_id] = list(refs)
                for ref in refs:
                    group, block_id = ref["group_id"], ref["block_id"]
                    epoch, pool = ref.get("alloc_epoch"), ref.get("pool_id")
                    if epoch is None or pool is None or group not in self.group_sizes:
                        continue
                    key = group, block_id
                    block = self.blocks.get(key)
                    if block is None or (block.pool_id, block.epoch) != (pool, str(epoch)):
                        if block is not None:
                            if block.owners:
                                self._violation(key, block, "recycle", "recycle_live_block", request_id=request_id)
                            self._transition(key, block, BlockState.INVALID, 0, "recycle")
                        block = Block(pool, str(epoch), self.group_sizes[group])
                        self.blocks[key] = block
                    block.owners.add(request_id)
                    # Upstream prefix reuse is a LOAD operation, not a fresh
                    # sequential write. Never revive a tainted existing epoch.
                    computed = request.get("num_computed_tokens") or 0
                    valid = min(block.size, max(0, computed - ref["logical_block_index"] * block.size))
                    if valid and block.state == BlockState.UNKNOWN:
                        self._transition(key, block, BlockState.SEALED_LOAD, valid, "prefix_load")
                pending = self.pending_loads.pop(request_id, None)
                if pending is not None:
                    self._load(pending)

    def load(self, meta: dict[str, Any]) -> None:
        with self._lock:
            if meta.get("request_id") not in self.requests or not self.group_sizes:
                self.pending_loads[meta["request_id"]] = {
                    key: meta[key] for key in ("request_id", "local_block_ids", "num_computed_tokens") if key in meta
                }
                return
            self._load(meta)

    def _load(self, meta: dict[str, Any]) -> None:
        request_id = meta["request_id"]
        expected = {(r["group_id"], r["block_id"]): r for r in self.requests.get(request_id, ())}
        for group, ids in enumerate(meta.get("local_block_ids", ())):
            for block_id in ids:
                key = group, block_id
                ref, block = expected.get(key), self.blocks.get(key)
                if ref is None or block is None or request_id not in block.owners:
                    self._violation(key, block, "load", "unowned_destination", request_id=request_id)
                    continue
                if (block.pool_id, block.epoch) != (ref["pool_id"], str(ref["alloc_epoch"])):
                    self._violation(key, block, "load", "stale_allocation_epoch", request_id=request_id)
                    continue
                computed = meta.get("num_computed_tokens")
                if computed is None:
                    self.emit(
                        "guard.coverage_gap", reason="load_valid_length_missing", group_id=group, block_id=block_id
                    )
                    continue
                valid = min(block.size, max(0, computed - ref["logical_block_index"] * block.size))
                self._transition(key, block, BlockState.SEALED_LOAD, valid, "transfer_load")

    def fail_load(self, request_id: str) -> None:
        with self._lock:
            self.pending_loads.pop(request_id, None)
            for ref in self.requests.get(request_id, ()):
                key = ref["group_id"], ref["block_id"]
                block = self.blocks.get(key)
                if block is not None and block.epoch == str(ref.get("alloc_epoch")):
                    self._transition(key, block, BlockState.INVALID, 0, "load_failed")

    def clear(self, group_id: int, block_id: int) -> None:
        """Physical clear/recycle; request release with cached content is different."""
        with self._lock:
            if block_id == 0:
                self.tainted_nulls.discard(group_id)
            key = group_id, block_id
            block = self.blocks.get(key)
            if block is not None:
                self._transition(key, block, BlockState.INVALID, 0, "clear")

    def copy(self, group_id: int, source_id: int, destination_id: int, request_id: str) -> None:
        """State transition for a local whole-block copy, without inspecting values."""
        with self._lock:
            source = self.blocks.get((group_id, source_id))
            key = group_id, destination_id
            target = self.blocks.get(key)
            if (
                source is None
                or target is None
                or source.state not in (BlockState.PARTIAL, BlockState.SEALED_LOAD, BlockState.SEALED_FILL)
                or request_id not in target.owners
            ):
                self._violation(key, target, "copy", "invalid_copy_source_or_destination", source_block_id=source_id)
                if target is not None:
                    self._transition(key, target, BlockState.INVALID, 0, "copy_failed")
            else:
                self._transition(key, target, BlockState.SEALED_LOAD, source.valid_end, "copy_block")

    def observe(self, groups: list[dict[str, Any]], request_ids: list[str], phase: str) -> None:
        """Audit WRITE_SLOT then READ_RANGE using attention's consumed metadata.

        An illegal write taints its destination so subsequent reads diagnose
        INVALIDATE_BLOCK, even when the block address itself remains legal.
        """
        with self._lock:
            for group in groups:
                group_id, size = group["group_id"], group["block_size"]
                null_id = group.get("null_block_id", 0)
                ranges: dict[int, list[int]] = {}
                for slot in group["slots"]:
                    if slot < 0:
                        continue
                    ranges.setdefault(slot // size, []).append(slot % size)
                for block_id, offsets in ranges.items():
                    key, offsets = (group_id, block_id), sorted(offsets)
                    block = self.blocks.get(key)
                    if block_id == null_id:
                        self._violation(key, None, "write", "write_null_block", offsets=offsets, phase=phase)
                        self.tainted_nulls.add(group_id)
                        continue
                    # Ring/compressed/state caches need their own validity rules.
                    # Their declared destinations still receive null protection.
                    if group.get("write_mode", "append") != "append":
                        continue
                    if phase == "warmup" and block is None:
                        continue  # Capture may use explicitly allocated scratch.
                    reason = None
                    if block is None:
                        reason = "unallocated_block"
                    elif phase in ("dummy", "warmup"):
                        reason = "dummy_write_to_live_block"
                    elif not block.owners.intersection(request_ids):
                        reason = "unowned_write"
                    elif block.state == BlockState.INVALID:
                        reason = "write_invalid_block"
                    elif offsets[0] < block.valid_end:
                        reason = "overwrite_valid_prefix"
                    elif offsets != list(range(block.valid_end, block.valid_end + len(offsets))):
                        reason = "noncontiguous_write"
                    if reason is not None:
                        self._violation(
                            key, block, "write", reason, offsets=offsets, phase=phase, request_ids=request_ids
                        )
                        if block is not None:
                            self._transition(key, block, BlockState.INVALID, block.valid_end, "illegal_write")
                    elif block is not None:
                        valid = offsets[-1] + 1
                        state = BlockState.SEALED_FILL if valid == size else BlockState.PARTIAL
                        self._transition(key, block, state, valid, "write_slot")
                for row, length in enumerate(group.get("seq_lens", ())):
                    if length <= 0:
                        continue
                    request_id = request_ids[row] if row < len(request_ids) else None
                    table = group["block_table"][row]
                    needed = (length + size - 1) // size
                    if needed > len(table):
                        self.emit(
                            "guard.violation",
                            operation="read",
                            reason="short_block_table",
                            group_id=group_id,
                            phase=phase,
                        )
                        continue
                    for logical, block_id in enumerate(table[:needed]):
                        if block_id == null_id:
                            if request_id is not None:
                                reason = (
                                    "read_tainted_null_block" if group_id in self.tainted_nulls else "read_null_block"
                                )
                                self._violation(
                                    (group_id, block_id), None, "read", reason, phase=phase, request_id=request_id
                                )
                            continue
                        key, end = (group_id, block_id), min(size, length - logical * size)
                        block = self.blocks.get(key)
                        if block is None:
                            reason = "read_unallocated_block"
                        elif block.state == BlockState.INVALID:
                            reason = "read_invalid_block"
                        elif request_id not in block.owners:
                            reason = "unowned_read"
                        elif end > block.valid_end:
                            reason = "read_unwritten_range"
                        else:
                            continue
                        self._violation(key, block, "read", reason, end_offset=end, phase=phase, request_id=request_id)
