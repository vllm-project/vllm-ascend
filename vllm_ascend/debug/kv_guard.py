# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV address legality: allocation generation, valid prefix and request owners.

A prefix of zero is unwritten; INVALID_PREFIX is tainted. Loading and appending
produce the same readable prefix, so their provenance is not persistent state.
"""

from __future__ import annotations

import threading
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

INVALID_PREFIX = -1
# Group and logical block index are the positions in this nested table.
BlockTable = list[list[tuple[int, int]]]


@dataclass(slots=True)
class Block:
    epoch: int
    valid_end: int = 0
    owners: set[str] = field(default_factory=set)


class KVGuard:
    """One worker's pool, with request bindings and a reverse ownership index."""

    def __init__(self, emit: Any):
        self.emit = emit
        self.pool_id: str | None = None
        self.blocks: dict[tuple[int, int], Block] = {}
        self.requests: dict[str, BlockTable] = {}
        self.loads: dict[str, dict[str, Any]] = {}
        self.group_sizes: dict[int, int] = {}
        self._lock = threading.RLock()

    def configure(self, group_sizes: dict[int, int]) -> None:
        self.group_sizes = group_sizes

    def _violation(self, key: tuple[int, int], block: Block | None, operation: str, reason: str, **fields: Any) -> None:
        self.emit(
            "guard.violation",
            operation=operation,
            reason=reason,
            group_id=key[0],
            block_id=key[1],
            pool_id=self.pool_id,
            alloc_epoch=block.epoch if block else None,
            valid_end=block.valid_end if block else 0,
            owner_request_ids=sorted(block.owners) if block else [],
            **fields,
        )

    def bind(self, context: dict[str, Any]) -> None:
        with self._lock:
            pool = context.get("pool_id", self.pool_id)
            if pool != self.pool_id:
                self.blocks.clear()
                self.requests.clear()
                self.loads.clear()
                self.pool_id = pool
            for request_id in context.get("released_request_ids", ()):
                for group, refs in enumerate(self.requests.pop(request_id, ())):
                    for block_id, _ in refs:
                        block = self.blocks.get((group, block_id))
                        if block is not None:
                            block.owners.discard(request_id)
                self.loads.pop(request_id, None)
            for request_id, groups in context.get("requests", {}).items():
                keys = {(group, block) for group, refs in enumerate(groups) for block, _ in refs}
                for group, refs in enumerate(self.requests.get(request_id, ())):
                    for block_id, _ in refs:
                        key = group, block_id
                        if key not in keys and key in self.blocks:
                            self.blocks[key].owners.discard(request_id)
                self.requests[request_id] = groups
                for group, refs in enumerate(groups):
                    for block_id, epoch in refs:
                        if block_id == 0:
                            continue
                        if epoch <= 0:
                            self.emit(
                                "guard.coverage_gap",
                                reason="allocation_generation_missing",
                                group_id=group,
                                block_id=block_id,
                                request_id=request_id,
                            )
                            continue
                        key = group, block_id
                        block = self.blocks.get(key)
                        if block is None or block.epoch != epoch:
                            if block is not None and block.owners:
                                self._violation(key, block, "recycle", "recycle_live_block", request_id=request_id)
                            block = Block(epoch)
                            self.blocks[key] = block
                        block.owners.add(request_id)
                        # A consumer's computed prefix is a read requirement,
                        # not evidence that this allocation has been written.
            for meta in context.get("loads", ()):
                self.begin_load(meta)

    def begin_load(self, meta: dict[str, Any]) -> None:
        """Register the scheduler's submission snapshot, before any receive."""
        with self._lock:
            if meta["pool_id"] != self.pool_id or meta["request_id"] not in self.requests:
                self.emit("guard.coverage_gap", reason="load_binding_missing", request_id=meta["request_id"])
                return
            if not 0 <= meta["token_start"] < meta["token_end"]:
                self.emit("guard.coverage_gap", reason="load_range_invalid", request_id=meta["request_id"])
                return
            self.loads[meta["request_id"]] = {**meta, "block_table": [list(refs) for refs in meta["block_table"]]}

    def _range_targets(self, meta: dict[str, Any], group_ids=None, block_ids=None):
        request_id = meta["request_id"]
        if meta["pool_id"] != self.pool_id:
            self.emit("guard.coverage_gap", reason="stale_pool_event", request_id=request_id)
            return
        current = self.requests.get(request_id, ())
        for group, refs in enumerate(meta["block_table"]):
            if group_ids is not None and group not in group_ids:
                continue
            size = self.group_sizes[group]
            first = meta["token_start"] // size
            last = (meta["token_end"] + size - 1) // size
            for logical, (block_id, epoch) in enumerate(refs[first:last], first):
                start = max(0, meta["token_start"] - logical * size)
                end = min(size, meta["token_end"] - logical * size)
                if start >= end or (block_ids is not None and block_id not in block_ids):
                    continue
                key = group, block_id
                block = self.blocks.get(key)
                if block is not None and block.epoch != epoch:
                    self._violation(key, block, "load", "stale_allocation_epoch", request_id=request_id)
                elif (
                    block is None
                    or group >= len(current)
                    or logical >= len(current[group])
                    or current[group][logical] != (block_id, epoch)
                    or request_id not in block.owners
                ):
                    self._violation(key, block, "load", "unowned_destination", request_id=request_id)
                else:
                    yield key, block, start, end

    def _active_load(self, meta: dict[str, Any]) -> bool:
        active = self.loads.get(meta["request_id"])
        if active == meta:
            return True
        self.emit(
            "guard.coverage_gap", reason="stale_load_completion", request_id=meta["request_id"], load_id=meta["load_id"]
        )
        return False

    def load(self, meta: dict[str, Any], *, group_ids=None, block_ids=None, finish: bool = True) -> None:
        """Only a completed, matching submission can establish readable ranges."""
        with self._lock:
            if not self._active_load(meta):
                return
            for key, block, start, end in self._range_targets(meta, group_ids, block_ids):
                if block.valid_end == INVALID_PREFIX:
                    self.emit("guard.coverage_gap", reason="load_destination_tainted", group_id=key[0], block_id=key[1])
                    continue
                if start > block.valid_end:
                    self.emit("guard.coverage_gap", reason="load_prefix_missing", group_id=key[0], block_id=key[1])
                    continue
                block.valid_end = max(block.valid_end, end)
            if finish:
                self.finish_load(meta)

    def finish_load(self, meta: dict[str, Any]) -> None:
        with self._lock:
            if self.loads.get(meta["request_id"]) == meta:
                self.loads.pop(meta["request_id"])

    def fail_load(self, meta: dict[str, Any], *, block_ids=None, finish: bool = True) -> None:
        with self._lock:
            if not self._active_load(meta):
                return
            for _, block, _, _ in self._range_targets(meta, block_ids=block_ids):
                # A block transfer can overwrite its local prefix before failing.
                # Untouched local blocks stay valid; failed targets can be recomputed.
                if block.valid_end != INVALID_PREFIX:
                    block.valid_end = 0
            if finish:
                self.finish_load(meta)

    def recompute(self, meta: dict[str, Any]) -> None:
        """Truncate only a framework-authorized replay, before auditing writes."""
        with self._lock:
            for _, block, start, _ in self._range_targets(meta):
                if block.valid_end != INVALID_PREFIX:
                    block.valid_end = min(block.valid_end, start)

    def clear(self, group_id: int, block_id: int) -> None:
        with self._lock:
            block = self.blocks.get((group_id, block_id))
            if block is not None:
                block.valid_end = INVALID_PREFIX

    def copy_blocks(self, group_id: int, copies: list[tuple[int, int]]) -> None:
        """Match the operator's snapshot-all-sources-before-writing semantics."""
        with self._lock:
            source_ends = {
                source: block.valid_end
                for source, _ in copies
                if (block := self.blocks.get((group_id, source))) is not None
            }
            for source_id, destination_id in copies:
                valid = source_ends.get(source_id, INVALID_PREFIX)
                key = group_id, destination_id
                target = self.blocks.get(key)
                if valid == INVALID_PREFIX or target is None or not target.owners:
                    self._violation(
                        key, target, "copy", "invalid_copy_source_or_destination", source_block_id=source_id
                    )
                    if target is not None:
                        target.valid_end = INVALID_PREFIX
                else:
                    target.valid_end = valid

    def observe(self, groups: list[dict[str, Any]], request_ids: list[str], phase: str) -> None:
        """Audit WRITE_SLOT then READ_RANGE using attention's consumed metadata.

        An illegal write taints its destination so subsequent reads diagnose
        an invalid prefix, even when the block address itself remains legal.
        """
        with self._lock:
            for group in groups:
                group_id = group["group_id"]
                size = self.group_sizes[group_id]
                ranges: dict[int, list[int]] = {}
                expected = group.get("expected_slots")
                query_ends = group.get("query_ends", ())
                token_offset = group.get("token_offset", 0)
                mismatched: set[int] = set()
                slots = group["slots"]
                # List equality handles the common valid path in C. Only an
                # actual mismatch needs per-token Python diagnostics.
                count = min(len(slots), max(0, len(expected) - token_offset)) if expected is not None else 0
                if expected is not None and (
                    slots[:count] != expected[token_offset : token_offset + count]
                    or any(slot >= 0 for slot in slots[count:])
                ):
                    for index, slot in enumerate(slots):
                        token_index = token_offset + index
                        target = expected[token_index] if token_index < len(expected) else None
                        if (target is not None and slot == target) or (target is None and slot < 0):
                            continue
                        destination = slot if slot >= 0 else target
                        assert destination is not None
                        block_id = destination // size
                        key = group_id, block_id
                        block = self.blocks.get(key)
                        if block_id in mismatched:
                            continue
                        self._violation(
                            key,
                            block,
                            "write",
                            "write_target_mismatch",
                            token_index=token_index,
                            slot=slot,
                            expected_slot=target,
                            request_id=request_ids[bisect_right(query_ends, token_index)]
                            if query_ends and token_index < query_ends[-1]
                            else None,
                            phase=phase,
                        )
                        mismatched.add(block_id)
                        if block is not None:
                            block.valid_end = INVALID_PREFIX
                for slot in slots:
                    if slot < 0:
                        continue
                    ranges.setdefault(slot // size, []).append(slot % size)
                for block_id, offsets in ranges.items():
                    key, offsets = (group_id, block_id), sorted(offsets)
                    block = self.blocks.get(key)
                    if block_id in mismatched:
                        continue
                    if block_id == 0:
                        self._violation(key, None, "write", "write_null_block", offsets=offsets, phase=phase)
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
                    elif block.valid_end == INVALID_PREFIX:
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
                            block.valid_end = INVALID_PREFIX
                    elif block is not None:
                        block.valid_end = offsets[-1] + 1
                for row, length in enumerate(group.get("seq_lens", ())):
                    if length <= 0:
                        continue
                    request_id = request_ids[row] if row < len(request_ids) else None
                    table = group["block_table"][row]
                    expected_tables = group.get("expected_block_table")
                    expected_lengths = group.get("expected_seq_lens")
                    if expected_lengths is not None and (
                        row >= len(expected_lengths) or length > expected_lengths[row]
                    ):
                        self.emit(
                            "guard.violation",
                            operation="read",
                            reason="read_length_mismatch",
                            group_id=group_id,
                            request_id=request_id,
                            length=length,
                            phase=phase,
                        )
                        continue
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
                        if expected_tables is not None and block_id != expected_tables[row][logical]:
                            self._violation(
                                (group_id, block_id),
                                self.blocks.get((group_id, block_id)),
                                "read",
                                "read_target_mismatch",
                                request_id=request_id,
                                logical_block_index=logical,
                                expected_block_id=expected_tables[row][logical],
                                phase=phase,
                            )
                            continue
                        if block_id == 0:
                            if request_id is not None:
                                reason = "read_null_block"
                                self._violation(
                                    (group_id, block_id), None, "read", reason, phase=phase, request_id=request_id
                                )
                            continue
                        key, end = (group_id, block_id), min(size, length - logical * size)
                        block = self.blocks.get(key)
                        if block is None:
                            reason = "read_unallocated_block"
                        elif block.valid_end == INVALID_PREFIX:
                            reason = "read_invalid_block"
                        elif request_id not in block.owners:
                            reason = "unowned_read"
                        elif end > block.valid_end:
                            reason = "read_unwritten_range"
                        else:
                            continue
                        self._violation(key, block, "read", reason, end_offset=end, phase=phase, request_id=request_id)
