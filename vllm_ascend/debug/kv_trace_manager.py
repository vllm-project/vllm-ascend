# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe request ownership at the KVCacheManager boundary, without a proxy."""

from __future__ import annotations

import functools
import logging
import uuid
from typing import Any


class PoolTrace:
    """Allocator identities, independent of request leases and cache contents."""

    METHODS = ("get_new_blocks", "touch", "free_blocks", "_maybe_evict_cached_block", "reset_prefix_cache")

    def __init__(self, pool: Any, trace: Any):
        self.pool = pool
        self.trace = trace
        self.pool_id = uuid.uuid4().hex
        self.null_id = pool.null_block.block_id
        # Existing content predates this observer. Never invent its generation.
        self.epochs: list[int | None] = [None] * len(pool.blocks)
        self.counters = [0] * len(pool.blocks)

    def reference(self, block_id: int, **fields: Any) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "block_id": block_id,
            "alloc_epoch": str(self.epochs[block_id]) if self.epochs[block_id] is not None else None,
            "coordinate_kind": "allocator",
            **fields,
        }

    def wrap(self, name: str) -> None:
        original = getattr(self.pool, name)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            if not self.trace.enabled:
                return original(*args, **kwargs)
            # free_blocks accepts a one-shot iterable, which the original also
            # materializes. Forward that same materialized input exactly once.
            blocks = None
            if name in ("touch", "free_blocks"):
                key = "blocks" if name == "touch" else "ordered_blocks"
                blocks = list(args[0] if args else kwargs[key])
                if args:
                    args = (blocks, *args[1:])
                else:
                    kwargs[key] = blocks
            try:
                result = original(*args, **kwargs)
            except BaseException:
                # A partially completed allocator operation has unknown state.
                self.epochs[:] = [None] * len(self.epochs)
                self.trace.emit("trace.gap", stage=name, pool_id=self.pool_id, reason="allocator_operation_failed")
                raise
            try:
                if name == "get_new_blocks":
                    for block in result:
                        if block.block_id == self.null_id:
                            continue
                        self.counters[block.block_id] += 1
                        self.epochs[block.block_id] = self.counters[block.block_id]
                        self.trace.emit("block.alloc", **self.reference(block.block_id), ref_count=block.ref_cnt)
                elif blocks is not None:
                    for block_id in dict.fromkeys(b.block_id for b in blocks if b.block_id != self.null_id):
                        self.trace.emit(
                            "block.ref_change",
                            **self.reference(block_id),
                            action=name,
                            ref_count=self.pool.blocks[block_id].ref_cnt,
                        )
                elif name == "_maybe_evict_cached_block" and result:
                    block = args[0] if args else kwargs["block"]
                    self.trace.emit("block.evict", **self.reference(block.block_id), reason="cache_binding_removed")
                elif name == "reset_prefix_cache":
                    self.trace.emit("cache.reset", pool_id=self.pool_id, success=bool(result))
            except Exception:
                self.epochs[:] = [None] * len(self.epochs)
                self.trace.emit("trace.observation_error", stage=name, pool_id=self.pool_id)
                logging.getLogger(__name__).exception("KV trace could not observe allocator")
            return result

        setattr(self.pool, name, observed)


def attach_pool_trace(pool: Any, trace: Any) -> PoolTrace | None:
    existing = getattr(pool, "_ascend_kv_trace", None)
    if existing is not None:
        return existing
    missing = [name for name in PoolTrace.METHODS if not callable(getattr(pool, name, None))]
    if missing or not hasattr(pool, "null_block"):
        trace.emit("trace.capability", capability="allocator_epochs", supported=False, reason="unsupported_pool_api")
        return None
    observer = PoolTrace(pool, trace)
    for name in PoolTrace.METHODS:
        observer.wrap(name)
    pool._ascend_kv_trace = observer
    trace.emit(
        "pool.baseline",
        pool_id=observer.pool_id,
        null_block_id=observer.null_id,
        num_blocks=len(pool.blocks),
        initial_epochs="unknown_until_observed_allocation",
    )
    return observer


class ManagerTrace:
    def __init__(self, manager: Any, trace: Any):
        self.manager = manager
        self.trace = trace
        self.requests: dict[str, tuple[list[int], ...]] = {}
        self.owners: dict[tuple[int, int], set[str]] = {}
        self.leases: dict[tuple[int, int], int] = {}
        self.pool = attach_pool_trace(manager.block_pool, trace)
        self.null_id = self.pool.null_id if self.pool else 0
        self.step = 0
        self.revisions: dict[str, tuple[Any, int]] = {}

    def refs(self, blocks: tuple[list[int], ...]) -> list[dict[str, Any]]:
        return [
            self.pool.reference(block, group_id=group, logical_block_index=index)
            for group, ids in enumerate(blocks)
            for index, block in enumerate(ids)
            if self.pool is not None and block != self.null_id
        ]

    def identity(self, block: int) -> dict[str, Any]:
        if self.pool is None:
            return {"alloc_epoch": None, "pool_id": None}
        ref = self.pool.reference(block)
        return {"alloc_epoch": ref["alloc_epoch"], "pool_id": ref["pool_id"]}

    def _blocks(self, request_id: str) -> tuple[list[int], ...]:
        return tuple(list(group) for group in self.manager.get_block_ids(request_id))

    def _keys(self, blocks):
        return {(group, block) for group, ids in enumerate(blocks) for block in ids if block != self.null_id}

    def update(self, request: Any, after: tuple[list[int], ...], action: str) -> None:
        request_id = request.request_id
        before = self.requests.get(request_id, ())
        old_keys, new_keys = self._keys(before), self._keys(after)
        for key in sorted(old_keys - new_keys):
            owners = self.owners[key]
            owners.discard(request_id)
            self.trace.emit(
                "block.release",
                request_id=request_id,
                group_id=key[0],
                block_id=key[1],
                lease=self.leases[key],
                remaining_owners=sorted(owners),
                ref_count=self.manager.block_pool.blocks[key[1]].ref_cnt,
                **self.identity(key[1]),
            )
        for key in sorted(new_keys - old_keys):
            owners = self.owners.setdefault(key, set())
            if not owners:
                self.leases[key] = self.leases.get(key, 0) + 1
            owners.add(request_id)
            self.trace.emit(
                "block.acquire",
                request_id=request_id,
                group_id=key[0],
                block_id=key[1],
                lease=self.leases[key],
                shared=len(owners) > 1,
                ref_count=self.manager.block_pool.blocks[key[1]].ref_cnt,
                **self.identity(key[1]),
            )
        if action == "free":
            self.requests.pop(request_id, None)
        else:
            self.requests[request_id] = after
        self.trace.emit(
            "request.blocks",
            request_id=request_id,
            action=action,
            block_ids=after,
            block_refs=self.refs(after),
            status=str(request.status),
            num_computed_tokens=request.num_computed_tokens,
        )

    def wrap(self, name: str) -> None:
        original = getattr(self.manager, name)

        @functools.wraps(original)
        def observed(request, *args, **kwargs):
            result = original(request, *args, **kwargs)
            if not self.trace.enabled:
                return result
            try:
                if name == "get_computed_blocks":
                    blocks, num_tokens = result
                    self.trace.emit(
                        "cache.lookup",
                        request_id=request.request_id,
                        block_ids=blocks.get_block_ids(),
                        num_cached_tokens=num_tokens,
                        block_refs=self.refs(blocks.get_block_ids()),
                    )
                elif name == "allocate_slots" and result is None:
                    self.trace.emit("block.allocation_failed", request_id=request.request_id)
                    # Sliding-window managers may release skipped blocks before
                    # discovering that the remaining allocation cannot fit.
                    if request.request_id in self.requests:
                        self.update(request, self._blocks(request.request_id), "allocation_failed")
                else:
                    self.update(request, () if name == "free" else self._blocks(request.request_id), name)
            except Exception:
                self.trace.emit("trace.observation_error", stage=name, request_id=request.request_id)
                logging.getLogger(__name__).exception("KV trace could not observe cache manager")
            return result

        setattr(self.manager, name, observed)

    def wrap_schedule(self, scheduler: Any) -> None:
        # Resolve the bound override, not just Scheduler.schedule: this also
        # covers Ascend scheduler subclasses that replace schedule entirely.
        original = scheduler.schedule

        @functools.wraps(original)
        def observed(*args, **kwargs):
            output = original(*args, **kwargs)
            if not self.trace.enabled:
                return output
            try:
                self.step += 1
                for request_id in output.finished_req_ids:
                    self.revisions.pop(request_id, None)
                computed = {
                    req.request_id: req.num_computed_tokens for req in getattr(output, "scheduled_new_reqs", ())
                }
                cached = getattr(output, "scheduled_cached_reqs", None)
                if cached is not None:
                    computed.update(zip(cached.req_ids, cached.num_computed_tokens))
                requests = []
                for request_id, num_tokens in output.num_scheduled_tokens.items():
                    blocks = self._blocks(request_id)
                    refs = self.refs(blocks)
                    signature = (tuple(tuple(ids) for ids in blocks), tuple(r["alloc_epoch"] for r in refs))
                    previous, revision = self.revisions.get(request_id, (None, 0))
                    revision += signature != previous
                    self.revisions[request_id] = (signature, revision)
                    requests.append(
                        {
                            "request_id": request_id,
                            "mapping_revision": revision,
                            "block_ids": blocks,
                            "block_refs": refs,
                            "num_scheduled_tokens": num_tokens,
                            "num_computed_tokens": computed.get(request_id),
                        }
                    )
                context = {
                    "context_version": 1,
                    "run_id": self.trace.config.run_id,
                    "scheduler_id": self.trace.trace_id,
                    "step_id": f"{self.trace.trace_id}:{self.step}",
                    "pool_id": self.pool.pool_id if self.pool else None,
                    "null_block_id": self.null_id,
                    "requests": requests,
                }
                # The pinned multiproc MessageQueue pickles SchedulerOutput,
                # preserving attributes and subclasses. Other transports must
                # demonstrate this contract or workers report missing context.
                context["dispatch_event_id"] = self.trace.emit("schedule.dispatch", **context)
                output._ascend_kv_trace_context = context
            except Exception:
                self.trace.emit("trace.observation_error", stage="schedule.dispatch")
                logging.getLogger(__name__).exception("KV trace could not attach scheduler context")
            return output

        scheduler.schedule = observed


def attach_manager_trace(manager: Any, trace: Any) -> None:
    if getattr(manager, "_ascend_kv_trace", None) is not None:
        return
    observer = ManagerTrace(manager, trace)
    manager._ascend_kv_trace = observer
    for name in ("allocate_slots", "free", "cache_blocks", "get_computed_blocks"):
        observer.wrap(name)
    trace.emit(
        "cache.config",
        num_blocks=len(manager.block_pool.blocks),
        groups=[
            {"group_id": i, "block_size": group.kv_cache_spec.block_size, "layers": group.layer_names}
            for i, group in enumerate(manager.kv_cache_config.kv_cache_groups)
        ],
    )
