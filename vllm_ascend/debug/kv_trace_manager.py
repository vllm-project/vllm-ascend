# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe request ownership at the KVCacheManager boundary, without a proxy."""

from __future__ import annotations

import functools
import logging
from typing import Any


class ManagerTrace:
    def __init__(self, manager: Any, trace: Any):
        self.manager = manager
        self.trace = trace
        self.requests: dict[str, tuple[list[int], ...]] = {}
        self.owners: dict[tuple[int, int], set[str]] = {}
        self.leases: dict[tuple[int, int], int] = {}

    def _blocks(self, request_id: str) -> tuple[list[int], ...]:
        return tuple(list(group) for group in self.manager.get_block_ids(request_id))

    @staticmethod
    def _keys(blocks):
        return {(group, block) for group, ids in enumerate(blocks) for block in ids if block != 0}

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
