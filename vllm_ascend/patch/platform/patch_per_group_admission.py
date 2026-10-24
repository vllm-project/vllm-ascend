# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Keep independent-pool admission at the upstream pre-allocation gates.

Only demand acquisition and the two capacity predicates are rewritten. The
upstream normalization, window reclamation, prefix touches, external loads,
CoW and cache-commit ordering stay intact. Changed upstream anchors fail
explicitly rather than falling back to unsafe aggregate admission.
"""

from __future__ import annotations

import functools
import inspect
import textwrap

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.patch.platform.patch_per_group_pools import (
    GroupBlockReservation,
    PerGroupBlockPools,
)


def _get_block_demand(coordinator, *args, **kwargs) -> int | tuple[int, ...]:
    if isinstance(coordinator.block_pool, PerGroupBlockPools):
        return coordinator.get_num_blocks_to_allocate_by_group(*args, **kwargs)
    return coordinator.get_num_blocks_to_allocate(*args, **kwargs)


def _can_allocate(
    block_pool,
    demand: int | tuple[int, ...],
    reserved_blocks: int | GroupBlockReservation = 0,
    watermark_blocks: int = 0,
) -> bool:
    if isinstance(block_pool, PerGroupBlockPools):
        if not isinstance(demand, tuple):
            raise TypeError("Independent-pool admission requires per-group demand")
        return block_pool.can_allocate(demand, reserved_blocks, watermark_blocks)
    if not isinstance(demand, int) or not isinstance(reserved_blocks, int):
        raise TypeError("Shared-pool admission requires scalar demand and reservation")
    return demand + watermark_blocks <= block_pool.get_num_free_blocks() - reserved_blocks


def _inflight_reservations(scheduler) -> GroupBlockReservation:
    coordinator = scheduler.kv_cache_manager.coordinator
    reserved = [0] * len(coordinator.single_type_managers)
    for request in scheduler._inflight_prefills:
        full_num_tokens = min(request.num_tokens, scheduler.max_model_len)
        demand = coordinator.get_num_blocks_to_allocate_by_group(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=scheduler.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
            # The pin's own boundary, so reservations see the same demand the
            # request itself will meet at admission.
            prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
        )
        for group_id, blocks in enumerate(demand):
            reserved[group_id] += blocks
    return GroupBlockReservation(tuple(reserved))


def _patch_allocate_slots(manager_cls=KVCacheManager) -> None:
    original = manager_cls.allocate_slots
    if getattr(original, "_ascend_per_group_admission", False):
        return
    source = getattr(original, "_ascend_source", None)
    if source is None:
        source = textwrap.dedent(inspect.getsource(original))
    demand_call = "self.coordinator.get_num_blocks_to_allocate("
    if source.count(demand_call) != 2:
        raise RuntimeError("Cannot install per-group admission: unexpected allocate_slots demand calls")
    source = source.replace(demand_call, "_ascend_get_block_demand(self.coordinator, ")
    replacements = (
        (
            "reserved_blocks: int = 0,",
            "reserved_blocks: int | GroupBlockReservation = 0,",
        ),
        (
            "required_blocks = num_blocks_to_allocate + watermark_blocks\n"
            "        if required_blocks > self.block_pool.get_num_free_blocks():",
            "if not _ascend_can_allocate(\n"
            "            self.block_pool, num_blocks_to_allocate, watermark_blocks=watermark_blocks\n"
            "        ):",
        ),
        (
            "available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks\n"
            "    required_blocks = num_blocks_to_allocate + watermark_blocks\n"
            "    if required_blocks > available_blocks:",
            "if not _ascend_can_allocate(\n"
            "        self.block_pool, num_blocks_to_allocate, reserved_blocks, watermark_blocks\n"
            "    ):",
        ),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError("Cannot install per-group admission: unexpected allocate_slots capacity gate")
        source = source.replace(before, after)
    namespace = dict(original.__globals__)
    namespace.update(
        _ascend_get_block_demand=_get_block_demand,
        _ascend_can_allocate=_can_allocate,
        GroupBlockReservation=GroupBlockReservation,
    )
    result = {}
    exec(compile(source, inspect.getfile(original), "exec"), namespace, result)
    patched = result[original.__name__]
    functools.update_wrapper(patched, original)
    # Other source-based patches must compose from this body, not unwrap it.
    patched._ascend_source = source
    patched._ascend_per_group_admission = True
    manager_cls.allocate_slots = patched


def _patch_inflight_reservations() -> None:
    original = Scheduler._inflight_prefill_reserved_blocks

    @functools.wraps(original)
    def reserved_blocks(self) -> int | GroupBlockReservation:
        if isinstance(self.kv_cache_manager.block_pool, PerGroupBlockPools):
            return _inflight_reservations(self)
        return original(self)

    Scheduler._inflight_prefill_reserved_blocks = reserved_blocks


_patch_allocate_slots()
_patch_inflight_reservations()
