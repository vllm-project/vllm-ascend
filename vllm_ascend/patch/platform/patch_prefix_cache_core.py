# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Compatibility patches required by the DeepSeek V4 partial prefix cache.

This installs only the hooks the partial (sub-block) compressed prefix-cache
feature depends on:

* ``scheduler_block_size`` threading (Scheduler -> KVCacheManager -> coordinator)
  so the coordinator can align cache hits to the scheduler block size instead of
  the raw hash block size.
* the copy-forward hook that surfaces ``coordinator.take_copy_block_ids()`` as
  ``scheduler_output.new_block_ids_to_copy`` for the model runner to materialize.
* the eviction cleanup hook that drops stale partial short-key entries.

The general prefix-cache-core-reuse primitives from vLLM PR #43447 (free-list
``prepend``, sliding-window mask/free) are intentionally not carried here: the
partial path does not use them. Each hook is gated to no-op when the supported
vLLM version already provides the behavior.
"""

import inspect
from collections.abc import Callable

from vllm.logger import logger
from vllm.v1.core import kv_cache_manager as kv_cache_manager_mod
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm_ascend.core.single_type_kv_cache_manager import (
    remove_partial_cache_entries_for_block,
)


def _source_contains(fn: Callable, *needles: str) -> bool:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return all(needle in source for needle in needles)


def _patch_kv_cache_manager_scheduler_block_size() -> None:
    """Forward scheduler cache-hit granularity on vLLM versions without it."""
    kv_cache_manager_cls = kv_cache_manager_mod.KVCacheManager
    current_init = kv_cache_manager_cls.__init__
    if "scheduler_block_size" in inspect.signature(current_init).parameters:
        return
    if getattr(current_init, "_vllm_ascend_scheduler_block_size_patch", False):
        return

    original_init = current_init
    original_signature = inspect.signature(original_init)

    def __init__(
        self,
        *args,
        scheduler_block_size: int | None = None,
        **kwargs,
    ) -> None:
        bound = original_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if scheduler_block_size is None:
            scheduler_block_size = bound.arguments.get("hash_block_size")

        original_get = kv_cache_manager_mod.get_kv_cache_coordinator

        def get_with_scheduler_block_size(*coord_args, **coord_kwargs):
            try:
                supports_scheduler_block_size = (
                    "scheduler_block_size"
                    in inspect.signature(original_get).parameters
                )
            except (TypeError, ValueError):
                supports_scheduler_block_size = False
            if supports_scheduler_block_size:
                coord_kwargs.setdefault(
                    "scheduler_block_size", scheduler_block_size
                )
            return original_get(*coord_args, **coord_kwargs)

        kv_cache_manager_mod.get_kv_cache_coordinator = (
            get_with_scheduler_block_size
        )
        try:
            return original_init(self, *args, **kwargs)
        finally:
            kv_cache_manager_mod.get_kv_cache_coordinator = original_get

    __init__._vllm_ascend_scheduler_block_size_patch = True
    kv_cache_manager_cls.__init__ = __init__
    logger.debug("Patched KVCacheManager.__init__(scheduler_block_size=...).")


def _patch_scheduler_scheduler_block_size() -> None:
    """Pass Scheduler.block_size to KVCacheManager on older vLLM."""
    try:
        from vllm.v1.core.sched import scheduler as scheduler_mod
    except ImportError:
        return

    scheduler_cls = scheduler_mod.Scheduler
    current_init = scheduler_cls.__init__
    if getattr(current_init, "_vllm_ascend_scheduler_block_size_patch", False):
        return
    if _source_contains(current_init, "scheduler_block_size=self.block_size"):
        return
    try:
        if "scheduler_block_size" in inspect.signature(scheduler_mod.KVCacheManager.__init__).parameters:
            # Upstream already threads the hit granularity natively; forwarding
            # the scheduler's hash-sized block here would violate the manager's
            # divisibility assert. Nothing to patch.
            return
    except (TypeError, ValueError):
        pass

    original_init = current_init
    original_signature = inspect.signature(original_init)

    def __init__(self, *args, **kwargs) -> None:
        bound = original_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        scheduler_block_size = bound.arguments.get("block_size")

        original_kv_cache_manager = scheduler_mod.KVCacheManager

        def kv_cache_manager_with_scheduler_block_size(*kv_args, **kv_kwargs):
            if scheduler_block_size is not None:
                kv_kwargs.setdefault("scheduler_block_size", scheduler_block_size)
            return original_kv_cache_manager(*kv_args, **kv_kwargs)

        scheduler_mod.KVCacheManager = kv_cache_manager_with_scheduler_block_size
        try:
            return original_init(self, *args, **kwargs)
        finally:
            scheduler_mod.KVCacheManager = original_kv_cache_manager

    __init__._vllm_ascend_scheduler_block_size_patch = True
    scheduler_cls.__init__ = __init__
    logger.debug("Patched Scheduler.__init__ to pass scheduler_block_size.")


def _patch_partial_prefix_cache_cleanup() -> None:
    current = BlockPool._maybe_evict_cached_block
    if getattr(current, "_vllm_ascend_partial_prefix_cache_cleanup_patch", False):
        return

    original_maybe_evict = current

    def _maybe_evict_cached_block(self: BlockPool, block: KVCacheBlock) -> bool:
        evicted = original_maybe_evict(self, block)
        remove_partial_cache_entries_for_block(self, block.block_id)
        return evicted

    _maybe_evict_cached_block._vllm_ascend_partial_prefix_cache_cleanup_patch = True
    BlockPool._maybe_evict_cached_block = _maybe_evict_cached_block
    logger.debug("Patched BlockPool partial prefix-cache cleanup.")


def _patch_scheduler_copy_blocks() -> None:
    try:
        from vllm.v1.core.sched import scheduler as scheduler_mod
    except ImportError:
        return

    scheduler_classes = [scheduler_mod.Scheduler]
    for module_name, class_name in (
        ("vllm_ascend.core.scheduler_profiling_chunk", "ProfilingChunkScheduler"),
        ("vllm_ascend.core.scheduler_dynamic_batch", "SchedulerDynamicBatch"),
        ("vllm_ascend.patch.platform.patch_balance_schedule", "BalanceScheduler"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            scheduler_classes.append(getattr(module, class_name))
        except Exception:
            continue

    for scheduler_cls in scheduler_classes:
        current = scheduler_cls.schedule
        if getattr(current, "_vllm_ascend_copy_blocks_patch", False):
            continue
        original_schedule = current

        def schedule(self, *args, __original_schedule=original_schedule, **kwargs):
            scheduler_output = __original_schedule(self, *args, **kwargs)
            coordinator = getattr(self.kv_cache_manager, "coordinator", None)
            take_copy_block_ids = getattr(coordinator, "take_copy_block_ids", None)
            if take_copy_block_ids is not None:
                copy_block_ids = take_copy_block_ids()
                if copy_block_ids:
                    scheduler_output.new_block_ids_to_copy = copy_block_ids
            return scheduler_output

        schedule._vllm_ascend_copy_blocks_patch = True
        scheduler_cls.schedule = schedule
    logger.debug("Patched Scheduler.schedule to forward KV copy blocks.")


_patch_kv_cache_manager_scheduler_block_size()
_patch_scheduler_scheduler_block_size()
_patch_partial_prefix_cache_cleanup()
_patch_scheduler_copy_blocks()
