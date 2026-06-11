# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Compatibility patch for prefix-cache block reuse.

This carries the small-core part of vLLM PR #43447 needed by the DeepSeek V4
prefix-cache coordinator patch. Remove it when the supported vLLM version
contains these APIs and behaviors.
"""

import inspect
from collections.abc import Callable, Iterable

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core import kv_cache_manager as kv_cache_manager_mod
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    SlidingWindowManager,
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
                supports_scheduler_block_size = "scheduler_block_size" in inspect.signature(original_get).parameters
            except (TypeError, ValueError):
                supports_scheduler_block_size = False
            if supports_scheduler_block_size:
                coord_kwargs.setdefault("scheduler_block_size", scheduler_block_size)
            return original_get(*coord_args, **coord_kwargs)

        kv_cache_manager_mod.get_kv_cache_coordinator = get_with_scheduler_block_size
        try:
            return original_init(self, *args, **kwargs)
        finally:
            kv_cache_manager_mod.get_kv_cache_coordinator = original_get

    __init__._vllm_ascend_scheduler_block_size_patch = True  # type: ignore[attr-defined]
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

    __init__._vllm_ascend_scheduler_block_size_patch = True  # type: ignore[attr-defined]
    scheduler_cls.__init__ = __init__
    logger.debug("Patched Scheduler.__init__ to pass scheduler_block_size.")


def _patch_free_queue_prepend() -> None:
    if hasattr(FreeKVCacheBlockQueue, "prepend_n"):
        return

    def prepend_n(self: FreeKVCacheBlockQueue, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks at the front of the free list."""
        if len(blocks) == 0:
            return

        first_block = self.fake_free_list_head.next_free_block
        assert first_block is not None, "next_free_block of fake_free_list_head should always exist"

        prev_block = self.fake_free_list_head
        for block in blocks:
            block.prev_free_block = prev_block
            prev_block.next_free_block = block
            prev_block = block

        prev_block.next_free_block = first_block
        first_block.prev_free_block = prev_block

        self.num_free_blocks += len(blocks)

    FreeKVCacheBlockQueue.prepend_n = prepend_n
    logger.debug("Patched FreeKVCacheBlockQueue.prepend_n for prefix-cache reuse.")


def _patch_block_pool_free_blocks() -> None:
    if "prepend" in inspect.signature(BlockPool.free_blocks).parameters:
        return

    def free_blocks(
        self: BlockPool,
        ordered_blocks: Iterable[KVCacheBlock],
        prepend: bool = False,
    ) -> None:
        """Free blocks, optionally placing newly-free blocks at queue front."""
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        freed_blocks = [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        if prepend:
            self.free_block_queue.prepend_n(freed_blocks)
        else:
            self.free_block_queue.append_n(freed_blocks)

    BlockPool.free_blocks = free_blocks
    logger.debug("Patched BlockPool.free_blocks(prepend=...).")


def _patch_remove_skipped_blocks() -> None:
    if _source_contains(
        SingleTypeKVCacheManager.remove_skipped_blocks,
        "prepend=True",
        "block_hash is None",
    ):
        return

    def remove_skipped_blocks(
        self: SingleTypeKVCacheManager,
        request_id: str,
        total_computed_tokens: int,
    ) -> None:
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            return

        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        removed_cached_blocks: list[KVCacheBlock] = []
        removed_uncached_blocks: list[KVCacheBlock] = []

        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] == self._null_block:
                break
            if blocks[i].block_hash is None:
                removed_uncached_blocks.append(blocks[i])
            else:
                removed_cached_blocks.append(blocks[i])
            blocks[i] = self._null_block

        self.block_pool.free_blocks(removed_cached_blocks)
        self.block_pool.free_blocks(removed_uncached_blocks, prepend=True)

    SingleTypeKVCacheManager.remove_skipped_blocks = remove_skipped_blocks
    logger.debug("Patched SingleTypeKVCacheManager.remove_skipped_blocks.")


def _patch_sliding_window_mask() -> None:
    if not hasattr(SlidingWindowManager, "_cache_block_mask"):
        return

    if _source_contains(
        SlidingWindowManager._cache_block_mask,
        "use_eagle",
        "shift = 1 if use_eagle else 0",
    ):
        return

    def _cache_block_mask(
        self: SlidingWindowManager,
        num_cached_blocks: int,
        num_full_blocks: int,
        alignment_tokens: int,
    ) -> list[bool] | None:
        assert alignment_tokens > self.block_size
        per_segment = alignment_tokens // self.block_size
        tail = cdiv(self.sliding_window - 1, self.block_size)
        use_eagle = getattr(self, "use_eagle", False)
        if use_eagle:
            tail += 1
        if tail >= per_segment:
            return None
        skip = per_segment - tail
        shift = 1 if use_eagle else 0
        return [i >= shift and (i - shift) % per_segment >= skip for i in range(num_cached_blocks, num_full_blocks)]

    SlidingWindowManager._cache_block_mask = _cache_block_mask
    logger.debug("Patched SlidingWindowManager._cache_block_mask.")


def _patch_sliding_window_free() -> None:
    if _source_contains(SlidingWindowManager.free, "prepend=True", "block_hash is None"):
        return

    def free(self: SlidingWindowManager, request_id: str) -> None:
        req_blocks = self.req_to_blocks.pop(request_id, [])
        if req_blocks:
            cached_blocks: list[KVCacheBlock] = []
            uncached_blocks: list[KVCacheBlock] = []
            for block in reversed(req_blocks):
                if block.block_hash is None:
                    uncached_blocks.append(block)
                else:
                    cached_blocks.append(block)
            self.block_pool.free_blocks(cached_blocks)
            self.block_pool.free_blocks(uncached_blocks, prepend=True)
        self.num_cached_block.pop(request_id, None)

    SlidingWindowManager.free = free
    logger.debug("Patched SlidingWindowManager.free.")


_patch_kv_cache_manager_scheduler_block_size()
_patch_scheduler_scheduler_block_size()
_patch_free_queue_prepend()
_patch_block_pool_free_blocks()
_patch_remove_skipped_blocks()
_patch_sliding_window_mask()
_patch_sliding_window_free()
