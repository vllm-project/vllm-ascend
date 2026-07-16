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
from vllm.v1.kv_cache_interface import SlidingWindowSpec

try:
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
except ImportError:  # pragma: no cover - older vLLM without DSv4 MLA SWA spec
    SlidingWindowMLASpec = SlidingWindowSpec
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


def _patch_free_queue_prepend() -> None:
    if hasattr(FreeKVCacheBlockQueue, "prepend_n"):
        return

    def prepend_n(self: FreeKVCacheBlockQueue, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks at the front of the free list."""
        if len(blocks) == 0:
            return

        first_block = self.fake_free_list_head.next_free_block
        assert first_block is not None, (
            "next_free_block of fake_free_list_head should always exist"
        )

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
        freed_blocks = [
            block for block in blocks_list if block.ref_cnt == 0 and not block.is_null
        ]
        if prepend:
            self.free_block_queue.prepend_n(freed_blocks)
        else:
            self.free_block_queue.append_n(freed_blocks)

    BlockPool.free_blocks = free_blocks
    logger.debug("Patched BlockPool.free_blocks(prepend=...).")


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


def _patch_cache_full_blocks_mask() -> None:
    """Add a ``block_mask`` argument to ``BlockPool.cache_full_blocks``.

    The vLLM v0.21.0 base ``cache_full_blocks`` has no ``block_mask`` parameter
    (vLLM PR #43447 is built on a newer base that already carries it), so the
    sparse SWA retention mask computed in ``SlidingWindowManager.cache_blocks``
    would have nowhere to land. This wraps the original so that:

    * ``block_mask is None`` -> the original function is called verbatim, i.e.
      every full block is cached exactly as before (byte-for-byte safety net for
      the env-unset / dense path).
    * ``block_mask`` provided -> blocks whose ``mask[j]`` is ``False`` (``j`` is
      the 0-based index into ``new_full_blocks``, i.e. the ``blocks`` slice
      ``[num_cached_blocks:num_full_blocks]``) are temporarily replaced with the
      null block so the original function's ``is_null`` branch skips them. The
      ``blocks`` list is restored afterwards so the request's block table is not
      polluted; masked-out blocks keep ``block_hash is None`` (scratch) and are
      reclaimed with front-insert priority by the existing free patches.
    """
    if "block_mask" in inspect.signature(BlockPool.cache_full_blocks).parameters:
        return
    if getattr(BlockPool.cache_full_blocks, "_vllm_ascend_block_mask_patch", False):
        return

    original_cache_full_blocks = BlockPool.cache_full_blocks

    def cache_full_blocks(
        self: BlockPool,
        *,
        request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        if block_mask is None:
            return original_cache_full_blocks(
                self,
                request=request,
                blocks=blocks,
                num_cached_blocks=num_cached_blocks,
                num_full_blocks=num_full_blocks,
                block_size=block_size,
                kv_cache_group_id=kv_cache_group_id,
            )

        # Mask mode: temporarily swap masked blocks for the null block so the
        # original function's "skip null" branch leaves them uncached, then
        # restore the request's block list verbatim.
        saved: list[tuple[int, KVCacheBlock]] = []
        null_block = self.null_block
        for j in range(num_full_blocks - num_cached_blocks):
            if not block_mask[j]:
                idx = num_cached_blocks + j
                saved.append((idx, blocks[idx]))
                blocks[idx] = null_block
        try:
            original_cache_full_blocks(
                self,
                request=request,
                blocks=blocks,
                num_cached_blocks=num_cached_blocks,
                num_full_blocks=num_full_blocks,
                block_size=block_size,
                kv_cache_group_id=kv_cache_group_id,
            )
        finally:
            for idx, blk in saved:
                blocks[idx] = blk

    cache_full_blocks._vllm_ascend_block_mask_patch = True
    BlockPool.cache_full_blocks = cache_full_blocks
    logger.debug("Patched BlockPool.cache_full_blocks(block_mask=...).")


def _patch_sliding_window_mask() -> None:
    """Inject the retention three-state mask onto ``SlidingWindowManager``.

    vLLM v0.21.0's ``SlidingWindowManager`` has neither ``reachable_block_mask``
    nor ``_contiguous_blocks_for_hit`` (the cache-hit side inlines
    ``cdiv(sliding_window-1, block_size)`` + EAGLE ``+1``), and the base
    ``cache_blocks`` takes no retention argument. We therefore inject all three
    pieces of vLLM PR #43447's algorithm here.

    Deliberate deviation from PR #43447 (see §2c of the design): when
    ``retention_interval is None`` (env unset) we return ``None`` to keep the
    prior dense "cache every full block" behavior byte-for-byte, rather than the
    PR's dense-per-segment sparsification which would tighten the existing SWA
    hit rate. Only the 0 / >0 states walk the sparse path.
    """
    if hasattr(SlidingWindowManager, "reachable_block_mask") and _source_contains(
        SlidingWindowManager.cache_blocks, "retention_interval"
    ):
        return

    @staticmethod
    def _contiguous_blocks_for_hit(
        window_size: int, block_size: int, use_eagle: bool
    ) -> int:
        # Mirror find_longest_cache_hit's hit granularity exactly:
        # cdiv(sliding_window-1, block_size), +1 when EAGLE is enabled. Keeping
        # this identical to the lookup side avoids cache-vs-hit misalignment.
        need = cdiv(window_size - 1, block_size)
        return need + 1 if use_eagle else need

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec,
        use_eagle: bool,
        retention_interval: int | None = None,
        num_prompt_tokens: int | None = None,
    ) -> list[bool] | None:
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        # Deviation from PR #43447: env-unset (None) keeps the dense cache-all
        # path so existing hit behavior is unchanged. cache_full_blocks receives
        # block_mask=None and runs verbatim.
        if retention_interval is None:
            return None
        if alignment_tokens is None:
            return None
        assert alignment_tokens % kv_cache_spec.block_size == 0
        block_size = kv_cache_spec.block_size
        need = cls._contiguous_blocks_for_hit(
            window_size=kv_cache_spec.sliding_window,
            block_size=block_size,
            use_eagle=use_eagle,
        )
        shift = 1 if use_eagle else 0
        mask = [False] * (end_block - start_block)

        # retention_interval == 0 -> only the latest replay boundary is kept
        # (segment tail loop disabled); >0 -> keep one tail per interval segment.
        segment_tokens = None if retention_interval == 0 else retention_interval
        if segment_tokens is not None:
            per_segment = segment_tokens // block_size
            if need >= per_segment:
                # The whole segment is within reach; nothing can be dropped, so
                # fall back to caching everything (dense).
                return None
            for i in range(start_block, end_block):
                if i >= shift and (i - shift) % per_segment >= per_segment - need:
                    mask[i - start_block] = True

        # Replay tail: always retain the contiguous tail covering the latest
        # completed prompt boundary so a replay of the prompt still hits. This
        # only runs when retention is enabled (matches PR #43447: the dense None
        # path never adds a replay tail).
        if num_prompt_tokens is not None:
            latest = (num_prompt_tokens - 1) // alignment_tokens * alignment_tokens
            prompt_end_block = latest // block_size + shift
            for i in range(
                max(start_block, prompt_end_block - need),
                min(end_block, prompt_end_block),
            ):
                mask[i - start_block] = True
        return mask

    def cache_blocks(
        self: SlidingWindowManager,
        request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        num_cached = self.num_cached_block.get(request.request_id, 0)
        num_full = num_tokens // self.block_size
        if num_cached >= num_full:
            return
        # The coordinator injects scheduler_block_size / use_eagle onto each
        # manager; fall back conservatively so pure-SWA (unitary) coordinators
        # that bypass the Ascend coordinator still behave (dense) correctly.
        alignment_tokens = getattr(self, "scheduler_block_size", self.block_size)
        use_eagle = getattr(self, "use_eagle", False)
        block_mask = self.reachable_block_mask(
            start_block=num_cached,
            end_block=num_full,
            alignment_tokens=alignment_tokens,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=use_eagle,
            retention_interval=retention_interval,
            num_prompt_tokens=request.num_prompt_tokens,
        )
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached,
            num_full_blocks=num_full,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
            block_mask=block_mask,
        )
        self.num_cached_block[request.request_id] = num_full

    SlidingWindowManager._contiguous_blocks_for_hit = _contiguous_blocks_for_hit
    SlidingWindowManager.reachable_block_mask = reachable_block_mask
    SlidingWindowManager.cache_blocks = cache_blocks
    logger.debug(
        "Patched SlidingWindowManager.reachable_block_mask / cache_blocks "
        "(prefix-cache retention)."
    )


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
_patch_cache_full_blocks_mask()
_patch_partial_prefix_cache_cleanup()
_patch_scheduler_copy_blocks()
_patch_remove_skipped_blocks()
_patch_sliding_window_mask()
_patch_sliding_window_free()
