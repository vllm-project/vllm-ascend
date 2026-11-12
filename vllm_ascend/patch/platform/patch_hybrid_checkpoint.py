# SPDX-License-Identifier: Apache-2.0
"""Runtime port of FlashStates hybrid-state frontier checkpoints.

The upstream implementation lives in vLLM core (v0.33, PR #66881); vllm-ascend
is a plugin, so this module applies the same scheduler / KV-cache hooks at
import time. It is gated by ``additional_config["flashstate"]`` and is inert
unless enabled.

This is the first (exact-end frontier) stage:
- a finished request's recurrent state at its exact end is cached under a
  frontier key;
- the next request extending that prefix resumes there;
- GreedyDual-Size eviction is optional.

Decode-boundary snapshots are taken here too: vLLM 0.32 has no
``MambaManager.decode_boundary_scratch`` / ``queue_block_copy`` /
``state_cached_listener``, so this module installs them as thin shims over the
existing ``_pending_cow_copies`` device-copy path. In-chunk prefill export,
speculative compact state and the Mooncake store tier are not ported.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, cast

from vllm.logger import logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager

from vllm_ascend.hybrid_checkpoint.compat import install_block_hash_map_compat
from vllm_ascend.hybrid_checkpoint.cost import PrefillCostModel
from vllm_ascend.hybrid_checkpoint.manager import (
    HybridCheckpointManager,
    check_frontier_checkpoint_config,
)

_FRONTIER_HIT_ATTR = "_flashstate_pending_hit"


def _settings(vllm_config: Any) -> dict[str, Any]:
    additional = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional, dict):
        return {}
    settings = additional.get("flashstate", {})
    return settings if isinstance(settings, dict) else {}


def _enabled(vllm_config: Any) -> bool:
    return bool(_settings(vllm_config).get("enable_frontier", False))


# ---------------------------------------------------------------------------
# BlockPool: keys that are not the hash of the block's own token range.
# ---------------------------------------------------------------------------
_original_block_pool_init = BlockPool.__init__
_original_remove_cached_block_hashes = BlockPool._remove_cached_block_hashes


def _block_pool_init(self: BlockPool, *args, **kwargs) -> None:
    _original_block_pool_init(self, *args, **kwargs)
    # State owned by the checkpoint manager; None while it is off.
    self.removal_cause = cast("str | None", None)
    self.removal_causes = False
    self.eviction_policy = None
    self.key_removal_listener = None


@contextmanager
def _removing(self: BlockPool, cause: str = "allocation"):
    """Name the cause of the removals made inside the block."""
    previous = self.removal_cause
    self.removal_cause = cause
    try:
        yield
    finally:
        self.removal_cause = previous


def _cache_block_under_key(
    self: BlockPool,
    block: KVCacheBlock,
    block_hash_with_group_id: BlockHashWithGroupId,
    num_tokens: int,
    request: Any = None,
    parent_block_hash: BlockHash | None = None,
) -> None:
    assert not block.is_null
    self._insert_block_hash(block_hash_with_group_id, block, num_tokens=num_tokens)


def _uncache_key(
    self: BlockPool,
    block: KVCacheBlock,
    block_hash_with_group_id: BlockHashWithGroupId,
) -> None:
    keys = self.cached_block_hashes_by_block.get(block.block_id)
    if not keys or block_hash_with_group_id not in keys:
        return
    keys.remove(block_hash_with_group_id)
    if not keys:
        del self.cached_block_hashes_by_block[block.block_id]
    self.cached_block_hash_to_block.pop(block_hash_with_group_id, block.block_id)
    self._notify_keys_removed([block_hash_with_group_id])


def _notify_keys_removed(self: BlockPool, block_hashes: list[BlockHashWithGroupId]) -> None:
    listener = getattr(self, "key_removal_listener", None)
    if listener is not None and block_hashes:
        listener(block_hashes)


def _remove_cached_block_hashes(self: BlockPool, block: KVCacheBlock):
    """Every eviction path funnels through here in vLLM 0.32, so the frontier
    index learns about removals that its own ``uncache_key`` did not make."""
    removed = _original_remove_cached_block_hashes(self, block)
    if removed:
        self._notify_keys_removed(removed)
    return removed


def _install_block_pool_patches() -> None:
    BlockPool.__init__ = _block_pool_init  # type: ignore[method-assign]
    BlockPool._remove_cached_block_hashes = _remove_cached_block_hashes  # type: ignore[method-assign]
    if not hasattr(BlockPool, "removing"):
        BlockPool.removing = _removing  # type: ignore[attr-defined]
    if not hasattr(BlockPool, "cache_block_under_key"):
        BlockPool.cache_block_under_key = _cache_block_under_key  # type: ignore[attr-defined]
    if not hasattr(BlockPool, "uncache_key"):
        BlockPool.uncache_key = _uncache_key  # type: ignore[attr-defined]
    if not hasattr(BlockPool, "_notify_keys_removed"):
        BlockPool._notify_keys_removed = _notify_keys_removed  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# MambaManager: the device-state hooks the Ascend port has to supply.
# ---------------------------------------------------------------------------


def _upstream_mamba_manager() -> type[MambaManager]:
    """Return the upstream ``MambaManager`` rather than an Ascend subclass.

    ``patch_mamba_manager`` rebinds the module-level ``MambaManager`` name to
    ``AscendMambaManager``, and that patch is applied before this module is
    imported. Hooking the subclass would leave the base class untouched, which
    both breaks the "Ascend manager must not shadow ``__init__``" guard test and
    silently skips any other subclass of ``MambaManager``.
    """
    base = getattr(MambaManager, "__base__", None)
    if MambaManager.__name__ != "MambaManager" and base is not None and base.__name__ == "MambaManager":
        return base
    return MambaManager


_MambaManager = _upstream_mamba_manager()
_original_mamba_init = _MambaManager.__init__
_original_mamba_cache_blocks = _MambaManager.cache_blocks


def _mamba_init(self: MambaManager, *args, **kwargs) -> None:
    _original_mamba_init(self, *args, **kwargs)
    # Request -> scratch block holding the running state copied at a block
    # boundary the output crossed, and the request's computed tokens then.
    self.decode_boundary_scratch = cast("dict[str, tuple[KVCacheBlock, int]]", {})
    # Called with each state this group caches, its token count and the
    # request that cached it.
    self.state_cached_listener = None


def _queue_block_copy(self: MambaManager, source: KVCacheBlock, target: KVCacheBlock) -> None:
    """Copy ``source`` into ``target`` before the next forward. Both take a
    reference that is released once the copy has run."""
    source.ref_cnt += 1
    target.ref_cnt += 1
    self._pending_cow_copies.append((source, target))


def _mamba_cache_blocks(self: MambaManager, request: Any, num_tokens: int, *args, **kwargs) -> None:
    _original_mamba_cache_blocks(self, request, num_tokens, *args, **kwargs)
    listener = getattr(self, "state_cached_listener", None)
    if listener is None:
        return
    for grouped in self.cached_blocks_this_step:
        listener(get_block_hash(grouped), num_tokens, request)


def _install_mamba_patches() -> None:
    _MambaManager.__init__ = _mamba_init  # type: ignore[method-assign]
    if not hasattr(_MambaManager, "queue_block_copy"):
        _MambaManager.queue_block_copy = _queue_block_copy  # type: ignore[attr-defined]
    _MambaManager.cache_blocks = _mamba_cache_blocks  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# KVCacheManager: extend a block-aligned hit with a frontier.
# ---------------------------------------------------------------------------
_original_get_computed_blocks = KVCacheManager.get_computed_blocks
_original_allocate_slots = KVCacheManager.allocate_slots


def _get_computed_blocks(self: KVCacheManager, request: Any, max_cache_hit_length: int | None = None):
    blocks, num_new_computed_tokens, shared_prefix_boundary = _original_get_computed_blocks(
        self, request, max_cache_hit_length
    )
    checkpoints = getattr(self, "checkpoints", None)
    if checkpoints is None:
        return blocks, num_new_computed_tokens, shared_prefix_boundary

    aligned_hit = num_new_computed_tokens
    # Always look for a frontier: the first request of a session has no
    # recorded lookup yet, and gating on one would hide every checkpoint.
    extended, num_new_computed_tokens = checkpoints.extend_hit(
        request,
        blocks.blocks,
        num_new_computed_tokens,
        request.num_tokens - 1,
    )
    if num_new_computed_tokens != aligned_hit:
        blocks = KVCacheBlocks(extended)
        if shared_prefix_boundary and shared_prefix_boundary <= num_new_computed_tokens:
            # The frontier resume passes the junction the lagging group missed.
            shared_prefix_boundary = 0

    # ``allocated`` follows once the scheduler allocates the blocks; remember
    # the local hit length for it.
    setattr(request, _FRONTIER_HIT_ATTR, num_new_computed_tokens)
    checkpoints.admitted(request, num_new_computed_tokens, blocks, 0)
    return blocks, num_new_computed_tokens, shared_prefix_boundary


def _snapshot_decode_boundary(checkpoints: HybridCheckpointManager, request: Any) -> None:
    """Copy the running state into the request's scratch block when the output
    reached a block boundary. Runs before this step's allocation, so the
    queued device copy still sees the pre-step state."""
    computed = request.num_computed_tokens
    if computed <= request.num_prompt_tokens:
        return
    if computed % checkpoints.block_size:
        return
    sources = checkpoints.decode_boundary_sources(request)
    if sources is None:
        return
    checkpoints.snapshot_decode_boundary(request, sources)


def _allocate_slots(self: KVCacheManager, request: Any, *args, **kwargs):
    checkpoints = getattr(self, "checkpoints", None)
    if checkpoints is not None:
        try:
            _snapshot_decode_boundary(checkpoints, request)
        except Exception:
            logger.exception("flashstate: failed to snapshot a decode boundary")
    out = _original_allocate_slots(self, request, *args, **kwargs)
    if checkpoints is not None:
        hit = getattr(request, _FRONTIER_HIT_ATTR, None)
        if hit is not None:
            delattr(request, _FRONTIER_HIT_ATTR)
            if out is not None:
                checkpoints.allocated(request, hit)
    return out


# ---------------------------------------------------------------------------
# Scheduler: create the manager, publish on finish, release scratch blocks.
# ---------------------------------------------------------------------------
_original_scheduler_init = Scheduler.__init__
_original_free_request = Scheduler._free_request
_original_preempt_request = Scheduler._preempt_request
_original_reset_prefix_cache = Scheduler.reset_prefix_cache


# ``functools.wraps`` keeps ``inspect.signature`` on the upstream (or
# balance-scheduler subclass) parameter list, which the drop-in signature guard
# in tests/ut/patch/platform/test_patch_balance_schedule.py checks: the engine
# constructs the scheduler with keyword arguments.
@functools.wraps(_original_scheduler_init)
def _scheduler_init(self: Scheduler, *args, **kwargs) -> None:
    _original_scheduler_init(self, *args, **kwargs)

    vllm_config = getattr(self, "vllm_config", None)
    if vllm_config is None or not _enabled(vllm_config):
        self.mamba_checkpoints = None
        return

    if vllm_config.speculative_config is not None:
        raise ValueError("flashstate frontier checkpoints do not support speculative decoding yet")
    check_frontier_checkpoint_config(
        vllm_config,
        block_size=self.cache_config.block_size,
        use_v2_model_runner=self.use_v2_model_runner,
        enable_kv_cache_events=self.enable_kv_cache_events,
    )

    settings = _settings(vllm_config)
    eviction_cost = None
    if settings.get("prefix_cache_eviction", "lru") == "gds":
        coefficients = settings.get("prefill_cost_coefficients")
        if not coefficients:
            raise ValueError(
                'flashstate additional_config: prefix_cache_eviction="gds" requires prefill_cost_coefficients'
            )
        eviction_cost = PrefillCostModel(*coefficients)

    from vllm.utils.hashing import get_hash_fn_by_name

    self.mamba_checkpoints = HybridCheckpointManager(
        self.kv_cache_manager.coordinator,
        get_hash_fn_by_name(self.cache_config.prefix_caching_hash_algo),
        waiting=lambda: self.waiting,
        eviction_cost=eviction_cost,
        eviction_leaf_last=bool(settings.get("prefix_cache_leaf_last", False)),
        eviction_leaf_idle_s=settings.get("prefix_cache_leaf_idle_seconds"),
        speculative_decoding=False,
    )
    self.kv_cache_manager.checkpoints = self.mamba_checkpoints
    logger.info(
        "flashstate: frontier checkpoints enabled (gds=%s, leaf_last=%s)",
        eviction_cost is not None,
        bool(settings.get("prefix_cache_leaf_last", False)),
    )


def _free_request(self: Scheduler, request: Any, delay_free_blocks: bool = False):
    checkpoints = getattr(self, "mamba_checkpoints", None)
    if checkpoints is not None:
        try:
            checkpoints.request_finished(request)
        except Exception:
            logger.exception("flashstate: failed to publish frontier checkpoint")
    return _original_free_request(self, request, delay_free_blocks)


def _preempt_request(self: Scheduler, request: Any, timestamp: Any, drop_stale_output: bool = False):
    checkpoints = getattr(self, "mamba_checkpoints", None)
    if checkpoints is not None:
        try:
            checkpoints.drop_decode_boundary_scratch(request.request_id)
        except Exception:
            logger.exception("flashstate: failed to drop a scratch block")
    return _original_preempt_request(self, request, timestamp, drop_stale_output)


def _reset_prefix_cache(self: Scheduler, *args, **kwargs):
    reset = _original_reset_prefix_cache(self, *args, **kwargs)
    checkpoints = getattr(self, "mamba_checkpoints", None)
    if reset and checkpoints is not None:
        try:
            checkpoints.reset()
        except Exception:
            logger.exception("flashstate: failed to reset checkpoint state")
    return reset


def _install() -> None:
    install_block_hash_map_compat()
    _install_block_pool_patches()
    _install_mamba_patches()
    KVCacheManager.get_computed_blocks = _get_computed_blocks  # type: ignore[method-assign]
    KVCacheManager.allocate_slots = _allocate_slots  # type: ignore[method-assign]
    Scheduler.__init__ = _scheduler_init  # type: ignore[method-assign]
    Scheduler._free_request = _free_request  # type: ignore[method-assign]
    Scheduler._preempt_request = _preempt_request  # type: ignore[method-assign]
    Scheduler.reset_prefix_cache = _reset_prefix_cache  # type: ignore[method-assign]


_install()
