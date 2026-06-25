# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Patch: Enable KV Consumer partial-group caching for hybrid Mamba models.

When running P/D disaggregated inference on hybrid attention models
(FullAttention + Mamba, e.g. Jamba), the KV consumer side needs special
handling because Mamba states are *not* transferred via the KV connector.
Only the FullAttention KV cache should participate in prefix caching on
the consumer.

This patch monkey-patches four upstream vLLM modules so that, when the
current instance is a KV consumer with prefix caching enabled:

1. ``HybridKVCacheCoordinator.__init__``: sets the
   ``enable_kv_consumer_partial_group_caching`` flag and overrides
   ``lcm_block_size`` to ``hash_block_size``.
2. ``HybridKVCacheCoordinator.find_longest_cache_hit``: uses only the
   FullAttention hit length (ignoring Mamba hit results).
3. ``KVCacheManager.__init__``: threads ``vllm_config`` through to the
   coordinator factory.
4. ``get_kv_cache_coordinator``: accepts and forwards ``vllm_config``.
5. ``Scheduler.__init__``: passes ``vllm_config`` to ``KVCacheManager``
   and disables ``need_mamba_block_aligned_split`` for KV consumers.
6. ``Scheduler._mamba_block_aligned_split``: removes the assertion that
   blocks external computed tokens.
"""

import vllm.v1.core.kv_cache_coordinator
import vllm.v1.core.kv_cache_manager
import vllm.v1.core.sched.scheduler
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_coordinator import (
    FullAttentionSpec,
    HybridKVCacheCoordinator,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheSpec

# ---------------------------------------------------------------------------
# 1. Patch HybridKVCacheCoordinator.__init__
# ---------------------------------------------------------------------------
_orig_hybrid_init = HybridKVCacheCoordinator.__init__


def _patched_hybrid_init(self, *args, vllm_config: VllmConfig | None = None, **kwargs):
    """Wrap original __init__ to set partial-group caching flag."""
    _orig_hybrid_init(self, *args, **kwargs)

    # After original init, check if we should enable partial group caching.
    self.enable_kv_consumer_partial_group_caching = False
    if (
        vllm_config is not None
        and vllm_config.kv_transfer_config is not None
        and self.enable_caching
        and vllm_config.kv_transfer_config.is_kv_consumer
    ):
        self.enable_kv_consumer_partial_group_caching = True
        # Override lcm_block_size: on the consumer, only FullAttention
        # groups are cached, so the alignment granularity only needs to
        # be hash_block_size instead of lcm(all block sizes).
        self.lcm_block_size = self.hash_block_size
        logger.info(
            "KV consumer partial-group caching enabled: lcm_block_size overridden to hash_block_size=%d",
            self.hash_block_size,
        )


HybridKVCacheCoordinator.__init__ = _patched_hybrid_init
vllm.v1.core.kv_cache_coordinator.HybridKVCacheCoordinator.__init__ = _patched_hybrid_init

# ---------------------------------------------------------------------------
# 2. Patch HybridKVCacheCoordinator.find_longest_cache_hit
# ---------------------------------------------------------------------------


def _patched_find_longest_cache_hit(
    self,
    block_hashes: list[BlockHash],
    max_cache_hit_length: int,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    """Patched version that uses FullAttention-only hit length for KV consumers."""

    def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
        if kv_cache_spec.block_size == self.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(block_hashes, self.hash_block_size, kv_cache_spec.block_size)

    num_groups = len(self.kv_cache_config.kv_cache_groups)
    hit_length = max_cache_hit_length
    hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

    is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(self.attention_groups[0][0], FullAttentionSpec)

    eagle_verified: set[int] = set()
    full_attn_hit_length = 0

    while True:
        curr_hit_length = hit_length

        for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
            cached_blocks = hit_blocks_by_group[group_ids[0]]
            if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                curr_hit_length = curr_hit_length // spec.block_size * spec.block_size
                continue

            use_eagle = idx in self.eagle_attn_group_indices and idx not in eagle_verified

            _max_length = curr_hit_length
            if use_eagle:
                _max_length = min(curr_hit_length + spec.block_size, max_cache_hit_length)
            hit_blocks = manager_cls.find_longest_cache_hit(
                block_hashes=_get_block_hashes(spec),
                max_length=_max_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=spec,
                use_eagle=use_eagle,
                alignment_tokens=self.lcm_block_size,
            )
            _new_hit_length = len(hit_blocks[0]) * spec.block_size
            if use_eagle:
                eagle_verified.add(idx)
            elif _new_hit_length < curr_hit_length:
                eagle_verified.clear()
            curr_hit_length = _new_hit_length
            for group_id, blocks in zip(group_ids, hit_blocks):
                hit_blocks_by_group[group_id] = blocks

            # Track FullAttention hit length separately for KV consumer mode.
            if isinstance(spec, FullAttentionSpec):
                full_attn_hit_length = curr_hit_length

        if curr_hit_length >= hit_length:
            break
        hit_length = curr_hit_length
        if is_simple_hybrid:
            break

    # For KV consumers, the final hit length is determined solely by
    # FullAttention groups since Mamba states are not transferred.
    if getattr(self, "enable_kv_consumer_partial_group_caching", False):
        hit_length = full_attn_hit_length

    # Truncate full attention blocks to final hit_length (if present)
    spec, group_ids, _ = self.attention_groups[0]
    if isinstance(spec, FullAttentionSpec):
        num_blocks = hit_length // spec.block_size
        for group_id in group_ids:
            if (blks := hit_blocks_by_group[group_id]) is not None:
                del blks[num_blocks:]

    return tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group), hit_length


HybridKVCacheCoordinator.find_longest_cache_hit = _patched_find_longest_cache_hit
vllm.v1.core.kv_cache_coordinator.HybridKVCacheCoordinator.find_longest_cache_hit = _patched_find_longest_cache_hit


# ---------------------------------------------------------------------------
# 3. Patch get_kv_cache_coordinator — accept and forward vllm_config
# ---------------------------------------------------------------------------
_orig_get_kv_cache_coordinator = vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator


def _patched_get_kv_cache_coordinator(*args, vllm_config: VllmConfig | None = None, **kwargs):
    """Wrap factory to forward vllm_config to HybridKVCacheCoordinator."""
    # The original function uses positional/keyword args. We intercept
    # vllm_config and pass it through only when creating Hybrid coordinator.
    result = _orig_get_kv_cache_coordinator(*args, **kwargs)

    # If a HybridKVCacheCoordinator was created, re-init with vllm_config
    # to set the partial-group caching flag.
    if isinstance(result, HybridKVCacheCoordinator) and vllm_config is not None:
        # The coordinator already ran __init__ via the original factory,
        # but without vllm_config. We now apply the partial-group logic:
        result.enable_kv_consumer_partial_group_caching = False
        if (
            vllm_config.kv_transfer_config is not None
            and result.enable_caching
            and vllm_config.kv_transfer_config.is_kv_consumer
        ):
            result.enable_kv_consumer_partial_group_caching = True
            result.lcm_block_size = result.hash_block_size
            logger.info(
                "KV consumer partial-group caching enabled: lcm_block_size overridden to hash_block_size=%d",
                result.hash_block_size,
            )

    return result


vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = _patched_get_kv_cache_coordinator

# Also patch the direct import in kv_cache_manager module.
vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator = _patched_get_kv_cache_coordinator


# ---------------------------------------------------------------------------
# 4. Patch KVCacheManager.__init__ — accept and forward vllm_config
# ---------------------------------------------------------------------------
_orig_kv_cache_manager_init = KVCacheManager.__init__


def _patched_kv_cache_manager_init(self, *args, vllm_config: VllmConfig | None = None, **kwargs):
    """Wrap KVCacheManager.__init__ to forward vllm_config to coordinator."""
    _orig_kv_cache_manager_init(self, *args, **kwargs)

    # After original init, if vllm_config was provided, re-configure the
    # coordinator for KV consumer partial-group caching.
    if vllm_config is not None and isinstance(self.coordinator, HybridKVCacheCoordinator):
        self.coordinator.enable_kv_consumer_partial_group_caching = False
        if (
            vllm_config.kv_transfer_config is not None
            and self.coordinator.enable_caching
            and vllm_config.kv_transfer_config.is_kv_consumer
        ):
            self.coordinator.enable_kv_consumer_partial_group_caching = True
            self.coordinator.lcm_block_size = self.coordinator.hash_block_size


KVCacheManager.__init__ = _patched_kv_cache_manager_init
vllm.v1.core.kv_cache_manager.KVCacheManager.__init__ = _patched_kv_cache_manager_init


# ---------------------------------------------------------------------------
# 5. Patch Scheduler.__init__ — pass vllm_config and fix mamba alignment
# ---------------------------------------------------------------------------
_orig_scheduler_init = Scheduler.__init__


def _patched_scheduler_init(self, *args, **kwargs):
    """Wrap Scheduler.__init__ to pass vllm_config and disable mamba alignment for KV consumers."""
    _orig_scheduler_init(self, *args, **kwargs)

    # After original init, pass vllm_config through to KVCacheManager's
    # coordinator, and disable mamba block alignment for KV consumers.
    vllm_config = self.vllm_config
    if vllm_config.kv_transfer_config is not None and isinstance(
        self.kv_cache_manager.coordinator, HybridKVCacheCoordinator
    ):
        coordinator = self.kv_cache_manager.coordinator
        coordinator.enable_kv_consumer_partial_group_caching = False
        if coordinator.enable_caching and vllm_config.kv_transfer_config.is_kv_consumer:
            coordinator.enable_kv_consumer_partial_group_caching = True
            coordinator.lcm_block_size = coordinator.hash_block_size

    # Disable mamba block-aligned split for KV consumers:
    # The consumer receives KV from the producer without Mamba states,
    # so block-aligned splitting is not needed and would block external
    # computed tokens.
    if (
        self.need_mamba_block_aligned_split
        and vllm_config.kv_transfer_config is not None
        and vllm_config.kv_transfer_config.is_kv_consumer
    ):
        self.need_mamba_block_aligned_split = False


Scheduler.__init__ = _patched_scheduler_init
vllm.v1.core.sched.scheduler.Scheduler.__init__ = _patched_scheduler_init


# ---------------------------------------------------------------------------
# 6. Patch Scheduler._mamba_block_aligned_split — remove external token assert
# ---------------------------------------------------------------------------
_orig_mamba_block_aligned_split = Scheduler._mamba_block_aligned_split


def _patched_mamba_block_aligned_split(
    self,
    request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """Remove assertion blocking external computed tokens for KV consumers."""
    # Original asserts num_external_computed_tokens == 0.
    # For KV consumers, external computed tokens are expected.
    num_computed_tokens = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
        block_size = self.cache_config.block_size
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
        num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
        if num_computed_tokens_after_sched < last_cache_position:
            num_new_tokens = num_new_tokens // block_size * block_size
        elif num_computed_tokens < last_cache_position < num_computed_tokens_after_sched:
            num_new_tokens = last_cache_position - num_computed_tokens
        else:
            pass
    return num_new_tokens


Scheduler._mamba_block_aligned_split = _patched_mamba_block_aligned_split
vllm.v1.core.sched.scheduler.Scheduler._mamba_block_aligned_split = _patched_mamba_block_aligned_split
