# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Align per-instance cache writes with the effective EAGLE block-drop policy."""

from collections.abc import Sequence
from types import MethodType

from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.request import Request


def _reachable_block_mask_without_eagle_drop(
    self: SlidingWindowManager,
    start_block: int,
    end_block: int,
    alignment_tokens: int | None,
    kv_cache_spec: KVCacheSpec,
    use_eagle: bool,
    retention_interval: int | None = None,
    reachable_boundaries: Sequence[int] = (),
    dcp_world_size: int = 1,
    **kwargs,
) -> list[bool] | None:
    # ``use_eagle`` still describes the model in the manager and coordinator.
    # Only the tail's peek/shift must follow the role-specific lookup policy.
    return type(self).reachable_block_mask(
        start_block=start_block,
        end_block=end_block,
        alignment_tokens=alignment_tokens,
        kv_cache_spec=kv_cache_spec,
        use_eagle=False,
        retention_interval=retention_interval,
        reachable_boundaries=reachable_boundaries,
        dcp_world_size=dcp_world_size,
        **kwargs,
    )


def _cache_partial_tail_block_without_eagle_drop(
    self: FullAttentionManager,
    request: Request,
    num_tokens: int,
) -> None:
    hash_block_size = self.block_pool.hash_block_size
    # A resend is capped at P-1; an extension may reuse all P prompt tokens.
    # A hash-aligned prompt therefore has two legal final hash boundaries.
    # Publish both so FA can prove the Mamba checkpoint used by either lookup.
    boundaries = {
        max(request.num_prompt_tokens - 1, 0) // hash_block_size * hash_block_size,
        request.num_prompt_tokens // hash_block_size * hash_block_size,
    }
    blocks = self.req_to_blocks[request.request_id]
    # Promoting the primary entry to a longer prefix removes its old aliases.
    # Publish the longest entry first, then restore the shorter resend alias.
    for boundary_tokens in sorted(boundaries, reverse=True):
        if boundary_tokens == 0 or boundary_tokens > num_tokens:
            continue
        if boundary_tokens % self.block_size == 0:
            continue
        block_idx = boundary_tokens // self.block_size
        if block_idx >= len(blocks):
            continue
        self.block_pool.cache_partial_block(
            request=request,
            block=blocks[block_idx],
            num_tokens=boundary_tokens,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )


def configure_prefix_cache_write_policy(
    manager: SingleTypeKVCacheManager,
    effective_drop: bool,
) -> None:
    """Adapt skipped-drop writes without changing speculation metadata.

    Call only for roles whose cache-hit lookup suppresses the EAGLE drop.
    Class-level mask callers (for example external cache stores) retain the
    upstream API and behavior; the override belongs to this local manager.
    """
    if effective_drop:
        return
    if isinstance(manager, SlidingWindowManager) and manager.use_eagle:
        manager.reachable_block_mask = MethodType(  # type: ignore[method-assign]
            _reachable_block_mask_without_eagle_drop, manager
        )
    if isinstance(manager, FullAttentionManager) and manager.block_size > manager.block_pool.hash_block_size:
        manager._cache_partial_tail_block = MethodType(  # type: ignore[method-assign]
            _cache_partial_tail_block_without_eagle_drop, manager
        )
