# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import inspect
import os
from collections.abc import Iterable
from math import lcm
from typing import Any

import vllm.envs as vllm_envs
import vllm.v1.core.block_pool as block_pool_mod
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    MambaManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, SlidingWindowSpec
from vllm.v1.request import Request

ENV_NAME = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL"


def _read_retention_interval() -> int | None:
    if ENV_NAME not in os.environ:
        return None
    return int(os.environ[ENV_NAME])


if hasattr(vllm_envs, "environment_variables"):
    vllm_envs.environment_variables.setdefault(ENV_NAME, _read_retention_interval)


def _unwrap_specs(kv_cache_spec: KVCacheSpec) -> Iterable[KVCacheSpec]:
    nested_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
    if isinstance(nested_specs, dict):
        yield from nested_specs.values()
    else:
        yield kv_cache_spec


def _iter_kv_cache_specs(kv_cache_config: KVCacheConfig) -> Iterable[KVCacheSpec]:
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        yield from _unwrap_specs(kv_cache_group.kv_cache_spec)


def _is_sliding_window_spec(kv_cache_spec: KVCacheSpec) -> bool:
    return isinstance(kv_cache_spec, SlidingWindowSpec) or kv_cache_spec.__class__.__name__.endswith(
        "SlidingWindowMLASpec"
    )


def _validate_prefix_cache_retention_interval(
    retention_interval: int | None,
    scheduler_block_size: int,
    kv_cache_config: KVCacheConfig,
) -> None:
    if retention_interval is None:
        return

    if not any(_is_sliding_window_spec(spec) for spec in _iter_kv_cache_specs(kv_cache_config)):
        raise ValueError(
            f"{ENV_NAME} is set but this model has no sliding-window KV cache "
            "group, so retention has no effect. Unset it or use a model with "
            "sliding-window attention."
        )

    if retention_interval < 0 or retention_interval % scheduler_block_size != 0:
        raise ValueError(
            f"{ENV_NAME} ({retention_interval}) must be non-negative and a "
            f"multiple of scheduler_block_size ({scheduler_block_size})."
        )


def get_prefix_cache_retention_interval(
    kv_cache_config: KVCacheConfig,
    scheduler_block_size: int,
) -> int | None:
    retention_interval = getattr(vllm_envs, ENV_NAME, None)
    _validate_prefix_cache_retention_interval(retention_interval, scheduler_block_size, kv_cache_config)
    return retention_interval


def _patch_compress_manager_factory() -> None:
    import vllm.v1.core.kv_cache_coordinator as coordinator_mod
    import vllm.v1.core.single_type_kv_cache_manager as manager_mod

    orig_get_manager = coordinator_mod.get_manager_for_kv_cache_spec
    if getattr(orig_get_manager, "_ascend_compress_patched", False):
        return

    def get_manager_for_kv_cache_spec(
        kv_cache_spec: KVCacheSpec,
        **kwargs: Any,
    ) -> SingleTypeKVCacheManager:
        if (getattr(kv_cache_spec, "compress_ratio", 1) or 1) > 1:
            from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager

            return CompressAttentionManager(kv_cache_spec, **kwargs)
        return orig_get_manager(kv_cache_spec, **kwargs)

    get_manager_for_kv_cache_spec._ascend_compress_patched = True  # type: ignore[attr-defined]
    coordinator_mod.get_manager_for_kv_cache_spec = get_manager_for_kv_cache_spec
    manager_mod.get_manager_for_kv_cache_spec = get_manager_for_kv_cache_spec


def _add_prepend_n_to_free_queue() -> None:
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

    FreeKVCacheBlockQueue.prepend_n = prepend_n  # type: ignore[attr-defined]


def _patch_block_pool_free_blocks() -> None:
    if "prepend" in inspect.signature(BlockPool.free_blocks).parameters:
        return

    def free_blocks(
        self: BlockPool,
        ordered_blocks: Iterable[KVCacheBlock],
        prepend: bool = False,
    ) -> None:
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1

        freed_blocks = [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        if prepend:
            self.free_block_queue.prepend_n(freed_blocks)
        else:
            self.free_block_queue.append_n(freed_blocks)

    BlockPool.free_blocks = free_blocks  # type: ignore[method-assign]


def _patch_block_pool_cache_full_blocks() -> None:
    if "block_mask" in inspect.signature(BlockPool.cache_full_blocks).parameters:
        return

    orig_cache_full_blocks = BlockPool.cache_full_blocks

    def cache_full_blocks(
        self: BlockPool,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        if block_mask is None:
            return orig_cache_full_blocks(
                self,
                request,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                block_size,
                kv_cache_group_id,
            )

        if num_cached_blocks >= num_full_blocks:
            return

        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            block_hashes = request.block_hashes
        else:
            assert block_size % self.hash_block_size == 0
            block_hashes = block_pool_mod.BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[Any] | None = [] if self.enable_kv_cache_events else None
        for i, blk in enumerate(new_full_blocks):
            if not block_mask[i]:
                continue
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]
            block_hash_with_group_id = block_pool_mod.make_block_hash_with_group_id(block_hash, kv_cache_group_id)
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(block_pool_mod.maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            parent_block_hash = (
                None
                if num_cached_blocks == 0
                else block_pool_mod.maybe_convert_block_hash(block_hashes[num_cached_blocks - 1])
            )
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if not block_mask[i - num_cached_blocks] or blocks[i].is_null:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = block_pool_mod.generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                block_pool_mod.BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id if request.lora_request else None,
                    medium=block_pool_mod.MEDIUM_GPU,
                    lora_name=request.lora_request.name if request.lora_request else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                )
            )

    BlockPool.cache_full_blocks = cache_full_blocks  # type: ignore[method-assign]


def _default_reachable_block_mask(
    cls: type[SingleTypeKVCacheManager],
    start_block: int,
    end_block: int,
    alignment_tokens: int | None,
    kv_cache_spec: KVCacheSpec,
    use_eagle: bool,
    retention_interval: int | None = None,
    num_prompt_tokens: int | None = None,
) -> list[bool] | None:
    return None


def _contiguous_blocks_for_hit(window_size: int, block_size: int, use_eagle: bool) -> int:
    need = (window_size - 1 + block_size - 1) // block_size
    if use_eagle:
        need += 1
    return need


def _sliding_window_reachable_block_mask(
    cls: type[SlidingWindowManager],
    start_block: int,
    end_block: int,
    alignment_tokens: int | None,
    kv_cache_spec: KVCacheSpec,
    use_eagle: bool,
    retention_interval: int | None = None,
    num_prompt_tokens: int | None = None,
) -> list[bool] | None:
    assert _is_sliding_window_spec(kv_cache_spec)
    if retention_interval is None:
        return None
    if alignment_tokens is None:
        return None
    assert alignment_tokens % kv_cache_spec.block_size == 0

    block_size = kv_cache_spec.block_size
    need = _contiguous_blocks_for_hit(
        window_size=kv_cache_spec.sliding_window,
        block_size=block_size,
        use_eagle=use_eagle,
    )
    shift = 1 if use_eagle else 0
    mask = [False] * (end_block - start_block)

    segment_tokens = None
    if retention_interval > 0:
        segment_tokens = retention_interval
    if segment_tokens is not None:
        per_segment = segment_tokens // block_size
        if need >= per_segment:
            return None
        for i in range(start_block, end_block):
            if i >= shift and (i - shift) % per_segment >= per_segment - need:
                mask[i - start_block] = True

    if retention_interval is not None and num_prompt_tokens is not None:
        latest = num_prompt_tokens // alignment_tokens * alignment_tokens
        prompt_end_block = latest // block_size + shift
        for i in range(max(start_block, prompt_end_block - need), min(end_block, prompt_end_block)):
            mask[i - start_block] = True

    return mask


def _patch_single_type_cache_blocks() -> None:
    if not hasattr(SingleTypeKVCacheManager, "reachable_block_mask"):
        SingleTypeKVCacheManager.reachable_block_mask = classmethod(_default_reachable_block_mask)  # type: ignore[attr-defined]

    signature = inspect.signature(SingleTypeKVCacheManager.cache_blocks)
    if "retention_interval" not in signature.parameters:

        def cache_blocks(
            self: SingleTypeKVCacheManager,
            request: Request,
            num_tokens: int,
            retention_interval: int | None = None,
            alignment_tokens: int | None = None,
            use_eagle: bool = False,
        ) -> None:
            num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
            num_full_blocks = num_tokens // self.block_size
            if num_cached_blocks >= num_full_blocks:
                return

            block_mask = self.reachable_block_mask(  # type: ignore[attr-defined]
                start_block=num_cached_blocks,
                end_block=num_full_blocks,
                alignment_tokens=alignment_tokens or self.block_size,
                kv_cache_spec=self.kv_cache_spec,
                use_eagle=use_eagle,
                retention_interval=retention_interval,
                num_prompt_tokens=request.num_prompt_tokens,
            )
            self.block_pool.cache_full_blocks(
                request=request,
                blocks=self.req_to_blocks[request.request_id],
                num_cached_blocks=num_cached_blocks,
                num_full_blocks=num_full_blocks,
                block_size=self.block_size,
                kv_cache_group_id=self.kv_cache_group_id,
                block_mask=block_mask,
            )
            self.num_cached_block[request.request_id] = num_full_blocks

        SingleTypeKVCacheManager.cache_blocks = cache_blocks  # type: ignore[method-assign]

        def mamba_cache_blocks(
            self: MambaManager,
            request: Request,
            num_tokens: int,
            retention_interval: int | None = None,
            alignment_tokens: int | None = None,
            use_eagle: bool = False,
        ) -> None:
            num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)
            super(MambaManager, self).cache_blocks(
                request,
                num_tokens,
                retention_interval=retention_interval,
                alignment_tokens=alignment_tokens,
                use_eagle=use_eagle,
            )
            num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)
            if num_cached_blocks_after > num_cached_blocks_before:
                for block in self.req_to_blocks[request.request_id][num_cached_blocks_before:num_cached_blocks_after]:
                    if block.is_null:
                        continue
                    assert block.block_hash is not None
                    self.cached_blocks_this_step.add(block.block_hash)

        MambaManager.cache_blocks = mamba_cache_blocks  # type: ignore[method-assign]

        def cross_attention_cache_blocks(
            self: CrossAttentionManager,
            request: Request,
            num_tokens: int,
            retention_interval: int | None = None,
            alignment_tokens: int | None = None,
            use_eagle: bool = False,
        ) -> None:
            raise ValueError("Should not be called as prefix caching is disabled.")

        CrossAttentionManager.cache_blocks = cross_attention_cache_blocks  # type: ignore[method-assign]

    SlidingWindowManager.reachable_block_mask = classmethod(_sliding_window_reachable_block_mask)  # type: ignore[attr-defined]


def _patch_sliding_window_find_longest_cache_hit() -> None:
    def find_longest_cache_hit(
        cls: type[SlidingWindowManager],
        block_hashes: Any,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert _is_sliding_window_spec(kv_cache_spec)
        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."

        need = _contiguous_blocks_for_hit(
            window_size=kv_cache_spec.sliding_window,
            block_size=kv_cache_spec.block_size,
            use_eagle=use_eagle,
        )
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple([block_pool.null_block] * max_num_blocks for _ in range(len(kv_cache_group_ids)))
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        for i in range(max_num_blocks - 1, -1, -1):
            cached_block = block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids)
            if cached_block:
                if num_contiguous_blocks == 0 and block_size != alignment_tokens:
                    post_pop_blocks = i if use_eagle else i + 1
                    if post_pop_blocks * block_size % alignment_tokens != 0:
                        continue
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= need:
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while block_size != alignment_tokens and len(computed_blocks[0]) * block_size % alignment_tokens != 0:
                for computed in computed_blocks:
                    computed.pop()
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
            while block_size != alignment_tokens and len(computed_blocks[0]) * block_size % alignment_tokens != 0:
                for computed in computed_blocks:
                    computed.pop()
        return computed_blocks

    SlidingWindowManager.find_longest_cache_hit = classmethod(find_longest_cache_hit)  # type: ignore[method-assign]


def _patch_sliding_window_free() -> None:
    def remove_skipped_blocks(
        self: SingleTypeKVCacheManager,
        request_id: str,
        total_computed_tokens: int,
    ) -> None:
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            return

        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = min(num_skipped_tokens // self.block_size, len(blocks))
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

    def free(self: SingleTypeKVCacheManager, request_id: str) -> None:
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

    SlidingWindowManager.remove_skipped_blocks = remove_skipped_blocks  # type: ignore[method-assign]
    SlidingWindowManager.free = free  # type: ignore[method-assign]


def _manager_uses_eagle(coordinator: KVCacheCoordinator, manager: SingleTypeKVCacheManager) -> bool:
    eagle_group_ids = getattr(coordinator, "eagle_group_ids", None)
    if eagle_group_ids is not None:
        return manager.kv_cache_group_id in eagle_group_ids
    return getattr(coordinator, "use_eagle", False)


def _coordinator_alignment_tokens(coordinator: KVCacheCoordinator) -> int:
    kv_cache_config = getattr(coordinator, "kv_cache_config", None)
    kv_cache_groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if kv_cache_groups:
        block_sizes = [
            spec.block_size * max(getattr(spec, "compress_ratio", 1) or 1, 1)
            for group in kv_cache_groups
            for spec in _unwrap_specs(group.kv_cache_spec)
        ]
        if block_sizes:
            return lcm(*block_sizes)
    return (
        getattr(coordinator, "scheduler_block_size", None)
        or getattr(coordinator, "lcm_block_size", None)
        or getattr(coordinator, "block_size", None)
        or coordinator.single_type_managers[0].block_size
    )


def _cache_blocks_with_retention(
    coordinator: KVCacheCoordinator,
    request: Request,
    num_computed_tokens: int,
) -> None:
    alignment_tokens = _coordinator_alignment_tokens(coordinator)
    for manager in coordinator.single_type_managers:
        manager.cache_blocks(
            request,
            num_computed_tokens,
            retention_interval=getattr(coordinator, "retention_interval", None),
            alignment_tokens=alignment_tokens,
            use_eagle=_manager_uses_eagle(coordinator, manager),
        )


def _patch_upstream_coordinators() -> None:
    orig_init = KVCacheCoordinator.__init__
    orig_hybrid_init = HybridKVCacheCoordinator.__init__

    if not getattr(KVCacheCoordinator, "_ascend_retention_patched", False):

        def init(
            self: KVCacheCoordinator,
            kv_cache_config: KVCacheConfig,
            max_model_len: int,
            use_eagle: bool,
            enable_caching: bool,
            enable_kv_cache_events: bool,
            dcp_world_size: int,
            pcp_world_size: int,
            hash_block_size: int,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            orig_init(
                self,
                kv_cache_config,
                max_model_len,
                use_eagle,
                enable_caching,
                enable_kv_cache_events,
                dcp_world_size,
                pcp_world_size,
                hash_block_size,
                *args,
                **kwargs,
            )
            alignment_tokens = _coordinator_alignment_tokens(self)
            self.retention_interval = get_prefix_cache_retention_interval(kv_cache_config, alignment_tokens)

        def hybrid_init(
            self: HybridKVCacheCoordinator,
            kv_cache_config: KVCacheConfig,
            max_model_len: int,
            use_eagle: bool,
            enable_caching: bool,
            enable_kv_cache_events: bool,
            dcp_world_size: int,
            pcp_world_size: int,
            hash_block_size: int,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            orig_hybrid_init(
                self,
                kv_cache_config,
                max_model_len,
                use_eagle,
                enable_caching,
                enable_kv_cache_events,
                dcp_world_size,
                pcp_world_size,
                hash_block_size,
                *args,
                **kwargs,
            )
            alignment_tokens = _coordinator_alignment_tokens(self)
            self.retention_interval = get_prefix_cache_retention_interval(kv_cache_config, alignment_tokens)

        KVCacheCoordinator.__init__ = init  # type: ignore[method-assign]
        HybridKVCacheCoordinator.__init__ = hybrid_init  # type: ignore[method-assign]
        KVCacheCoordinator.cache_blocks = _cache_blocks_with_retention  # type: ignore[method-assign]
        HybridKVCacheCoordinator.cache_blocks = _cache_blocks_with_retention  # type: ignore[method-assign]
        KVCacheCoordinator._ascend_retention_patched = True  # type: ignore[attr-defined]


_add_prepend_n_to_free_queue()
_patch_block_pool_free_blocks()
_patch_block_pool_cache_full_blocks()
_patch_compress_manager_factory()
_patch_single_type_cache_blocks()
_patch_sliding_window_find_longest_cache_hit()
_patch_sliding_window_free()
_patch_upstream_coordinators()
