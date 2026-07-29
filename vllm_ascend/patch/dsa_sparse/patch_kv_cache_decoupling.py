# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""vLLM v0.23 compatible split block pools for DSA sparse offload.

The dense Indexer cache and the sparse MLA resident cache are different
physical tensors.  Their block ids therefore live in independent namespaces
and cannot safely be allocated from vLLM's default shared ``BlockPool``.

This module deliberately patches only two public scheduling boundaries:

* coordinator construction, to give each KV group its own ``BlockPool``;
* ``KVCacheManager.allocate_slots``, to perform capacity checks per group.

All per-spec managers otherwise remain the v0.23 implementations.  In
particular, their v0.23 MTP/admission-cap arguments and allocation semantics
are not replaced by the older DSA fork.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import vllm.v1.core.kv_cache_coordinator as coordinator_mod
import vllm.v1.core.kv_cache_manager as manager_mod
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.core.single_type_kv_cache_manager import (
    get_manager_for_kv_cache_spec,
)
from vllm_ascend.dsa_sparse.dsa_spec_utils import is_dsa_indexer_spec
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
)


class MultiBlockPool:
    """Aggregate view over one physical ``BlockPool`` per KV cache group.

    Managers receive the child pools directly.  The aggregate is exposed only
    through ``KVCacheManager.block_pool`` for usage, event and admission
    accounting.  Block ids intentionally overlap between children because the
    corresponding KV tensors are separate.
    """

    def __init__(
        self,
        num_gpu_blocks: Sequence[int],
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        if not num_gpu_blocks or any(int(value) <= 1
                                     for value in num_gpu_blocks):
            raise ValueError(
                "Every DSA KV group needs at least two blocks "
                f"(one null block plus capacity), got {tuple(num_gpu_blocks)}")
        self.block_pools = [
            BlockPool(
                num_gpu_blocks=int(num_blocks),
                enable_caching=enable_caching,
                hash_block_size=hash_block_size,
                enable_kv_cache_events=enable_kv_cache_events,
                metrics_collector=metrics_collector,
            )
            for num_blocks in num_gpu_blocks
        ]
        self.num_gpu_blocks = sum(
            pool.num_gpu_blocks for pool in self.block_pools)
        self.enable_caching = bool(enable_caching)
        self.enable_kv_cache_events = bool(enable_kv_cache_events)
        self.metrics_collector = metrics_collector

    @property
    def null_block(self) -> KVCacheBlock:
        # No manager consumes this aggregate sentinel; keep the conventional
        # attribute for diagnostics and generic KVCacheManager helpers.
        return self.block_pools[0].null_block

    @property
    def blocks(self) -> list[KVCacheBlock]:
        return [
            block for pool in self.block_pools for block in pool.blocks
        ]

    def get_num_free_blocks(self) -> int:
        return sum(
            pool.get_num_free_blocks() for pool in self.block_pools)

    def get_num_free_blocks_by_group(self) -> tuple[int, ...]:
        return tuple(
            pool.get_num_free_blocks() for pool in self.block_pools)

    def can_allocate(
        self,
        num_blocks_by_group: Sequence[int],
        *,
        reserved_blocks: int = 0,
    ) -> bool:
        if len(num_blocks_by_group) != len(self.block_pools):
            raise RuntimeError(
                "DSA allocation group count differs from block-pool count: "
                f"requested={len(num_blocks_by_group)}, "
                f"pools={len(self.block_pools)}")
        free_by_group = self.get_num_free_blocks_by_group()
        if any(int(needed) > int(free)
               for needed, free in zip(num_blocks_by_group, free_by_group)):
            return False
        return (
            sum(int(value) for value in num_blocks_by_group)
            <= sum(free_by_group) - max(0, int(reserved_blocks)))

    def get_usage(self) -> float:
        usable = sum(max(0, pool.num_gpu_blocks - 1)
                     for pool in self.block_pools)
        if usable == 0:
            return 0.0
        free = sum(pool.get_num_free_blocks() for pool in self.block_pools)
        return 1.0 - free / usable

    def evict_blocks(self, block_ids: set[int]) -> None:
        # DSA forbids external KV connectors/prefix caching.  Keep this method
        # well-defined for generic reset/diagnostic calls by applying the local
        # id set independently to every physical group.
        for pool in self.block_pools:
            pool.evict_blocks({
                block_id for block_id in block_ids
                if 0 <= block_id < pool.num_gpu_blocks
            })

    def reset_prefix_cache(self) -> bool:
        # Do not use all(generator): a False result from one physical group
        # must not prevent the remaining independent pools from being reset.
        results = [
            pool.reset_prefix_cache() for pool in self.block_pools
        ]
        return all(results)

    def take_events(self) -> list[Any]:
        events: list[Any] = []
        for pool in self.block_pools:
            events.extend(pool.take_events())
        return events


def _get_group_num_blocks(group: Any, default_num_blocks: int) -> int:
    return int(getattr(group, "dsa_num_blocks", default_num_blocks))


def _use_group_block_pools(kv_cache_config: KVCacheConfig) -> bool:
    return (
        len(kv_cache_config.kv_cache_groups) > 1
        and any(
            is_dsa_indexer_spec(group.kv_cache_spec)
            for group in kv_cache_config.kv_cache_groups
        )
    )


class DSASplitKVCacheCoordinator(AscendHybridKVCacheCoordinator):
    """The v0.23 Ascend hybrid coordinator with per-group physical pools."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        eagle_attn_layer_names: list[str] | None = None,
        metrics_collector: KVCacheMetricsCollector | None = None,
        max_num_batched_tokens: int | None = None,
        scheduler_block_size: int | None = None,
    ) -> None:
        if enable_caching:
            raise ValueError(
                "DSA split sparse offload does not support prefix caching")
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            eagle_attn_layer_names=eagle_attn_layer_names,
            metrics_collector=metrics_collector,
            max_num_batched_tokens=max_num_batched_tokens,
            scheduler_block_size=scheduler_block_size,
        )

        group_blocks = [
            _get_group_num_blocks(group, kv_cache_config.num_blocks)
            for group in kv_cache_config.kv_cache_groups
        ]
        split_pool = MultiBlockPool(
            group_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )
        self.block_pool = split_pool
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec,
                block_pool=split_pool.block_pools[group_id],
                enable_caching=enable_caching,
                kv_cache_group_id=group_id,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                scheduler_block_size=scheduler_block_size,
            )
            for group_id, group in enumerate(
                self.kv_cache_config.kv_cache_groups)
        )
        # Rebuild manager-class groupings and propagate the native MTP/EAGLE
        # annotations after replacing the temporary managers made by super().
        self.verify_and_split_kv_cache_groups()
        for attention_group_id in self.eagle_attn_group_indices:
            for group_id in self.attention_groups[
                    attention_group_id][1]:
                self.single_type_managers[group_id].use_eagle = True

    def get_num_blocks_to_allocate_by_group(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> list[int]:
        blocks: list[int] = []
        for group_id, manager in enumerate(self.single_type_managers):
            # Decoder-only DSA rejects cross attention during config
            # validation, so every group uses the main-model token domain.
            blocks.append(
                manager.get_num_blocks_to_allocate(
                    request_id=request_id,
                    num_tokens=num_tokens,
                    new_computed_blocks=new_computed_blocks[group_id],
                    total_computed_tokens=total_computed_tokens,
                    num_tokens_main_model=num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                ))
        return blocks

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        return sum(
            self.get_num_blocks_to_allocate_by_group(
                request_id=request_id,
                num_tokens=num_tokens,
                new_computed_blocks=new_computed_blocks,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            ))


_ORIGINAL_GET_KV_CACHE_COORDINATOR = (
    coordinator_mod.get_kv_cache_coordinator)
_ORIGINAL_ALLOCATE_SLOTS = manager_mod.KVCacheManager.allocate_slots


@wraps(_ORIGINAL_GET_KV_CACHE_COORDINATOR)
def _get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_num_batched_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    scheduler_block_size: int | None = None,
    eagle_attn_layer_names: list[str] | None = None,
    metrics_collector: KVCacheMetricsCollector | None = None,
):
    if _use_group_block_pools(kv_cache_config):
        return DSASplitKVCacheCoordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            scheduler_block_size=scheduler_block_size,
            eagle_attn_layer_names=eagle_attn_layer_names,
            metrics_collector=metrics_collector,
        )
    return _ORIGINAL_GET_KV_CACHE_COORDINATOR(
        kv_cache_config=kv_cache_config,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        use_eagle=use_eagle,
        enable_caching=enable_caching,
        enable_kv_cache_events=enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        scheduler_block_size=scheduler_block_size,
        eagle_attn_layer_names=eagle_attn_layer_names,
        metrics_collector=metrics_collector,
    )


def _can_allocate_by_group(
    manager: Any,
    *,
    request_id: str,
    num_tokens: int,
    new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
    num_encoder_tokens: int,
    total_computed_tokens: int,
    num_tokens_main_model: int,
    apply_admission_cap: bool,
    reserved_blocks: int = 0,
) -> bool:
    pool = manager.block_pool
    coordinator = manager.coordinator
    if not isinstance(pool, MultiBlockPool):
        return True
    needed = coordinator.get_num_blocks_to_allocate_by_group(
        request_id=request_id,
        num_tokens=num_tokens,
        new_computed_blocks=new_computed_blocks,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=total_computed_tokens,
        num_tokens_main_model=num_tokens_main_model,
        apply_admission_cap=apply_admission_cap,
    )
    return pool.can_allocate(needed, reserved_blocks=reserved_blocks)


@wraps(_ORIGINAL_ALLOCATE_SLOTS)
def _allocate_slots(
    self,
    request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks=None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    reserved_blocks: int = 0,
):
    if not isinstance(self.block_pool, MultiBlockPool):
        return _ORIGINAL_ALLOCATE_SLOTS(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
        )

    if new_computed_blocks is not None:
        new_computed_block_list = new_computed_blocks.blocks
    else:
        new_computed_block_list = self.empty_kv_cache_blocks.blocks
    num_local_computed_tokens = (
        request.num_computed_tokens + num_new_computed_tokens)
    total_computed_tokens = min(
        num_local_computed_tokens + num_external_computed_tokens,
        self.max_model_len,
    )

    if full_sequence_must_fit:
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        if not _can_allocate_by_group(
            self,
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=total_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
            reserved_blocks=reserved_blocks,
        ):
            return None

    num_tokens_main_model = total_computed_tokens + num_new_tokens
    num_tokens_need_slot = min(
        num_tokens_main_model + num_lookahead_tokens,
        self.max_model_len,
    )
    # Match v0.23 ordering: recycling managers may free skipped blocks before
    # the capacity prediction.  Calling this twice is idempotent and lets the
    # original implementation retain ownership of the actual allocation.
    self.coordinator.remove_skipped_blocks(
        request.request_id, total_computed_tokens)
    if not _can_allocate_by_group(
        self,
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        new_computed_blocks=new_computed_block_list,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=(
            num_local_computed_tokens + num_external_computed_tokens),
        num_tokens_main_model=num_tokens_main_model,
        apply_admission_cap=False,
        reserved_blocks=reserved_blocks,
    ):
        return None

    return _ORIGINAL_ALLOCATE_SLOTS(
        self,
        request,
        num_new_tokens,
        num_new_computed_tokens=num_new_computed_tokens,
        new_computed_blocks=new_computed_blocks,
        num_lookahead_tokens=num_lookahead_tokens,
        num_external_computed_tokens=num_external_computed_tokens,
        delay_cache_blocks=delay_cache_blocks,
        num_encoder_tokens=num_encoder_tokens,
        full_sequence_must_fit=full_sequence_must_fit,
        reserved_blocks=reserved_blocks,
    )


def install_dsa_kv_cache_decoupling_patch() -> None:
    if getattr(coordinator_mod, "_dsa_split_pool_patch_installed", False):
        return
    coordinator_mod.get_kv_cache_coordinator = _get_kv_cache_coordinator
    # kv_cache_manager imports the factory by value.
    manager_mod.get_kv_cache_coordinator = _get_kv_cache_coordinator
    manager_mod.KVCacheManager.allocate_slots = _allocate_slots
    coordinator_mod.MultiBlockPool = MultiBlockPool
    coordinator_mod.DSASplitKVCacheCoordinator = (
        DSASplitKVCacheCoordinator)
    coordinator_mod._dsa_split_pool_patch_installed = True


install_dsa_kv_cache_decoupling_patch()
