# SPDX-License-Identifier: Apache-2.0
"""No-prefix-cache coordinator backed by typed physical pages."""

from __future__ import annotations

from collections.abc import Sequence

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinatorNoPrefixCache
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import CrossAttentionSpec, KVCacheConfig

from vllm_ascend.core.single_type_kv_cache_manager import (
    get_manager_for_kv_cache_spec,
)
from vllm_ascend.core.typed_kv_cache import (
    TypedAddressAdmissionPool,
    TypedAddressPool,
    TypedAdmissionPool,
    TypedKVCachePlan,
    TypedRegionAdmissionPool,
    TypedRegionPool,
    TypedSuperpagePool,
)


class TypedKVCacheCoordinatorNoPrefixCache(KVCacheCoordinatorNoPrefixCache):
    """Experimental coordinator with atomic heterogeneous-page admission.

    Each single-type manager receives a group-specific pool facade.  The outer
    ``KVCacheManager`` receives an aggregate admission facade, so page demand
    is validated atomically in physical byte space rather than by incorrectly
    summing heterogeneous block counts.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        plan: TypedKVCachePlan,
        max_model_len: int,
        max_in_flight_tokens: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        if dcp_world_size != 1 or pcp_world_size != 1:
            raise ValueError("typed KV cache MVP does not support DCP or PCP")
        if metrics_collector is not None:
            raise ValueError("typed KV cache MVP does not expose block metrics")
        if len(kv_cache_config.kv_cache_groups) != len(plan.specs):
            raise ValueError("the typed plan must describe every KV cache group")
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, CrossAttentionSpec):
                raise ValueError("typed KV cache MVP does not support cross attention")
            typed_spec = plan.spec(group_id)
            if typed_spec.block_size_tokens != group.kv_cache_spec.block_size:
                raise ValueError(f"group {group_id} block size differs between config and plan")
            if typed_spec.page_size_bytes != group.kv_cache_spec.page_size_bytes:
                raise ValueError(f"group {group_id} page size differs between config and plan")
        if any(scheduler_block_size % spec.block_size_tokens for spec in plan.specs):
            raise ValueError("scheduler block size must align every typed group")

        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = False
        self.scheduler_block_size = scheduler_block_size
        self.retention_interval = None
        self.eagle_group_ids: set[int] = set()
        self.use_eagle = False
        if plan.is_addressed:
            self.typed_pool = TypedAddressPool(plan)
            self.block_pool = TypedAddressAdmissionPool(self.typed_pool)
        elif plan.is_partitioned:
            self.typed_pool = TypedRegionPool(plan)
            self.block_pool = TypedRegionAdmissionPool(self.typed_pool)
        else:
            self.typed_pool = TypedSuperpagePool(plan)
            self.block_pool = TypedAdmissionPool(self.typed_pool)
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec,
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                block_pool=self.typed_pool.for_group(group_id),
                enable_caching=False,
                kv_cache_group_id=group_id,
                dcp_world_size=1,
                pcp_world_size=1,
                scheduler_block_size=scheduler_block_size,
                needs_kv_cache_zeroing=kv_cache_config.needs_kv_cache_zeroing,
            )
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        if num_encoder_tokens:
            raise ValueError("typed KV cache MVP does not support encoder tokens")
        if any(new_computed_blocks):
            raise ValueError("prefix-cache blocks are unsupported in typed MVP")

        blocks_by_group: dict[int, int] = {}
        for group_id, manager in enumerate(self.single_type_managers):
            blocks_by_group[group_id] = manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks[group_id],
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        if isinstance(self.typed_pool, (TypedAddressPool, TypedRegionPool)):
            return 0 if self.typed_pool.can_allocate(blocks_by_group) else 1
        return self.typed_pool.additional_superpages_required(blocks_by_group)
