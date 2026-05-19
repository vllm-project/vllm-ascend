# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
from math import lcm

import vllm.v1.core.kv_cache_coordinator as kv_cache_coordinator
from vllm.v1.core.kv_cache_coordinator import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    FullAttentionSpec,
    HybridKVCacheCoordinator,
    KVCacheBlock,
    KVCacheConfig,
    KVCacheCoordinator,
    KVCacheMetricsCollector,
    KVCacheSpec,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import MambaSpec


class AscendHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    """CP-aware hybrid KV cache coordinator for Ascend.
    Upstream `HybridKVCacheCoordinator` assumes the logical block size used by
    prefix caching is always the raw `kv_cache_spec.block_size`. Under PCP/DCP,
    however, vllm-ascend hashes and allocates KV blocks using the virtual block
    size, i.e. `block_size * dcp_world_size * pcp_world_size`. This subclass
    keeps the upstream fixed-point lookup algorithm but makes all block-size
    calculations CP-aware.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        self.hash_block_size = hash_block_size
        for g in kv_cache_config.kv_cache_groups:
            if isinstance(g.kv_cache_spec, MambaSpec):
                self.hash_block_size = self._get_effective_block_size(g.kv_cache_spec)
                break

        KVCacheCoordinator.__init__(
            self,
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=self.hash_block_size,
            metrics_collector=metrics_collector,
        )
        assert all(
            self._get_effective_block_size(g.kv_cache_spec) % self.hash_block_size == 0
            for g in kv_cache_config.kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        self.verify_and_split_kv_cache_groups()

    def _get_effective_block_size(self, kv_cache_spec: KVCacheSpec) -> int:
        block_size = kv_cache_spec.block_size
        if self.dcp_world_size * self.pcp_world_size > 1:
            block_size *= self.dcp_world_size * self.pcp_world_size
        if isinstance(kv_cache_spec, MambaSpec):
            block_size = block_size // self.dcp_world_size
        return block_size

    def verify_and_split_kv_cache_groups(self) -> None:
        attention_groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, "Expected same manager class for identical KV cache specs."
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))
        assert len(attention_groups) > 1, "HybridKVCacheCoordinator requires at least two attention groups."
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )
        block_sizes = [self._get_effective_block_size(spec) for spec, _, _ in attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

        # Attention-group indices (into ``self.attention_groups``) that
        # contain at least one EAGLE/MTP KV cache group.
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(gid in self.eagle_group_ids for gid in group_ids)
        }

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            effective_block_size = self._get_effective_block_size(kv_cache_spec)
            if effective_block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(block_hashes, self.hash_block_size, effective_block_size)

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )
        eagle_verified: set[int] = set()
        while True:
            curr_hit_length = hit_length
            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                effective_block_size = self._get_effective_block_size(spec)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    num_blocks = curr_hit_length // effective_block_size
                    curr_hit_length = num_blocks * effective_block_size
                    continue
                use_eagle = idx in self.eagle_attn_group_indices and idx not in eagle_verified
                _max_length = curr_hit_length
                if use_eagle:
                    # Eagle needs to match one more block and then pop the last.
                    _max_length = min(curr_hit_length + effective_block_size, max_cache_hit_length)
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=curr_hit_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    use_eagle=use_eagle,
                    alignment_tokens=self.lcm_block_size,
                    dcp_world_size=self.dcp_world_size,
                    pcp_world_size=self.pcp_world_size,
                )
                _new_hit_length = len(hit_blocks[0]) * effective_block_size
                if use_eagle:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // self._get_effective_block_size(spec)
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
        return tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group), hit_length


kv_cache_coordinator.HybridKVCacheCoordinator = AscendHybridKVCacheCoordinator
