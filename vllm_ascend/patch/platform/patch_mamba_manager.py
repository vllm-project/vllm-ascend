# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
import vllm.v1.core.single_type_kv_cache_manager as single_type_kv_cache_manager
from vllm.v1.core.single_type_kv_cache_manager import (
    BlockHashList,
    BlockPool,
    KVCacheBlock,
    KVCacheSpec,
    MambaManager,
    MambaSpec,
)


class AscendMambaManager(MambaManager):
    def __init__(self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        if self.enable_caching:
            self.block_size = kv_cache_spec.block_size

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, MambaSpec), "MambaManager can only be used for mamba groups"
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple([] for _ in range(len(kv_cache_group_ids)))
        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids):
                if block_size != alignment_tokens and (i + 1) * block_size % alignment_tokens != 0:
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break
        return computed_blocks


single_type_kv_cache_manager.MambaManager = AscendMambaManager
single_type_kv_cache_manager.spec_manager_map[MambaSpec] = AscendMambaManager
