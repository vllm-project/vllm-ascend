# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix lookup results never contain circular scratch pages."""

from vllm.v1.core.kv_cache_manager import KVCacheManager

from vllm_ascend.core.circular_buffer import prefix_cacheable

_original_truncate = KVCacheManager.truncate_computed_blocks


def _truncate_computed_blocks(self, blocks, num_computed_tokens):
    groups = self.kv_cache_config.kv_cache_groups
    if all(prefix_cacheable(g.kv_cache_spec) for g in groups):
        return _original_truncate(self, blocks, num_computed_tokens)
    truncated = []
    for group_blocks, manager, group in zip(blocks.blocks, self.coordinator.single_type_managers, groups, strict=True):
        if not prefix_cacheable(group.kv_cache_spec):
            assert not group_blocks, "Scratch pages cannot be prefix-cache hits"
            truncated.append([])
            continue
        assert num_computed_tokens % manager.block_size == 0
        count = num_computed_tokens // manager.block_size
        assert count <= len(group_blocks)
        truncated.append(list(group_blocks[:count]))
    return self.create_kv_cache_blocks(tuple(truncated))


KVCacheManager.truncate_computed_blocks = _truncate_computed_blocks
