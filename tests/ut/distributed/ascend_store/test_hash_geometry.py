# SPDX-License-Identifier: Apache-2.0
"""Validate supported hash-recovery combinations at initialization."""

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    infer_group_block_sizes,
)


def geometry_config(*, alignment=4):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, prefix_match_unit=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}, is_kv_producer=False, is_kv_consumer=True),
    )
    cache = SimpleNamespace(
        kv_transfer_config=config.kv_transfer_config,
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32),
            )
        ],
        hash_block_size=4,
        cache_hit_alignment_tokens=alignment,
        prefix_cache_retention_interval=None,
    )
    return config, cache


def test_hash_recovery_without_mtp_is_available():
    config, cache = geometry_config()
    sizes = [
        size * config.parallel_config.decode_context_parallel_size
        for size in infer_group_block_sizes(config.cache_config.block_size, cache.kv_cache_groups)
    ]
    assert sizes == [16]
    assert cache.cache_hit_alignment_tokens == 4
