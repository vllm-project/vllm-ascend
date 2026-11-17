# SPDX-License-Identifier: Apache-2.0
"""Validate supported hash-recovery combinations at initialization."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import metadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import resolve_pool_cache_geometry


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
    geometry = resolve_pool_cache_geometry(config, cache, use_layerwise=False)
    assert geometry.partial_hits


@pytest.mark.parametrize("layerwise", [False, True])
def test_unresolved_alignment_keeps_whole_page_hits(layerwise):
    config, cache = geometry_config(alignment=None)
    with (
        patch.object(metadata.kv_cache_utils, "resolve_kv_cache_block_sizes", side_effect=AssertionError),
        patch.object(metadata.kv_cache_utils, "resolve_cache_hit_alignment_tokens", side_effect=AssertionError),
    ):
        geometry = resolve_pool_cache_geometry(config, cache, use_layerwise=layerwise)
    assert geometry.hash_block_size == 4
    assert geometry.core_hit_alignment == 16
    assert not geometry.partial_hits
    assert cache.cache_hit_alignment_tokens is None
