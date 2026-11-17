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
@pytest.mark.parametrize("hash_size", [None, 4])
def test_unresolved_alignment_keeps_whole_page_hits(layerwise, hash_size):
    config, cache = geometry_config(alignment=None)
    cache.hash_block_size = hash_size
    with (
        patch.object(metadata.kv_cache_utils, "resolve_kv_cache_block_sizes", return_value=(16, 4)) as resolve_hash,
        patch.object(metadata.kv_cache_utils, "resolve_cache_hit_alignment_tokens", side_effect=AssertionError),
    ):
        geometry = resolve_pool_cache_geometry(config, cache, use_layerwise=layerwise)
    assert geometry.hash_block_size == 4
    assert geometry.core_hit_alignment == 16
    assert not geometry.partial_hits
    if hash_size is None:
        resolve_hash.assert_called_once_with(cache, config)
    else:
        resolve_hash.assert_not_called()
    assert cache.cache_hit_alignment_tokens is None


def test_partial_hits_require_core_resolved_hash():
    config, cache = geometry_config()
    cache.hash_block_size = None
    with (
        patch.object(metadata.kv_cache_utils, "resolve_kv_cache_block_sizes") as resolve_hash,
        pytest.raises(ValueError, match="Core-resolved hash_block_size"),
    ):
        resolve_pool_cache_geometry(config, cache, use_layerwise=False)
    resolve_hash.assert_not_called()


def test_layerwise_can_resolve_missing_hash_without_enabling_partial_hits():
    config, cache = geometry_config()
    cache.hash_block_size = None
    with patch.object(metadata.kv_cache_utils, "resolve_kv_cache_block_sizes", return_value=(16, 4)):
        geometry = resolve_pool_cache_geometry(config, cache, use_layerwise=True)
    assert geometry.hash_block_size == 4
    assert not geometry.partial_hits


@pytest.mark.parametrize("prefix_unit", [None, 4])
def test_missing_cache_config_retains_legacy_contract(prefix_unit):
    config, _ = geometry_config()
    config.cache_config.prefix_match_unit = prefix_unit
    if prefix_unit is not None:
        with pytest.raises(ValueError, match="resolved KVCacheConfig"):
            resolve_pool_cache_geometry(config, None, use_layerwise=False)
    else:
        geometry = resolve_pool_cache_geometry(config, None, use_layerwise=False)
        assert geometry.hash_block_size == geometry.core_hit_alignment == 16
        assert not geometry.partial_hits
