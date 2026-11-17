# SPDX-License-Identifier: Apache-2.0
"""Validate supported hash-recovery combinations at initialization."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import resolve_pool_cache_geometry


def geometry_config(*, block_drop=False, marked=False, alignment=4):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, prefix_match_unit=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: block_drop),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
    )
    cache = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32),
                is_eagle_group=marked,
            )
        ],
        hash_block_size=4,
        cache_hit_alignment_tokens=alignment,
        prefix_cache_retention_interval=None,
    )
    return config, cache


@pytest.mark.parametrize("block_drop,marked", [(True, False), (False, True), (True, True)])
def test_hash_recovery_rejects_mtp_until_followup(block_drop, marked):
    config, cache = geometry_config(block_drop=block_drop, marked=marked)
    with pytest.raises(ValueError, match="hash recovery with MTP/EAGLE"):
        resolve_pool_cache_geometry(config, cache, use_layerwise=False)


@pytest.mark.parametrize("alignment,layerwise", [(16, False), (4, True)])
def test_existing_mtp_modes_remain_available(alignment, layerwise):
    config, cache = geometry_config(block_drop=True, marked=True, alignment=alignment)
    geometry = resolve_pool_cache_geometry(config, cache, use_layerwise=layerwise)
    assert not geometry.partial_hits
    assert geometry.groups[0].is_eagle_group


def test_hash_recovery_without_mtp_is_available():
    config, cache = geometry_config()
    geometry = resolve_pool_cache_geometry(config, cache, use_layerwise=False)
    assert geometry.partial_hits
