# SPDX-License-Identifier: Apache-2.0
"""Validate supported hash-recovery combinations at initialization."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import metadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    pool_uses_eagle,
    resolve_pool_block_sizes,
    resolve_pool_cache_groups,
    resolve_pool_hit_alignment,
    validate_pool_cache_config,
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
    sizes = resolve_pool_block_sizes(config, cache)
    assert sizes == [16]
    assert resolve_pool_hit_alignment(cache, sizes[0]) == 4
    assert not pool_uses_eagle(config, cache)
    validate_pool_cache_config(config, sizes, [0], cache.hash_block_size, True)


def test_unresolved_alignment_keeps_whole_page_hits():
    config, cache = geometry_config(alignment=None)
    with (
        patch.object(metadata.kv_cache_utils, "resolve_kv_cache_block_sizes", side_effect=AssertionError),
        patch.object(metadata.kv_cache_utils, "resolve_cache_hit_alignment_tokens", side_effect=AssertionError),
    ):
        alignment = resolve_pool_hit_alignment(cache, 16)
    assert cache.hash_block_size == 4
    assert alignment == 16
    assert cache.cache_hit_alignment_tokens is None


@pytest.mark.parametrize("dcp", [1, 2, 4])
@pytest.mark.parametrize("producer, consumer", [(True, False), (False, True), (True, True)])
def test_dcp_specs_and_eagle_role_policy(dcp, producer, consumer):
    config, cache = geometry_config()
    config.parallel_config.decode_context_parallel_size = dcp
    config.kv_transfer_config.is_kv_producer = producer
    config.kv_transfer_config.is_kv_consumer = consumer
    config.speculative_config = SimpleNamespace(use_eagle_block_drop=lambda: True)
    cache.kv_cache_groups[0].is_eagle_group = True
    sizes = resolve_pool_block_sizes(config, cache)
    groups = resolve_pool_cache_groups(config, cache)
    assert sizes == [16 * dcp]
    assert groups[0].kv_cache_spec.block_size == sizes[0]
    assert groups[0].is_eagle_group is (not (producer and not consumer))
    assert pool_uses_eagle(config, cache) is (not (producer and not consumer))
    assert cache.kv_cache_groups[0].kv_cache_spec.block_size == 16
    assert cache.kv_cache_groups[0].is_eagle_group


@pytest.mark.parametrize("partial_hits", [False, True])
def test_partial_chunk_policy_only_restricts_partial_hits(partial_hits):
    config, cache = geometry_config()
    config.kv_transfer_config.kv_connector_extra_config["discard_partial_chunks"] = False
    if partial_hits:
        with pytest.raises(ValueError, match="discard_partial_chunks"):
            validate_pool_cache_config(config, [16], [0], 4, partial_hits)
    else:
        validate_pool_cache_config(config, [16], [0], 4, partial_hits)


def test_cache_hash_validation_excludes_private_groups():
    config, cache = geometry_config()
    validate_pool_cache_config(config, [16, 7], [0], 4, True)
    with pytest.raises(ValueError, match="cannot fold Core hash size"):
        validate_pool_cache_config(config, [16, 7], [0, 1], 4, True)


def test_dcp_keeps_unsharded_mamba_checkpoint_size():
    config, cache = geometry_config()
    config.parallel_config.decode_context_parallel_size = 4
    cache.kv_cache_groups.append(
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(block_size=16, shapes=((2,),), dtypes=(torch.float32,), mamba_cache_mode="align"),
        )
    )
    assert resolve_pool_block_sizes(config, cache) == [64, 16]
    groups = resolve_pool_cache_groups(config, cache)
    assert [group.kv_cache_spec.block_size for group in groups] == [64, 16]
