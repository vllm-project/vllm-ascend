# SPDX-License-Identifier: Apache-2.0
"""Bulk-pooling contracts for Step5's main, summary, and request-private ring."""

from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split

import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_physical_layer_index,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    KeyMetadata,
    PoolKey,
    infer_cacheable_group_ids,
    infer_group_block_sizes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


@pytest.fixture
def groups():
    return [
        KVCacheGroupSpec(
            ["model.layers.3.self_attn.attn"],
            FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=192, dtype=torch.bfloat16),
        ),
        KVCacheGroupSpec(
            ["model.layers.3.self_attn.csa_tail_ring"],
            CircularBufferSpec(block_size=128, num_kv_heads=1, head_size=128, head_size_v=0, dtype=torch.bfloat16),
        ),
        KVCacheGroupSpec(
            ["model.layers.3.self_attn.sparse_summary_cache"],
            FullAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=256,
                head_size_v=0,
                dtype=torch.uint8,
                tokens_per_state=8,
            ),
        ),
    ]


def test_ring_does_not_participate_in_external_prefixes(groups):
    assert infer_cacheable_group_ids(groups) == [0, 2]
    assert infer_group_block_sizes(128, groups) == [128, 128, 128]
    assert [group.kv_cache_spec.page_size_bytes for group in groups] == [98304, 32768, 4096]


def test_wrapped_auxiliary_groups_preserve_ring_exclusion(groups):
    wrapped = [
        KVCacheGroupSpec(
            group.layer_names,
            UniformTypeKVCacheSpecs(
                block_size=128, kv_cache_specs={name: group.kv_cache_spec for name in group.layer_names}
            ),
        )
        for group in groups
    ]
    assert infer_cacheable_group_ids(wrapped) == [0, 2]
    assert infer_group_block_sizes(128, wrapped) == [128, 128, 128]


def test_layerwise_groups_join_summary_and_main_without_advancing_for_ring(groups):
    groups[0].layer_names.append("model.layers.92.self_attn.attn")
    worker = object.__new__(LayerwiseKVPoolWorker)
    worker.num_layers = 2
    worker.num_kv_cache_groups = len(groups)
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    worker.hf_config = SimpleNamespace(num_hidden_layers=92)
    worker.cacheable_group_ids = infer_cacheable_group_ids(groups)
    worker.use_layerwise_transfer = True
    worker.use_block_key_layerwise = False
    worker.use_layerwise = True
    worker._extra_config = {}
    worker._extract_physical_layer_index = lambda name: get_physical_layer_index(name, 92)
    worker._init_layerwise_config()
    assert worker._global_to_local_layer == {3: 0, 92: 1}
    assert worker._groups_for_layerwise_transfer(0) == [(0, 0), (2, 0)]
    assert worker._groups_for_layerwise_transfer(1) == [(0, 1)]
    assert not worker.layerwise_offload


def test_summary_transfer_uses_physical_bytes_without_changing_token_alignment():
    worker = SimpleNamespace(num_blocks=4)
    summary = torch.zeros(4, 16, 256, dtype=torch.uint8)
    block_bytes, stride, region_bytes, scale = KVPoolWorker._get_cache_block_metadata(worker, summary)
    assert (block_bytes, stride, region_bytes, scale) == (4096, 4096, 16384, 1)


def test_main_and_summary_keys_preserve_all_four_gqa_shards():
    # TP8 replicates four main KV heads twice. The replicated summary can
    # safely use the same four keys; it must not collapse the main KV to one.
    keys = {
        PoolKey(KeyMetadata("step", rank // 2, 0, 0, group), "abc").to_string() for rank in range(8) for group in (0, 2)
    }
    assert len(keys) == 8
    for group in (0, 2):
        assert sum(f"@group:{group}@" in key for key in keys) == 4


@pytest.mark.parametrize("summary_blocks,expected_tokens", [(0, 0), (1, 128), (2, 256)])
def test_main_and_summary_must_both_cover_a_prefix(groups, summary_blocks, expected_tokens):
    coordinator = AscendStoreCoordinator(
        groups,
        scheduler_block_size=128,
        hash_block_size=128,
        group_block_sizes=[128, 128, 128],
        group_cache_families=["default"] * 3,
    )
    hashes = [bytes([index + 1]) * 32 for index in range(2)]
    present = {(0, block_hash) for block_hash in hashes}
    present.update((2, block_hash) for block_hash in hashes[:summary_blocks])
    # The request-private ring has no external key, even when the prefix hits.
    _, hit_tokens = coordinator.find_longest_cache_hit(hashes, 256, ExternalCachedBlockPool(128, present))
    assert hit_tokens == expected_tokens
