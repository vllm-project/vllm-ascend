# SPDX-License-Identifier: Apache-2.0
"""Private physical rings must not be treated as token-page alignment units."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

import vllm_ascend.patch.platform.patch_kv_cache_coordinator as coordinator
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager


def config_with_ring(capacity):
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=1152, prefix_cacheable=True)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4096, prefix_cacheable=True)),
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=capacity, prefix_cacheable=False, is_circular=True)
            ),
        ]
    )


@pytest.mark.parametrize("capacity", [4, 7, 12])
@pytest.mark.parametrize("enable_caching", [False, True])
def test_private_ring_routes_to_ascend_without_inflating_scheduler(monkeypatch, capacity, enable_caching):
    ascend, upstream = Mock(), Mock(side_effect=AssertionError("Private ring reached ordinary coordinator"))
    monkeypatch.setattr(coordinator, "AscendHybridKVCacheCoordinator", ascend)
    monkeypatch.setattr(coordinator, "_orig_get_kv_cache_coordinator", upstream)
    result = coordinator.get_kv_cache_coordinator(
        config_with_ring(capacity),
        4096,
        enable_caching=enable_caching,
        scheduler_block_size=36864,
        hash_block_size=128 if enable_caching else 36864,
    )
    assert result is ascend.return_value
    assert ascend.call_args.kwargs["scheduler_block_size"] == 36864
    upstream.assert_not_called()


@pytest.mark.parametrize("scheduler,hash_size", [(4096, 4096), (36864, 7), (36864, 0)])
def test_private_ring_does_not_disable_real_alignment_validation(monkeypatch, scheduler, hash_size):
    ascend = Mock()
    monkeypatch.setattr(coordinator, "AscendHybridKVCacheCoordinator", ascend)
    with pytest.raises(AssertionError):
        coordinator.get_kv_cache_coordinator(
            config_with_ring(7), 4096, scheduler_block_size=scheduler, hash_block_size=hash_size, enable_caching=False
        )
    ascend.assert_not_called()


def test_ordinary_no_prefix_cache_still_uses_upstream(monkeypatch):
    upstream = Mock()
    monkeypatch.setattr(coordinator, "_orig_get_kv_cache_coordinator", upstream)
    config = config_with_ring(7)
    config.kv_cache_groups.pop()
    assert (
        coordinator.get_kv_cache_coordinator(
            config, 4096, enable_caching=False, scheduler_block_size=36864, hash_block_size=36864
        )
        is upstream.return_value
    )


@pytest.mark.parametrize("enable_caching", [False, True])
def test_real_coordinator_initializes_seven_token_private_ring(enable_caching):
    register_ascend_kv_cache_specs()
    attention = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    tail = AscendIndexerKPoolTailSpec(
        block_size=7, num_kv_heads=1, head_size=128, dtype=torch.float32, sliding_window=4, compress_ratio=4
    )
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attention"], kv_cache_spec=attention),
            KVCacheGroupSpec(layer_names=["tail"], kv_cache_spec=tail),
        ],
    )
    actual = coordinator.get_kv_cache_coordinator(
        config,
        4096,
        max_in_flight_tokens=512,
        scheduler_block_size=128,
        hash_block_size=128,
        enable_caching=enable_caching,
        use_eagle=True,
        num_prefill_lookahead=1,
    )
    assert isinstance(actual, coordinator.AscendHybridKVCacheCoordinator)
    assert actual.scheduler_block_size == 128
    assert isinstance(actual.single_type_managers[1], KpoolTailManager)
