# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Old-API regressions: copy unchanged onto the published Plan B for RED.

No imports of new admission APIs: a missing new symbol must not substitute
for demonstrating the original allocation failure. Execution is deferred
until the baseline environment is explicitly handed off.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus


def _manager():
    full = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32)
    sliding = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32, sliding_window=32)
    config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["sliding"], sliding, False),
            KVCacheGroupSpec(["full"], full, False),
        ],
    )
    config.group_num_blocks = (8, 16)
    return KVCacheManager(
        config,
        max_model_len=128,
        scheduler_block_size=16,
        hash_block_size=16,
        max_in_flight_tokens=32,
        enable_caching=True,
    )


@pytest.mark.parametrize("group_id", [0, 1], ids=["sliding-shortfall", "full-shortfall"])
def test_independent_pool_shortfall_returns_backpressure_without_partial_allocation(group_id):
    manager = _manager()
    pools = manager.coordinator.block_pools
    tight_pool = pools[group_id]
    tight_pool.get_new_blocks(tight_pool.get_num_free_blocks() - 1)
    free_before = tuple(pool.get_num_free_blocks() for pool in pools)
    refs_before = tuple(tuple(block.ref_cnt for block in pool.blocks) for pool in pools)
    request = SimpleNamespace(
        request_id="must-wait",
        num_tokens=32,
        num_prompt_tokens=32,
        num_computed_tokens=0,
        num_in_flight_tokens=0,
        status=RequestStatus.WAITING,
    )

    assert manager.allocate_slots(request, 32, delay_cache_blocks=True) is None
    assert tuple(pool.get_num_free_blocks() for pool in pools) == free_before
    assert tuple(tuple(block.ref_cnt for block in pool.blocks) for pool in pools) == refs_before


def test_full_pressure_is_visible_when_old_aggregate_usage_is_below_75_percent():
    manager = _manager()
    full = manager.coordinator.block_pools[1]
    full.get_new_blocks(full.get_num_free_blocks() - 1)

    # Old aggregate: (24 - 8) / 24 = 66.7%; Full: 14 / 15 = 93.3%.
    assert manager.usage == pytest.approx(14 / 15)
