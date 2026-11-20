# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm prefix queries: batching must not weaken any component's hit check."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import (
    LayerwiseKVPoolScheduler,
)


def make_scheduler(batch_limit=0):
    scheduler = object.__new__(LayerwiseKVPoolScheduler)
    scheduler.use_layerwise_transfer = True
    scheduler.grouped_block_size = [16, 32]
    scheduler.layerwise_max_transfer_blocks = batch_limit
    scheduler.store_scheduler = SimpleNamespace(batch_is_readable=Mock())
    scheduler._make_layerwise_hit_check_keys = lambda group, h: [f"{group}:{h}:0", f"{group}:{h}:1"]
    scheduler.cache_coordinator = AscendStoreCoordinator(
        [
            KVCacheGroupSpec(
                [f"layer.{i}"],
                FullAttentionSpec(block_size=size, num_kv_heads=1, head_size=1, dtype=torch.float32),
            )
            for i, size in enumerate(scheduler.grouped_block_size)
        ],
        scheduler_block_size=32,
        hash_block_size=16,
        group_block_sizes=scheduler.grouped_block_size,
        group_cache_families=["c1", "c1"],
    )
    return scheduler


def request():
    return SimpleNamespace(request_id="prefix", block_hashes=[bytes([i]) * 32 for i in range(4)])


@pytest.mark.parametrize("limit,expected_calls", [(0, 1), (2, 3)])
@pytest.mark.parametrize("local", [0, 16, 32, 64])
def test_batch_matches_serial_hit_and_keeps_hbm_boundary(limit, expected_calls, local):
    scheduler = make_scheduler(limit)
    backend = scheduler.store_scheduler.batch_is_readable
    backend.side_effect = lambda keys: [True] * len(keys)
    batched = scheduler._lookup_layerwise_with_coordinator(request(), 64, local)
    batched_keys = [key for call in backend.call_args_list for key in call.args[0]]
    if local == 0:
        assert backend.call_count == expected_calls
    elif local == 64:
        backend.assert_not_called()
    scheduler.use_layerwise_transfer = False
    backend.reset_mock()
    serial = scheduler._lookup_layerwise_with_coordinator(request(), 64, local)
    serial_keys = [key for call in backend.call_args_list for key in call.args[0]]
    assert batched == serial == 64
    assert batched_keys == serial_keys


def test_missing_group_replica_does_not_borrow_another_group_hit():
    scheduler = make_scheduler()
    backend = scheduler.store_scheduler.batch_is_readable
    # Only the last replica of the coarser group's second block is missing.
    missing = f"1:{request().block_hashes[3].hex()}:1"
    backend.side_effect = lambda keys: [key != missing for key in keys]
    assert scheduler._lookup_layerwise_with_coordinator(request(), 64) == 32
    assert backend.call_count == 1


def test_reachability_masks_are_applied_before_batching():
    scheduler = make_scheduler()
    backend = scheduler.store_scheduler.batch_is_readable
    backend.side_effect = lambda keys: [True] * len(keys)
    with patch.object(scheduler.cache_coordinator, "lookup_mask", return_value=([False, False, True, True], [])):
        scheduler._lookup_layerwise_with_coordinator(request(), 64)
    keys = backend.call_args.args[0]
    assert len(keys) == 4
    assert all(key.startswith("0:") for key in keys)
    assert all(request().block_hashes[0].hex() not in key for key in keys)


def test_next_lookup_rechecks_eviction_instead_of_caching_hits():
    scheduler = make_scheduler()
    backend = scheduler.store_scheduler.batch_is_readable
    backend.side_effect = lambda keys: [True] * len(keys)
    assert scheduler._lookup_layerwise_with_coordinator(request(), 64) == 64
    backend.side_effect = lambda keys: [False] * len(keys)
    assert scheduler._lookup_layerwise_with_coordinator(request(), 64) == 0
    assert backend.call_count == 2


@pytest.mark.parametrize("reply", [[], [True], [1] * 12])
def test_malformed_backend_results_fail_closed(reply):
    scheduler = make_scheduler()
    scheduler.store_scheduler.batch_is_readable.return_value = reply
    with pytest.raises(RuntimeError, match="invalid results"):
        scheduler._lookup_layerwise_with_coordinator(request(), 64)


def test_wrong_group_count_is_rejected():
    coordinator = make_scheduler().cache_coordinator
    with pytest.raises(ValueError, match="one hit result per cache group"):
        coordinator.find_reachable_hit_tokens(request().block_hashes, 64, Mock(), batch_query=lambda _: [])
