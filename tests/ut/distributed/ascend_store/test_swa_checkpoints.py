# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GVA replay boundary regressions using upstream SWA lookup (CPU only)."""

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, KVCacheGroupSpec, SlidingWindowSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler

pytestmark = pytest.mark.skipif(
    "reachable_boundaries" not in inspect.signature(SlidingWindowManager.reachable_block_mask).parameters,
    reason="requires upstream periodic SWA retention, not the simplified dependency stub",
)

BLOCK_SIZE = 128
CACHED_PROMPT_TOKENS = 16384
RETENTION_INTERVAL = 2048


@pytest.fixture(autouse=True)
def upstream_managers():
    # Bind the real algorithms without initializing a platform/worker just
    # to populate the global spec registry. Pool membership is simulated.
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
        side_effect=lambda spec: SlidingWindowManager if isinstance(spec, SlidingWindowSpec) else FullAttentionManager,
    ):
        yield


def make_coordinator():
    common = dict(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=16, dtype=torch.float32)
    specs = [
        FullAttentionSpec(**common),  # CSA main KV.
        FullAttentionSpec(**{**common, "head_size": 8}),  # Independent CSA summaries.
        SlidingWindowSpec(**common, sliding_window=512),
        CircularBufferSpec(**common),  # Private tail ring must never participate.
    ]
    return AscendStoreCoordinator(
        [KVCacheGroupSpec([f"layer.{i}"], spec) for i, spec in enumerate(specs)],
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        group_block_sizes=[BLOCK_SIZE] * len(specs),
        group_cache_families=["default"] * len(specs),
        retention_interval=RETENTION_INTERVAL,
    )


@pytest.mark.parametrize(
    "prompt_tokens,expected",
    [
        pytest.param(2048, 0, id="missing-window-at-aligned-boundary"),
        pytest.param(4096, 2048, id="fallback-to-earlier-checkpoint"),
        pytest.param(2049, 2048, id="unaligned-prompt-keeps-last-full-page"),
        pytest.param(128, None, id="no-full-page-before-last-token"),
    ],
)
def test_private_state_lookup_checks_the_window_at_the_actual_resume_boundary(prompt_tokens, expected):
    coord = make_coordinator()
    # A longer request populated periodic checkpoints. Its window at 2048
    # is complete, but it did not retain the older block needed at 1920.
    hashes = [index.to_bytes(32, "big") for index in range(CACHED_PROMPT_TOKENS // BLOCK_SIZE)]
    masks = coord.store_mask(CACHED_PROMPT_TOKENS, num_prompt_tokens=CACHED_PROMPT_TOKENS)
    pool = {
        (group_id, block_hash)
        for group_id in coord.cacheable_group_ids
        for block_hash, retained in zip(hashes, masks[group_id])
        if retained
    }
    scheduler = object.__new__(LayerwiseKVPoolScheduler)
    scheduler.use_block_key_layerwise = False
    scheduler.use_eagle = False
    scheduler.use_layerwise_transfer = True
    scheduler.cacheable_group_ids = coord.cacheable_group_ids
    scheduler.original_block_size = [BLOCK_SIZE] * 4
    scheduler.cache_transfer_granularity = BLOCK_SIZE
    request = SimpleNamespace(prompt_token_ids=[0] * prompt_tokens, block_hashes=hashes[: prompt_tokens // BLOCK_SIZE])

    def query(group_id, group_hashes, mask):
        return [
            block_hash
            for index, block_hash in enumerate(group_hashes)
            if (mask is None or mask[index]) and (group_id, block_hash) in pool
        ]

    scheduler._get_layerwise_hit_tokens = lambda request, length, computed: coord.find_reachable_hit_tokens(
        request.block_hashes, length, query
    )
    hit = scheduler._lookup_external_hit_tokens(request, 0)
    assert hit == expected
    if hit:
        assert hit < prompt_tokens
        for group_id, mask in enumerate(coord.load_mask(request.block_hashes, hit)):
            assert all(
                not required or (group_id, request.block_hashes[index]) in pool for index, required in enumerate(mask)
            )
