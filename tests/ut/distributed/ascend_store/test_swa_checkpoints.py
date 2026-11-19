# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint save/query contracts using upstream SWA algorithms (CPU only)."""

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
PROMPT_TOKENS = 16384
COMMON_BLOCKS = 115  # 14720 complete shared tokens, approximately 90% of 16K.


@pytest.fixture(autouse=True)
def upstream_managers():
    # Bind the real algorithms without initializing a platform/worker just
    # to populate the global spec registry. Pool membership is simulated.
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
        side_effect=lambda spec: SlidingWindowManager if isinstance(spec, SlidingWindowSpec) else FullAttentionManager,
    ):
        yield


def make_coordinator(interval, *, use_eagle=False):
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
        retention_interval=interval,
        use_eagle=use_eagle,
    )


def hashes(count=PROMPT_TOKENS // BLOCK_SIZE):
    return [index.to_bytes(32, "big") for index in range(count)]


def changed_suffix():
    original = hashes()
    return original[:COMMON_BLOCKS] + [bytes([255]) + item[1:] for item in original[COMMON_BLOCKS:]]


def save(coord, end=PROMPT_TOKENS, prompt=PROMPT_TOKENS, start=0):
    masks = coord.store_mask(end, num_prompt_tokens=prompt)
    assert masks[3] == []  # Private ring has no external objects.
    return {
        (group_id, block_hash)
        for group_id in coord.cacheable_group_ids
        for index, block_hash in enumerate(hashes(end // BLOCK_SIZE))
        if start // BLOCK_SIZE <= index and masks[group_id][index]
    }


def lookup(coord, pool, request_hashes):
    def query(group_id, group_hashes, mask):
        assert group_id != 3
        return [
            block_hash
            for index, block_hash in enumerate(group_hashes)
            if (mask is None or mask[index]) and (group_id, block_hash) in pool
        ]

    return coord.find_reachable_hit_tokens(request_hashes, len(request_hashes) * BLOCK_SIZE, query)


@pytest.mark.parametrize("interval,expected", [(0, 0), (2048, 14336), (4096, 12288), (None, 14720)])
def test_shared_prefix_recovers_only_at_complete_saved_window(interval, expected):
    coord = make_coordinator(interval)
    assert lookup(coord, save(coord), changed_suffix()) == expected


def test_periodic_checkpoints_keep_exact_prompt_hit_and_sparse_swa_storage():
    coord = make_coordinator(2048)
    pool = save(coord)
    assert lookup(coord, pool, hashes()) == PROMPT_TOKENS
    # Eight four-block windows plus one extra block for prompt-1 replay.
    assert sum(group_id == 2 for group_id, _ in pool) == 33


@pytest.mark.parametrize("missing_group,missing_index", [(0, 100), (1, 100), (2, 111)])
def test_incomplete_main_summary_or_swa_falls_back(missing_group, missing_index):
    coord = make_coordinator(2048)
    pool = save(coord)
    pool.remove((missing_group, hashes()[missing_index]))
    assert lookup(coord, pool, changed_suffix()) == 12288


def test_no_complete_window_returns_zero():
    coord = make_coordinator(2048)
    pool = {(group_id, block_hash) for group_id, block_hash in save(coord) if group_id != 2}
    assert lookup(coord, pool, changed_suffix()) == 0


def test_chunked_prefill_saves_window_pieces_before_later_chunks_recycle_them():
    coord = make_coordinator(2048)
    pool = set()
    start = 0
    # Several checkpoints straddle chunk boundaries, including the 14336 one.
    for end in (1920, 2176, 3968, 8448, 14080, 14464, PROMPT_TOKENS):
        pool.update(save(coord, start=start, end=end))
        start = end
    assert pool == save(coord)
    assert lookup(coord, pool, changed_suffix()) == 14336


def test_lookup_also_recovers_nonperiodic_replay_checkpoint():
    coord = make_coordinator(2048)
    pool = save(coord, end=14208, prompt=14209)
    assert lookup(coord, pool, changed_suffix()) == 14208


@pytest.mark.parametrize(
    "prompt_tokens,interval,expected,use_eagle",
    [
        (2048, 2048, 0, False),
        (4096, 2048, 2048, False),
        (2049, 2048, 2048, False),
        (16384, 2048, 16256, False),
        (16385, 2048, 16384, False),
        (2048, None, 1920, False),
        (128, 2048, None, False),
        (2048, None, 1920, True),
        (2049, None, 2048, True),
    ],
)
def test_private_state_lookup_checks_the_window_at_the_actual_resume_boundary(
    prompt_tokens, interval, expected, use_eagle
):
    coord = make_coordinator(interval, use_eagle=use_eagle)
    pool = save(coord)
    scheduler = object.__new__(LayerwiseKVPoolScheduler)
    scheduler.use_block_key_layerwise = False
    scheduler.use_eagle = use_eagle
    scheduler.use_layerwise_transfer = True
    scheduler.cacheable_group_ids = coord.cacheable_group_ids
    scheduler.original_block_size = [BLOCK_SIZE] * 4
    scheduler.cache_transfer_granularity = BLOCK_SIZE
    request = SimpleNamespace(prompt_token_ids=[0] * prompt_tokens, block_hashes=hashes())

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
