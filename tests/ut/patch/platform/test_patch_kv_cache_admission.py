# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    FullAttentionManager,
    SlidingWindowManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_ascend.patch.platform import patch_kv_cache_admission  # noqa: F401


def _manager(kind="sliding", cap=8197, enable_caching=True, block_pool=None):
    register_all_kvcache_specs(None)
    kwargs = dict(block_size=2, num_kv_heads=1, head_size=8, dtype=torch.float16)
    if kind == "sliding":
        cls = SlidingWindowManager
        spec = SlidingWindowSpec(sliding_window=8, **kwargs)
    elif kind == "chunked":
        cls = ChunkedLocalAttentionManager
        spec = ChunkedLocalAttentionSpec(attention_chunk_size=8, **kwargs)
    else:
        cls = FullAttentionManager
        spec = FullAttentionSpec(**kwargs)
    if block_pool is None:
        null = KVCacheBlock(block_id=0)
        null.is_null = True
        block_pool = SimpleNamespace(null_block=null)
    return cls(
        kv_cache_spec=spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=2,
        max_admission_blocks_per_request=cap,
    )


def _required(manager, *, tokens=113484, computed=102400, new_blocks=(), local=0, capped=True):
    return manager.get_num_blocks_to_allocate(
        "request",
        tokens,
        new_blocks,
        computed,
        local,
        tokens,
        apply_admission_cap=capped,
    )


def _loaded_prefix(manager, skipped=51196, held=4):
    manager.req_to_blocks["request"] = [manager._null_block] * skipped + [
        KVCacheBlock(block_id=i + 1, ref_cnt=1) for i in range(held)
    ]
    if manager.enable_caching:
        manager.num_cached_block["request"] = skipped + held


@pytest.mark.parametrize("kind", ["sliding", "chunked"])
def test_loaded_prefix_reserves_next_chunk(kind):
    manager = _manager(kind)
    _loaded_prefix(manager)
    # The stalled request has 102400 cached tokens and 11084 left.
    # The 51196 null entries consume no physical capacity.
    assert _required(manager) == 5542
    assert _required(manager, tokens=200000) == 8193
    assert _required(manager, tokens=110592, capped=False) == 4096


@pytest.mark.parametrize("kind, expected", [("sliding", 5546), ("chunked", 5542)])
def test_remote_hit_reserves_window_and_remaining_tokens(kind, expected):
    assert _required(_manager(kind)) == expected


@pytest.mark.parametrize("kind", ["sliding", "chunked"])
@pytest.mark.parametrize("ref_cnt", [0, 1, 2])
def test_local_hit_counts_only_evictable_blocks_inside_window(kind, ref_cnt):
    manager = _manager(kind)
    blocks = [manager._null_block] * 51196 + [KVCacheBlock(block_id=i + 1, ref_cnt=ref_cnt) for i in range(4)]
    expected = 5546 if kind == "sliding" and ref_cnt == 0 else 5542
    assert _required(manager, new_blocks=blocks, local=102400) == expected


@pytest.mark.parametrize("kind", ["sliding", "chunked"])
@pytest.mark.parametrize("sharing", ["exclusive", "shared", "mixed"])
def test_shared_held_blocks_do_not_offset_admission_capacity(kind, sharing):
    # Use real allocations and prefix-hit references: a shared block returns
    # to the pool only after both requests release it.
    num_pool_blocks = 64
    cap = 10
    computed = 206
    pool = BlockPool(num_gpu_blocks=num_pool_blocks, enable_caching=True, hash_block_size=2)
    manager = _manager(kind, cap=cap, block_pool=pool)
    manager.add_local_computed_blocks("request", [], 0, computed)
    manager.allocate_external_computed_blocks("request", 0, computed)
    blocks = manager.req_to_blocks["request"]
    loaded_blocks = [block for block in blocks if not block.is_null]
    assert len(blocks) > cap > len(loaded_blocks) > 0
    exclusive_blocks = len(loaded_blocks)
    if sharing != "exclusive":
        manager.add_local_computed_blocks("other", list(blocks), computed, 0)
        assert all(block.ref_cnt == 2 for block in loaded_blocks)
        exclusive_blocks = 0
        if sharing == "mixed":
            exclusive_blocks = 2
            allocated_tokens = computed + exclusive_blocks * manager.block_size
            manager.allocate_new_blocks("request", allocated_tokens, allocated_tokens)
            assert all(block.ref_cnt == 1 for block in blocks[-exclusive_blocks:])

    target_tokens = 400
    uncapped = target_tokens // manager.block_size - len(blocks)
    assert _required(manager, tokens=target_tokens, computed=computed, capped=False) == uncapped
    assert _required(manager, tokens=target_tokens, computed=computed) == cap - exclusive_blocks
    # A short tail still needs only one block, even when all held blocks are
    # shared. Rejected speculative tokens need no additional allocation.
    tail_tokens = (len(blocks) + 1) * manager.block_size
    assert _required(manager, tokens=tail_tokens, computed=computed) == 1
    rejected_tokens = (len(blocks) - 1) * manager.block_size
    assert _required(manager, tokens=rejected_tokens, computed=computed) == 0
    manager.max_admission_blocks_per_request = exclusive_blocks
    assert _required(manager, tokens=target_tokens, computed=computed) == 0

    free_before = pool.get_num_free_blocks()
    manager.free("request")
    assert pool.get_num_free_blocks() == free_before + exclusive_blocks
    if sharing != "exclusive":
        assert all(block.ref_cnt == 1 for block in loaded_blocks)
        manager.free("other")
    assert pool.get_num_free_blocks() == num_pool_blocks - 1


@pytest.mark.parametrize("kind", ["sliding", "chunked"])
@pytest.mark.parametrize("enable_caching", [True, False])
def test_async_inflight_blocks_remain_physically_held(kind, enable_caching):
    manager = _manager(kind, enable_caching=enable_caching)
    _loaded_prefix(manager, held=4100)
    assert ("request" in manager.num_cached_block) == enable_caching
    # Optimistic computed tokens include a still-running 8192-token chunk.
    # Its blocks cannot be treated as freed until the worker completes it.
    assert _required(manager, tokens=200000, computed=110592) == 4097


@pytest.mark.parametrize("skipped, held, expected", [(0, 0, 8197), (0, 4, 8193), (51200, 0, 8197)])
@pytest.mark.parametrize("kind", ["sliding", "chunked"])
@pytest.mark.parametrize("enable_caching", [True, False])
def test_null_prefix_boundaries(skipped, held, expected, kind, enable_caching):
    manager = _manager(kind, enable_caching=enable_caching)
    _loaded_prefix(manager, skipped=skipped, held=held)
    computed = (skipped + held) * manager.block_size
    assert _required(manager, tokens=200000, computed=computed) == expected


@pytest.mark.parametrize("kind", ["sliding", "chunked"])
def test_uncached_inflight_prefill_reserves_only_remaining_capacity(kind):
    # A cold prefill takes the ordinary allocation path, which does not create
    # a num_cached_block entry when prefix caching is disabled.
    max_model_len = 128
    chunk_tokens = 16
    num_blocks = 256
    spec = _manager(kind).kv_cache_spec
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer"], kv_cache_spec=spec)],
    )
    cache = KVCacheManager(
        config,
        max_model_len=max_model_len,
        scheduler_block_size=spec.block_size,
        hash_block_size=spec.block_size,
        max_in_flight_tokens=2 * chunk_tokens,
        enable_caching=False,
    )
    request = Request("request", [1] * max_model_len, SamplingParams(max_tokens=1), None)
    assert cache.allocate_slots(request, num_new_tokens=chunk_tokens) is not None
    manager = cache.coordinator.single_type_managers[0]
    assert request.request_id not in manager.num_cached_block
    blocks = manager.req_to_blocks[request.request_id]
    assert len(blocks) == chunk_tokens // spec.block_size
    assert all(not block.is_null for block in blocks)

    # Scheduling advances this count before the worker finishes the chunk.
    # All of its blocks are still held and must count against the cap.
    request.num_computed_tokens = chunk_tokens
    request.num_in_flight_tokens = chunk_tokens
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_cache_manager = cache
    scheduler.max_model_len = max_model_len
    scheduler.num_spec_tokens = 0
    scheduler._inflight_prefills = {request}
    expected = manager.max_admission_blocks_per_request - len(blocks)
    assert scheduler._inflight_prefill_reserved_blocks() == expected


def test_uncapped_and_full_attention_preserve_logical_accounting():
    manager = _manager(cap=None)
    _loaded_prefix(manager)
    assert _required(manager, tokens=200000) == 48800
    full = _manager("full", cap=None)
    _loaded_prefix(full, skipped=0, held=51200)
    assert _required(full, tokens=200000) == 48800


def test_rejected_speculative_tokens_need_no_new_blocks():
    manager = _manager()
    _loaded_prefix(manager)
    assert _required(manager, tokens=102398) == 0


def test_async_remote_admission_leaves_capacity_to_resume_prefill():
    # Reproduce the six shared-pool groups and chunk size from the stalled
    # engine. Use real allocation and scheduler reservation accounting.
    common = dict(num_kv_heads=1, head_size=8, dtype=torch.float16)
    specs = [
        FullAttentionSpec(block_size=128, **common),
        FullAttentionSpec(block_size=4096, **common),
        SlidingWindowSpec(block_size=32, sliding_window=128, **common),
        SlidingWindowSpec(block_size=32, sliding_window=128, **common),
        SlidingWindowSpec(block_size=2, sliding_window=8, **common),
        SlidingWindowSpec(block_size=8, sliding_window=128, **common),
    ]
    register_all_kvcache_specs(None)
    config = KVCacheConfig(
        num_blocks=13724,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[str(i)], kv_cache_spec=spec) for i, spec in enumerate(specs)],
    )
    cache = KVCacheManager(
        config,
        max_model_len=200000,
        scheduler_block_size=4096,
        hash_block_size=32,
        max_in_flight_tokens=16384,
        enable_caching=False,
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_cache_manager = cache
    scheduler.max_model_len = 200000
    scheduler.num_spec_tokens = 0
    scheduler._inflight_prefills = set()
    admitted = []
    for i in range(16):
        request = Request(str(i), [1] * 113484, SamplingParams(max_tokens=1), None)
        blocks = cache.allocate_slots(
            request,
            num_new_tokens=0,
            num_external_computed_tokens=102400,
            delay_cache_blocks=True,
            full_sequence_must_fit=True,
            reserved_blocks=scheduler._inflight_prefill_reserved_blocks(),
        )
        if blocks is None:
            break
        request.num_computed_tokens = 102400
        scheduler._inflight_prefills.add(request)
        admitted.append(request)

    assert admitted
    # At least one loaded request must be able to run. With logical-length
    # subtraction, async loads pin the pool and this allocation returns None.
    resumed = cache.allocate_slots(
        admitted[0], num_new_tokens=8192, full_sequence_must_fit=True, delay_cache_blocks=True
    )
    assert resumed is not None
    assert sum(len(group) for group in resumed.blocks) == 5698


@pytest.mark.parametrize("capped, expected", [(True, 8193), (False, 48800)])
def test_prefill_end_argument_preserves_allocation_semantics(capped, expected):
    manager = _manager()
    _loaded_prefix(manager)
    assert (
        manager.get_num_blocks_to_allocate(
            "request", 200000, (), 102400, 0, 200000, apply_admission_cap=capped, prefill_end=200000
        )
        == expected
    )
