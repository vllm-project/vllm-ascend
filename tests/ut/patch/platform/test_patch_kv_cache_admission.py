# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
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


def _manager(kind="sliding", cap=8197):
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
    null = KVCacheBlock(block_id=0)
    null.is_null = True
    return cls(
        kv_cache_spec=spec,
        block_pool=SimpleNamespace(null_block=null),
        enable_caching=True,
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


def test_local_hit_counts_only_evictable_blocks_inside_window():
    manager = _manager()
    blocks = [manager._null_block] * 51196 + [KVCacheBlock(block_id=i + 1) for i in range(4)]
    assert _required(manager, new_blocks=blocks, local=102400) == 5546
    for block in blocks[-4:]:
        block.ref_cnt = 1
    assert _required(manager, new_blocks=blocks, local=102400) == 5542


def test_async_inflight_blocks_remain_physically_held():
    manager = _manager()
    _loaded_prefix(manager, held=4100)
    # Optimistic computed tokens include a still-running 8192-token chunk.
    # Its blocks cannot be treated as freed until the worker completes it.
    assert _required(manager, tokens=200000, computed=110592) == 4097


@pytest.mark.parametrize("skipped, held, expected", [(0, 0, 8197), (0, 4, 8193), (51200, 0, 8197)])
def test_null_prefix_boundaries(skipped, held, expected):
    manager = _manager()
    _loaded_prefix(manager, skipped=skipped, held=held)
    assert _required(manager, tokens=200000) == expected


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
